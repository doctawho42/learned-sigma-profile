"""Helpers for extracting infinite-dilution activity coefficients from ThermoML JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from rdkit import Chem

from .utils import canonicalize

THERMOML_JSON_ROOT = "https://trc.nist.gov/ThermoML"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def doi_to_json_url(doi: str) -> str:
    """Map a DOI to the corresponding NIST ThermoML JSON endpoint."""
    normalized = doi.strip()
    return f"{THERMOML_JSON_ROOT}/{quote(normalized, safe='/+:()')}.json"


def fetch_thermoml_json(
    doi: str,
    *,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    """Fetch one ThermoML JSON record by DOI."""
    url = doi_to_json_url(doi)
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"ThermoML JSON fetch failed for DOI {doi}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"ThermoML JSON fetch failed for DOI {doi}: {exc.reason}") from exc


def load_thermoml_json(path: Path) -> dict[str, Any]:
    """Load one local ThermoML JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    """Normalize optional scalar-or-list ThermoML nodes to a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _extract_org_num(node: Any) -> int | None:
    """Extract the ThermoML compound registry number if present."""
    if not isinstance(node, dict):
        return None
    if "nOrgNum" in node:
        try:
            return int(node["nOrgNum"])
        except (TypeError, ValueError):
            return None
    reg = node.get("RegNum")
    if isinstance(reg, dict):
        try:
            return int(reg.get("nOrgNum"))
        except (TypeError, ValueError):
            return None
    return None


def _first_text(value: Any) -> str | None:
    """Return a representative string from a ThermoML text node."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_nested_label(node: Any) -> str | None:
    """Return the first descriptive string value from a nested ThermoML node."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "tml_elements":
                continue
            label = _extract_nested_label(value)
            if label is not None:
                return label
    return None


def _compound_lookup(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Build a compound lookup by ThermoML org number."""
    lookup: dict[int, dict[str, Any]] = {}
    for fallback_org_num, compound in enumerate(_as_list(record.get("Compound")), start=1):
        org_num = _extract_org_num(compound) or fallback_org_num
        inchi = compound.get("sStandardInChI")
        smiles = None
        if isinstance(inchi, str) and inchi.strip():
            mol = Chem.MolFromInchi(inchi)
            if mol is not None:
                smiles = canonicalize(Chem.MolToSmiles(mol, canonical=True))
        lookup[org_num] = {
            "org_num": org_num,
            "name": _first_text(compound.get("sCommonName")),
            "inchi": inchi,
            "smiles": smiles,
            "formula": compound.get("sFormulaMolec"),
        }
    return lookup


def _property_metadata(property_node: dict[str, Any]) -> dict[str, Any] | None:
    """Return the small subset of property metadata relevant to IDAC extraction."""
    method_block = property_node.get("Property-MethodID", {})
    property_group = method_block.get("PropertyGroup", {})
    activity_block = property_group.get("ActivityFugacityOsmoticProp")
    if not isinstance(activity_block, dict):
        return None
    prop_name = activity_block.get("ePropName")
    if prop_name != "Activity coefficient":
        return None
    return {
        "prop_number": property_node.get("nPropNumber"),
        "solute_org_num": _extract_org_num(method_block),
        "method": activity_block.get("eMethodName"),
        "phase": property_node.get("PropPhaseID", {}).get("ePropPhase"),
        "standard_state": property_node.get("eStandardState"),
    }


def _is_infinite_dilution(
    solute_org_num: int | None,
    constraints: list[dict[str, Any]],
) -> bool:
    """Check whether the dataset constrains the solute mole fraction to zero."""
    if solute_org_num is None:
        return False
    for constraint in constraints:
        constraint_id = constraint.get("ConstraintID", {})
        label = _extract_nested_label(constraint_id.get("ConstraintType"))
        if not isinstance(label, str) or "mole fraction" not in label.lower():
            continue
        if _extract_org_num(constraint_id) != solute_org_num:
            continue
        try:
            value = float(constraint.get("nConstraintValue"))
        except (TypeError, ValueError):
            continue
        if abs(value) <= 1.0e-12:
            return True
    return False


def _solute_mole_fraction_var_numbers(
    solute_org_num: int | None,
    variables: list[dict[str, Any]],
) -> set[int]:
    """Return variable numbers that encode the solute mole fraction."""
    if solute_org_num is None:
        return set()

    var_numbers: set[int] = set()
    for variable in variables:
        if not isinstance(variable, dict):
            continue
        variable_id = variable.get("VariableID", {})
        label = _extract_nested_label(variable_id.get("VariableType"))
        if not isinstance(label, str):
            continue
        lowered = label.lower()
        if "mole fraction" not in lowered and "component composition" not in lowered:
            continue
        if _extract_org_num(variable_id) != solute_org_num:
            continue
        try:
            var_numbers.add(int(variable.get("nVarNumber")))
        except (TypeError, ValueError):
            continue
    return var_numbers


def _num_values_is_infinite_dilution(
    num_values: dict[str, Any],
    *,
    solute_var_numbers: set[int],
) -> bool:
    """Check whether one NumValues block sets the solute mole fraction to zero."""
    if not solute_var_numbers:
        return False

    for variable_value in _as_list(num_values.get("VariableValue")):
        try:
            var_number = int(variable_value.get("nVarNumber"))
            value = float(variable_value.get("nVarValue"))
        except (TypeError, ValueError):
            continue
        if var_number in solute_var_numbers and abs(value) <= 1.0e-12:
            return True
    return False


def _temperature_from_num_values(
    num_values: dict[str, Any],
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
) -> float | None:
    """Resolve the temperature for one NumValues block."""
    temperature_var_numbers = {
        int(variable.get("nVarNumber"))
        for variable in variables
        if isinstance(variable, dict)
        and variable.get("nVarNumber") is not None
        and isinstance(
            _extract_nested_label(variable.get("VariableID", {}).get("VariableType")),
            str,
        )
        and "temperature" in _extract_nested_label(variable.get("VariableID", {}).get("VariableType")).lower()
    }

    for variable_value in _as_list(num_values.get("VariableValue")):
        try:
            var_number = int(variable_value.get("nVarNumber"))
            value = float(variable_value.get("nVarValue"))
        except (TypeError, ValueError):
            continue
        if var_number in temperature_var_numbers:
            return value

    for constraint in constraints:
        label = _extract_nested_label(constraint.get("ConstraintID", {}).get("ConstraintType"))
        if isinstance(label, str) and "temperature" in label.lower():
            try:
                return float(constraint.get("nConstraintValue"))
            except (TypeError, ValueError):
                continue
    return None


def extract_idac_rows(record: dict[str, Any], *, source_label: str) -> list[dict[str, Any]]:
    """Extract IDAC rows from one ThermoML JSON record."""
    citation = record.get("Citation", {})
    compounds = _compound_lookup(record)

    rows: list[dict[str, Any]] = []
    for dataset in _as_list(record.get("PureOrMixtureData")):
        component_ids = [
            org_num
            for component in _as_list(dataset.get("Component"))
            if (org_num := _extract_org_num(component)) is not None
        ]
        if len(component_ids) != 2:
            continue

        constraints = _as_list(dataset.get("Constraint"))
        variables = _as_list(dataset.get("Variable"))

        for property_node in _as_list(dataset.get("Property")):
            metadata = _property_metadata(property_node)
            if metadata is None:
                continue

            solute_org_num = metadata["solute_org_num"]
            if solute_org_num not in component_ids:
                continue
            constrained_idac = _is_infinite_dilution(solute_org_num, constraints)
            solute_var_numbers = _solute_mole_fraction_var_numbers(solute_org_num, variables)
            if not constrained_idac and not solute_var_numbers:
                continue

            solvent_candidates = [
                org_num for org_num in component_ids if org_num != solute_org_num
            ]
            if len(solvent_candidates) != 1:
                continue
            solvent_org_num = solvent_candidates[0]

            for num_values in _as_list(dataset.get("NumValues")):
                if not constrained_idac and not _num_values_is_infinite_dilution(
                    num_values,
                    solute_var_numbers=solute_var_numbers,
                ):
                    continue
                temperature = _temperature_from_num_values(num_values, variables, constraints)
                for property_value in _as_list(num_values.get("PropertyValue")):
                    try:
                        prop_number = int(property_value.get("nPropNumber"))
                        target_prop_number = int(metadata["prop_number"])
                    except (TypeError, ValueError):
                        continue
                    if prop_number != target_prop_number:
                        continue
                    try:
                        gamma_inf = float(property_value.get("nPropValue"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(gamma_inf) or gamma_inf <= 0:
                        continue

                    solute = compounds.get(solute_org_num, {})
                    solvent = compounds.get(solvent_org_num, {})
                    rows.append(
                        {
                            "doi": citation.get("sDOI"),
                            "title": citation.get("sTitle"),
                            "journal": citation.get("sPubName"),
                            "year": citation.get("yrPubYr"),
                            "source_json": source_label,
                            "dataset_number": dataset.get("nPureOrMixtureDataNumber"),
                            "method": metadata["method"],
                            "phase": metadata["phase"],
                            "standard_state": metadata["standard_state"],
                            "solute_name": solute.get("name"),
                            "solvent_name": solvent.get("name"),
                            "solute_inchi": solute.get("inchi"),
                            "solvent_inchi": solvent.get("inchi"),
                            "solute_smiles": solute.get("smiles"),
                            "solvent_smiles": solvent.get("smiles"),
                            "temperature": temperature,
                            "gamma_inf": gamma_inf,
                            "ln_gamma_inf": math.log(gamma_inf),
                        }
                    )
    return rows
