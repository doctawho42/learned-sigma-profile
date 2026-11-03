"""Helpers for extracting finite-composition activity-side ThermoML data."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .thermoml_idac import (
    _as_list,
    _compound_lookup,
    _extract_nested_label,
    _extract_org_num,
    doi_to_json_url,
)

ACTIVITY_COEFF_PROP = "Activity coefficient"
REL_ACTIVITY_PROP = "(Relative) activity"
HEXCESS_PROP = "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol"
HMIX_SOLVENT_PROP = "Molar enthalpy of mixing with solvent, kJ/mol"

PROPERTY_SPECS = {
    ACTIVITY_COEFF_PROP: {
        "property_key": "activity_coefficient",
        "measurement_kind": "direct_activity",
        "canonical_unit": "gamma",
        "scale": 1.0,
        "require_positive": True,
        "abs_max": 1_000_000.0,
        "log_value": True,
    },
    REL_ACTIVITY_PROP: {
        "property_key": "relative_activity",
        "measurement_kind": "direct_activity",
        "canonical_unit": "activity",
        "scale": 1.0,
        "require_positive": True,
        "abs_max": 100.0,
        "log_value": True,
    },
    HEXCESS_PROP: {
        "property_key": "excess_molar_enthalpy",
        "measurement_kind": "excess_thermo",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    HMIX_SOLVENT_PROP: {
        "property_key": "molar_enthalpy_mixing_with_solvent",
        "measurement_kind": "excess_thermo",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
}

_COMPOSITION_BASIS_ORDER = {
    "mole_fraction": 0,
    "molality_mol_per_kg": 1,
    "mass_fraction": 2,
    "mass_ratio_solute_to_solvent": 3,
}


def _labels(node: Any, key: str) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for current_key, value in node.items():
            if current_key == key and isinstance(value, (str, int, float)):
                out.append(str(value))
            out.extend(_labels(value, key))
    elif isinstance(node, list):
        for value in node:
            out.extend(_labels(value, key))
    return out


def _first_label(node: Any, *keys: str) -> str | None:
    for key in keys:
        labels = _labels(node, key)
        for label in labels:
            stripped = label.strip()
            if stripped:
                return stripped
    return None


def _first_numeric(node: Any, *keys: str) -> float | None:
    for key in keys:
        for label in _labels(node, key):
            try:
                value = float(label)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _normalize_composition_basis(label: str | None) -> str | None:
    if not isinstance(label, str):
        return None
    lowered = label.lower()
    if "mole fraction" in lowered:
        return "mole_fraction"
    if "molality" in lowered:
        return "molality_mol_per_kg"
    if "mass fraction" in lowered:
        return "mass_fraction"
    if "mass ratio of solute to solvent" in lowered:
        return "mass_ratio_solute_to_solvent"
    return None


def _state_var_numbers(
    variables: list[dict[str, Any]],
    *,
    token: str,
) -> set[int]:
    var_numbers: set[int] = set()
    for variable in variables:
        if not isinstance(variable, dict):
            continue
        variable_id = variable.get("VariableID", {})
        label = _extract_nested_label(variable_id.get("VariableType"))
        if not isinstance(label, str) or token not in label.lower():
            continue
        try:
            var_numbers.add(int(variable.get("nVarNumber")))
        except (TypeError, ValueError):
            continue
    return var_numbers


def _state_from_num_values(
    num_values: dict[str, Any],
    *,
    variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    token: str,
) -> float | None:
    var_numbers = _state_var_numbers(variables, token=token)

    for variable_value in _as_list(num_values.get("VariableValue")):
        try:
            var_number = int(variable_value.get("nVarNumber"))
            value = float(variable_value.get("nVarValue"))
        except (TypeError, ValueError):
            continue
        if var_number in var_numbers and math.isfinite(value):
            return value

    for constraint in constraints:
        label = _extract_nested_label(constraint.get("ConstraintID", {}).get("ConstraintType"))
        if isinstance(label, str) and token in label.lower():
            try:
                value = float(constraint.get("nConstraintValue"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def _composition_var_lookup(variables: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for variable in variables:
        if not isinstance(variable, dict):
            continue
        variable_id = variable.get("VariableID", {})
        label = _extract_nested_label(variable_id.get("VariableType"))
        basis = _normalize_composition_basis(label)
        if basis is None:
            continue
        try:
            var_number = int(variable.get("nVarNumber"))
        except (TypeError, ValueError):
            continue
        lookup[var_number] = {
            "composition_basis": basis,
            "composition_label": label,
            "composition_org_num": _extract_org_num(variable_id),
            "source": "variable",
        }
    return lookup


def _composition_entries(
    num_values: dict[str, Any],
    *,
    constraints: list[dict[str, Any]],
    composition_var_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for variable_value in _as_list(num_values.get("VariableValue")):
        try:
            var_number = int(variable_value.get("nVarNumber"))
            value = float(variable_value.get("nVarValue"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        metadata = composition_var_lookup.get(var_number)
        if metadata is None:
            continue
        entries.append(
            {
                **metadata,
                "value": value,
            }
        )

    for constraint in constraints:
        constraint_id = constraint.get("ConstraintID", {})
        label = _extract_nested_label(constraint_id.get("ConstraintType"))
        basis = _normalize_composition_basis(label)
        if basis is None:
            continue
        try:
            value = float(constraint.get("nConstraintValue"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        entries.append(
            {
                "composition_basis": basis,
                "composition_label": label,
                "composition_org_num": _extract_org_num(constraint_id),
                "source": "constraint",
                "value": value,
            }
        )

    return entries


def _select_composition_entry(
    entries: list[dict[str, Any]],
    *,
    target_org_num: int | None,
) -> dict[str, Any] | None:
    if not entries:
        return None

    def sort_key(entry: dict[str, Any]) -> tuple[int, int, int, int]:
        composition_org_num = entry.get("composition_org_num")
        basis = str(entry.get("composition_basis"))
        source = str(entry.get("source"))
        target_rank = 0 if target_org_num is not None and composition_org_num == target_org_num else 1
        source_rank = 0 if source == "variable" else 1
        basis_rank = _COMPOSITION_BASIS_ORDER.get(basis, 99)
        has_component_rank = 0 if composition_org_num is not None else 1
        return (target_rank, source_rank, basis_rank, has_component_rank)

    return sorted(entries, key=sort_key)[0]


def _property_metadata(property_node: dict[str, Any]) -> dict[str, Any] | None:
    method_block = property_node.get("Property-MethodID", {})
    property_group = method_block.get("PropertyGroup", {})
    prop_name = _first_label(property_group, "ePropName")
    if prop_name not in PROPERTY_SPECS:
        return None
    spec = PROPERTY_SPECS[prop_name]
    return {
        "prop_number": property_node.get("nPropNumber"),
        "property_name": prop_name,
        "property_key": spec["property_key"],
        "measurement_kind": spec["measurement_kind"],
        "canonical_unit": spec["canonical_unit"],
        "scale": float(spec["scale"]),
        "require_positive": bool(spec["require_positive"]),
        "abs_max": float(spec["abs_max"]),
        "log_value": bool(spec["log_value"]),
        "target_org_num": _extract_org_num(method_block),
        "method": _first_label(property_group, "sMethodName", "eMethodName"),
        "phase": _first_label(property_node.get("PropPhaseID", {}), "ePropPhase"),
        "standard_state": property_node.get("eStandardState"),
        "uncertainty": _first_numeric(
            property_node,
            "nCombUncertValue",
            "nStdUncertValue",
            "nExpandUncertValue",
            "nCombExpandUncertValue",
        ),
    }


def _is_idac_like(
    metadata: dict[str, Any],
    *,
    composition_entries: list[dict[str, Any]],
) -> bool:
    standard_state = str(metadata.get("standard_state") or "").lower()
    if "infinite dilution" in standard_state:
        return True
    for entry in composition_entries:
        value = entry.get("value")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and abs(numeric) <= 1.0e-12:
            return True
    return False


def extract_activity_measurements(
    record: dict[str, Any],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    """
    Extract finite-composition activity-side rows from one ThermoML JSON record.

    Scope:
      - finite-composition `Activity coefficient`
      - finite-composition `(Relative) activity`
      - excess enthalpy / mixing-with-solvent pair properties

    Infinite-dilution activity-coefficient rows are intentionally excluded here
    because they are already handled by the dedicated ThermoML IDAC path.
    """

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

        component_records = [compounds.get(org_num, {}) for org_num in component_ids]
        component_smiles = [record.get("smiles") for record in component_records]
        if any(not isinstance(smiles, str) or not smiles for smiles in component_smiles):
            continue
        pair_smiles_sorted = sorted(component_smiles)
        pair_key_sorted = f"{pair_smiles_sorted[0]}>>{pair_smiles_sorted[1]}"

        constraints = _as_list(dataset.get("Constraint"))
        variables = _as_list(dataset.get("Variable"))
        composition_var_lookup = _composition_var_lookup(variables)

        for property_node in _as_list(dataset.get("Property")):
            metadata = _property_metadata(property_node)
            if metadata is None:
                continue

            target_org_num = metadata["target_org_num"]
            if target_org_num is not None and target_org_num not in component_ids:
                continue
            target = compounds.get(target_org_num, {}) if target_org_num is not None else {}
            other_org_num = next(
                (org_num for org_num in component_ids if org_num != target_org_num),
                None,
            )
            other = compounds.get(other_org_num, {}) if other_org_num is not None else {}

            for num_values in _as_list(dataset.get("NumValues")):
                composition_entries = _composition_entries(
                    num_values,
                    constraints=constraints,
                    composition_var_lookup=composition_var_lookup,
                )
                selected_composition = _select_composition_entry(
                    composition_entries,
                    target_org_num=target_org_num,
                )
                if selected_composition is None:
                    continue
                try:
                    composition_value = float(selected_composition["value"])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(composition_value) or abs(composition_value) <= 1.0e-12:
                    continue

                is_idac_like = _is_idac_like(
                    metadata,
                    composition_entries=composition_entries,
                )
                if metadata["measurement_kind"] == "direct_activity" and is_idac_like:
                    continue

                temperature = _state_from_num_values(
                    num_values,
                    variables=variables,
                    constraints=constraints,
                    token="temperature",
                )
                pressure = _state_from_num_values(
                    num_values,
                    variables=variables,
                    constraints=constraints,
                    token="pressure",
                )

                for property_value in _as_list(num_values.get("PropertyValue")):
                    try:
                        prop_number = int(property_value.get("nPropNumber"))
                        target_prop_number = int(metadata["prop_number"])
                    except (TypeError, ValueError):
                        continue
                    if prop_number != target_prop_number:
                        continue

                    try:
                        raw_value = float(property_value.get("nPropValue"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(raw_value):
                        continue

                    value = raw_value * float(metadata["scale"])
                    if not math.isfinite(value):
                        continue
                    if metadata["require_positive"] and value <= 0:
                        continue
                    if abs(value) > float(metadata["abs_max"]):
                        continue

                    composition_org_num = selected_composition.get("composition_org_num")
                    composition_component = compounds.get(composition_org_num, {})
                    target_smiles = target.get("smiles")
                    other_smiles = other.get("smiles")
                    targeted_pair_key = (
                        f"{target_smiles}>>{other_smiles}"
                        if isinstance(target_smiles, str)
                        and target_smiles
                        and isinstance(other_smiles, str)
                        and other_smiles
                        else None
                    )
                    reverse_targeted_pair_key = (
                        f"{other_smiles}>>{target_smiles}"
                        if targeted_pair_key is not None
                        else None
                    )

                    uncertainty = _first_numeric(
                        property_value,
                        "nCombUncertValue",
                        "nStdUncertValue",
                        "nExpandUncertValue",
                        "nCombExpandUncertValue",
                    )
                    if uncertainty is None:
                        uncertainty = metadata["uncertainty"]
                    if uncertainty is not None and metadata["canonical_unit"] == "J/mol":
                        uncertainty = float(uncertainty) * float(metadata["scale"])

                    ln_value = math.log(value) if metadata["log_value"] and value > 0 else None
                    rows.append(
                        {
                            "source_label": source_label,
                            "doi": citation.get("sDOI"),
                            "json_url": (
                                doi_to_json_url(citation["sDOI"])
                                if isinstance(citation.get("sDOI"), str)
                                else None
                            ),
                            "title": citation.get("sTitle"),
                            "journal": citation.get("sPubName"),
                            "year": citation.get("yrPubYr"),
                            "dataset_number": dataset.get("nPureOrMixtureDataNumber"),
                            "prop_number": prop_number,
                            "property_name": metadata["property_name"],
                            "property_key": metadata["property_key"],
                            "measurement_kind": metadata["measurement_kind"],
                            "method": metadata["method"],
                            "phase": metadata["phase"],
                            "standard_state": metadata["standard_state"],
                            "canonical_unit": metadata["canonical_unit"],
                            "temperature": temperature,
                            "pressure_kpa": pressure,
                            "value": value,
                            "raw_value": raw_value,
                            "ln_value": ln_value,
                            "uncertainty": uncertainty,
                            "pair_component_smiles_a": pair_smiles_sorted[0],
                            "pair_component_smiles_b": pair_smiles_sorted[1],
                            "pair_key_sorted": pair_key_sorted,
                            "target_smiles": target_smiles,
                            "target_name": target.get("name"),
                            "other_smiles": other_smiles,
                            "other_name": other.get("name"),
                            "targeted_pair_key": targeted_pair_key,
                            "reverse_targeted_pair_key": reverse_targeted_pair_key,
                            "composition_basis": selected_composition.get("composition_basis"),
                            "composition_label": selected_composition.get("composition_label"),
                            "composition_source": selected_composition.get("source"),
                            "composition_value": composition_value,
                            "composition_smiles": composition_component.get("smiles"),
                            "composition_name": composition_component.get("name"),
                            "is_direct_activity": metadata["measurement_kind"] == "direct_activity",
                            "is_idac_like": is_idac_like,
                        }
                    )

    return rows


def measurements_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert extracted activity-side rows to a stable dataframe shape."""
    columns = [
        "source_label",
        "doi",
        "json_url",
        "title",
        "journal",
        "year",
        "dataset_number",
        "prop_number",
        "property_name",
        "property_key",
        "measurement_kind",
        "method",
        "phase",
        "standard_state",
        "canonical_unit",
        "temperature",
        "pressure_kpa",
        "value",
        "raw_value",
        "ln_value",
        "uncertainty",
        "pair_component_smiles_a",
        "pair_component_smiles_b",
        "pair_key_sorted",
        "target_smiles",
        "target_name",
        "other_smiles",
        "other_name",
        "targeted_pair_key",
        "reverse_targeted_pair_key",
        "composition_basis",
        "composition_label",
        "composition_source",
        "composition_value",
        "composition_smiles",
        "composition_name",
        "is_direct_activity",
        "is_idac_like",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns].copy()


def aggregate_activity_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Exact-state aggregation for ThermoML activity-side rows.

    The grouping keys keep pair identity, targeted component, composition
    setting, and thermodynamic state explicit so that the result remains a
    supervision artifact, not a lossy pair-only summary.
    """
    columns = [
        "property_key",
        "measurement_kind",
        "pair_component_smiles_a",
        "pair_component_smiles_b",
        "pair_key_sorted",
        "target_smiles",
        "target_name",
        "other_smiles",
        "other_name",
        "targeted_pair_key",
        "reverse_targeted_pair_key",
        "composition_basis",
        "composition_label",
        "composition_source",
        "composition_value",
        "composition_smiles",
        "composition_name",
        "temperature",
        "pressure_kpa",
        "phase",
        "standard_state",
        "canonical_unit",
        "method",
        "value",
        "value_min",
        "value_max",
        "value_std",
        "ln_value",
        "ln_value_min",
        "ln_value_max",
        "ln_value_std",
        "uncertainty_median",
        "n_measurements",
        "n_dois",
        "available_dois",
        "example_doi",
        "is_direct_activity",
        "is_idac_like",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    work = frame.copy()
    numeric_columns = ("temperature", "pressure_kpa", "composition_value", "value", "ln_value", "uncertainty")
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["temperature_key"] = work["temperature"].round(6)
    work["pressure_key"] = work["pressure_kpa"].round(6)
    work["composition_key"] = work["composition_value"].round(6)

    group_columns = [
        "property_key",
        "measurement_kind",
        "pair_component_smiles_a",
        "pair_component_smiles_b",
        "pair_key_sorted",
        "target_smiles",
        "target_name",
        "other_smiles",
        "other_name",
        "targeted_pair_key",
        "reverse_targeted_pair_key",
        "composition_basis",
        "composition_label",
        "composition_source",
        "composition_key",
        "composition_smiles",
        "composition_name",
        "temperature_key",
        "pressure_key",
        "phase",
        "standard_state",
        "canonical_unit",
        "method",
        "is_direct_activity",
        "is_idac_like",
    ]

    rows: list[dict[str, Any]] = []
    for _, group in work.groupby(group_columns, dropna=False, sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        ln_values = pd.to_numeric(group["ln_value"], errors="coerce").dropna()
        uncertainties = pd.to_numeric(group["uncertainty"], errors="coerce").dropna()
        doi_values = sorted(
            {
                str(value).strip()
                for value in group["doi"].dropna().tolist()
                if str(value).strip()
            }
        )
        exemplar = group.iloc[0]
        rows.append(
            {
                "property_key": exemplar["property_key"],
                "measurement_kind": exemplar["measurement_kind"],
                "pair_component_smiles_a": exemplar["pair_component_smiles_a"],
                "pair_component_smiles_b": exemplar["pair_component_smiles_b"],
                "pair_key_sorted": exemplar["pair_key_sorted"],
                "target_smiles": exemplar["target_smiles"],
                "target_name": exemplar["target_name"],
                "other_smiles": exemplar["other_smiles"],
                "other_name": exemplar["other_name"],
                "targeted_pair_key": exemplar["targeted_pair_key"],
                "reverse_targeted_pair_key": exemplar["reverse_targeted_pair_key"],
                "composition_basis": exemplar["composition_basis"],
                "composition_label": exemplar["composition_label"],
                "composition_source": exemplar["composition_source"],
                "composition_value": exemplar["composition_key"],
                "composition_smiles": exemplar["composition_smiles"],
                "composition_name": exemplar["composition_name"],
                "temperature": exemplar["temperature_key"],
                "pressure_kpa": exemplar["pressure_key"],
                "phase": exemplar["phase"],
                "standard_state": exemplar["standard_state"],
                "canonical_unit": exemplar["canonical_unit"],
                "method": exemplar["method"],
                "value": float(values.median()),
                "value_min": float(values.min()),
                "value_max": float(values.max()),
                "value_std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                "ln_value": float(ln_values.median()) if not ln_values.empty else pd.NA,
                "ln_value_min": float(ln_values.min()) if not ln_values.empty else pd.NA,
                "ln_value_max": float(ln_values.max()) if not ln_values.empty else pd.NA,
                "ln_value_std": (
                    float(ln_values.std(ddof=0)) if len(ln_values) > 1 else 0.0
                )
                if not ln_values.empty
                else pd.NA,
                "uncertainty_median": float(uncertainties.median()) if not uncertainties.empty else pd.NA,
                "n_measurements": int(len(values)),
                "n_dois": int(len(doi_values)),
                "available_dois": ";".join(doi_values),
                "example_doi": doi_values[0] if doi_values else pd.NA,
                "is_direct_activity": bool(exemplar["is_direct_activity"]),
                "is_idac_like": bool(exemplar["is_idac_like"]),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns].copy()
