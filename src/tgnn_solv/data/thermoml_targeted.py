"""Helpers for exact SLE-pair coverage audits over ThermoML JSON records."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from .thermoml_activity import (
    _composition_entries,
    _composition_var_lookup,
    _first_numeric,
    _is_idac_like,
    _select_composition_entry,
    _state_from_num_values,
)
from .thermoml_idac import (
    _as_list,
    _compound_lookup,
    _extract_nested_label,
    _extract_org_num,
    doi_to_json_url,
)


DEFAULT_OUTPUT_COLUMNS = [
    "doi",
    "source_title",
    "source_label",
    "dataset_number",
    "pair_component_smiles_a",
    "pair_component_smiles_b",
    "pair_component_name_a",
    "pair_component_name_b",
    "pair_key_sorted",
    "property_label",
    "property_family",
    "property_method",
    "property_phase",
    "property_standard_state",
    "property_target_smiles",
    "property_target_name",
    "dataset_variable_labels",
    "dataset_constraint_labels",
    "dataset_has_temperature",
    "dataset_has_pressure",
    "dataset_has_composition",
]

TARGETED_MEASUREMENT_COLUMNS = [
    "source_label",
    "doi",
    "json_url",
    "title",
    "journal",
    "year",
    "dataset_number",
    "prop_number",
    "property_label",
    "property_key",
    "property_family",
    "property_method",
    "property_phase",
    "property_standard_state",
    "reported_unit",
    "canonical_unit",
    "temperature",
    "pressure_kpa",
    "value",
    "raw_value",
    "ln_value",
    "uncertainty",
    "pair_component_smiles_a",
    "pair_component_smiles_b",
    "pair_component_name_a",
    "pair_component_name_b",
    "pair_key_sorted",
    "property_target_smiles",
    "property_target_name",
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
    "is_candidate_activity_signal",
    "is_idac_like",
]

ACTIVITY_SIGNAL_CANDIDATE_FAMILIES = frozenset(
    {"direct_activity", "excess_thermo", "solution_thermo", "vle_like"}
)

_PROPERTY_VALUE_SPECS: dict[str, dict[str, Any]] = {
    "Activity coefficient": {
        "property_key": "activity_coefficient",
        "canonical_unit": "gamma",
        "scale": 1.0,
        "require_positive": True,
        "abs_max": 1_000_000.0,
        "log_value": True,
    },
    "(Relative) activity": {
        "property_key": "relative_activity",
        "canonical_unit": "activity",
        "scale": 1.0,
        "require_positive": True,
        "abs_max": 100.0,
        "log_value": True,
    },
    "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol": {
        "property_key": "excess_molar_enthalpy",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Molar enthalpy of mixing with solvent, kJ/mol": {
        "property_key": "molar_enthalpy_mixing_with_solvent",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Molar enthalpy of dilution, kJ/mol": {
        "property_key": "molar_enthalpy_of_dilution",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Molar enthalpy of solution, kJ/mol": {
        "property_key": "molar_enthalpy_of_solution",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Osmotic coefficient": {
        "property_key": "osmotic_coefficient",
        "canonical_unit": "dimensionless",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Mole fraction": {
        "property_key": "mole_fraction",
        "canonical_unit": "fraction",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 1.0,
        "log_value": False,
    },
    "Mass fraction": {
        "property_key": "mass_fraction",
        "canonical_unit": "fraction",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 1.0,
        "log_value": False,
    },
    "Molality, mol/kg": {
        "property_key": "molality",
        "canonical_unit": "mol/kg",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Mass ratio of solute to solvent": {
        "property_key": "mass_ratio_solute_to_solvent",
        "canonical_unit": "ratio",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 1_000_000.0,
        "log_value": False,
    },
    "Boiling temperature at pressure P, K": {
        "property_key": "boiling_temperature",
        "canonical_unit": "K",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 10_000.0,
        "log_value": False,
    },
    "Solid-liquid equilibrium temperature, K": {
        "property_key": "solid_liquid_equilibrium_temperature",
        "canonical_unit": "K",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 10_000.0,
        "log_value": False,
    },
    "Eutectic temperature, K": {
        "property_key": "eutectic_temperature",
        "canonical_unit": "K",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 10_000.0,
        "log_value": False,
    },
    "Triple point temperature, K": {
        "property_key": "triple_point_temperature",
        "canonical_unit": "K",
        "scale": 1.0,
        "require_positive": False,
        "abs_max": 10_000.0,
        "log_value": False,
    },
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


def _unique_join(values: list[str]) -> str:
    clean = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    return "|".join(dict.fromkeys(clean))


def _slugify_property_label(label: str | None) -> str:
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return normalized or "unknown"


def _reported_unit(property_label: str | None) -> str | None:
    if not isinstance(property_label, str):
        return None
    if "," in property_label:
        tail = property_label.rsplit(",", maxsplit=1)[-1].strip()
        if tail:
            return tail
    if property_label in {
        "Activity coefficient",
        "(Relative) activity",
        "Osmotic coefficient",
        "Mole fraction",
        "Mass fraction",
        "Mass ratio of solute to solvent",
    }:
        return "dimensionless"
    return None


def _value_spec(property_label: str | None) -> dict[str, Any]:
    if not isinstance(property_label, str):
        return {
            "property_key": "unknown",
            "canonical_unit": None,
            "scale": 1.0,
            "require_positive": False,
            "abs_max": 1_000_000_000_000.0,
            "log_value": False,
        }
    spec = dict(
        _PROPERTY_VALUE_SPECS.get(
            property_label,
            {
                "property_key": _slugify_property_label(property_label),
                "canonical_unit": _reported_unit(property_label),
                "scale": 1.0,
                "require_positive": False,
                "abs_max": 1_000_000_000_000.0,
                "log_value": False,
            },
        )
    )
    if not spec.get("property_key"):
        spec["property_key"] = _slugify_property_label(property_label)
    return spec


def classify_property_family(label: str | None) -> str:
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    lowered = label.lower()
    if label in {"Activity coefficient", "(Relative) activity"}:
        return "direct_activity"
    if (
        "osmotic coefficient" in lowered
        or "enthalpy of solution" in lowered
        or "enthalpy of dilution" in lowered
    ):
        return "solution_thermo"
    if "excess" in lowered or "mixing" in lowered:
        return "excess_thermo"
    if (
        "vapor" in lowered
        or "sublimation pressure" in lowered
        or "boiling temperature" in lowered
        or "partial pressure" in lowered
        or "azeotropic" in lowered
    ):
        return "vle_like"
    if (
        "melting" in lowered
        or "solid-liquid equilibrium" in lowered
        or "transition or fusion" in lowered
        or "enthalpy of fusion" in lowered
        or "eutectic" in lowered
    ):
        return "crystal_like"
    if "heat capacity" in lowered:
        return "heat_capacity"
    if (
        "mole fraction" in lowered
        or "mass fraction" in lowered
        or "molality" in lowered
        or "mass ratio of solute to solvent" in lowered
    ):
        return "composition_like"
    return "other"


def _dataset_components(
    dataset: dict[str, Any],
    *,
    compounds: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    for component in _as_list(dataset.get("Component")):
        org_num = _extract_org_num(component)
        if org_num is None:
            continue
        compound = compounds.get(org_num)
        if compound is None:
            continue
        smiles = compound.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        if smiles in seen_smiles:
            continue
        seen_smiles.add(smiles)
        components.append(
            {
                "org_num": org_num,
                "smiles": smiles,
                "name": compound.get("name"),
            }
        )
    return components


def _dataset_label_summary(dataset: dict[str, Any]) -> dict[str, object]:
    variable_labels = (
        _labels(dataset.get("Variable"), "eVariableType")
        + _labels(dataset.get("Variable"), "eTemperature")
        + _labels(dataset.get("Variable"), "eComponentComposition")
        + _labels(dataset.get("Variable"), "ePressure")
    )
    constraint_labels = (
        _labels(dataset.get("Constraint"), "eConstraintType")
        + _labels(dataset.get("Constraint"), "eTemperature")
        + _labels(dataset.get("Constraint"), "eComponentComposition")
        + _labels(dataset.get("Constraint"), "ePressure")
    )
    combined = [label.lower() for label in [*variable_labels, *constraint_labels] if isinstance(label, str)]
    return {
        "dataset_variable_labels": _unique_join(variable_labels),
        "dataset_constraint_labels": _unique_join(constraint_labels),
        "dataset_has_temperature": any("temperature" in label for label in combined),
        "dataset_has_pressure": any("pressure" in label for label in combined),
        "dataset_has_composition": any(
            token in label
            for label in combined
            for token in ("mole fraction", "mass fraction", "molality", "composition")
        ),
    }


def extract_targeted_pair_rows(
    record: dict[str, Any],
    *,
    source_label: str,
    target_pairs: set[str] | None = None,
) -> list[dict[str, object]]:
    """Extract one row per ThermoML property for binary pairs of interest."""
    citation = record.get("Citation", {}) if isinstance(record, dict) else {}
    compounds = _compound_lookup(record)
    rows: list[dict[str, object]] = []

    for dataset in _as_list(record.get("PureOrMixtureData")):
        if not isinstance(dataset, dict):
            continue
        components = _dataset_components(dataset, compounds=compounds)
        if len(components) != 2:
            continue
        pair_smiles = sorted(component["smiles"] for component in components)
        pair_key_sorted = ">>".join(pair_smiles)
        if target_pairs is not None and pair_key_sorted not in target_pairs:
            continue

        component_by_smiles = {component["smiles"]: component for component in components}
        label_summary = _dataset_label_summary(dataset)

        for prop in _as_list(dataset.get("Property")):
            if not isinstance(prop, dict):
                continue
            property_labels = list(dict.fromkeys(_labels(prop, "ePropName") or ["<none>"]))
            property_method = _unique_join(_labels(prop, "eMethodName"))
            property_phase = _unique_join(_labels(prop, "ePropPhase"))
            property_standard_state = prop.get("eStandardState")
            property_target = compounds.get(_extract_org_num(prop.get("Property-MethodID", {})), {})
            property_target_smiles = property_target.get("smiles")
            property_target_name = property_target.get("name")

            for property_label in property_labels:
                rows.append(
                    {
                        "doi": citation.get("sDOI"),
                        "source_title": citation.get("sTitle"),
                        "source_label": source_label,
                        "dataset_number": dataset.get("nPureOrMixtureDataNumber"),
                        "pair_component_smiles_a": pair_smiles[0],
                        "pair_component_smiles_b": pair_smiles[1],
                        "pair_component_name_a": component_by_smiles[pair_smiles[0]].get("name"),
                        "pair_component_name_b": component_by_smiles[pair_smiles[1]].get("name"),
                        "pair_key_sorted": pair_key_sorted,
                        "property_label": property_label,
                        "property_family": classify_property_family(property_label),
                        "property_method": property_method or None,
                        "property_phase": property_phase or None,
                        "property_standard_state": property_standard_state,
                        "property_target_smiles": property_target_smiles,
                        "property_target_name": property_target_name,
                        **label_summary,
                    }
                )
    return rows


def measurements_to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DEFAULT_OUTPUT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in DEFAULT_OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[DEFAULT_OUTPUT_COLUMNS].copy()


def extract_targeted_measurement_rows(
    record: dict[str, Any],
    *,
    source_label: str,
    target_pairs: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract exact-pair ThermoML measurements for binary pairs of interest."""
    citation = record.get("Citation", {}) if isinstance(record, dict) else {}
    compounds = _compound_lookup(record)
    rows: list[dict[str, Any]] = []

    for dataset in _as_list(record.get("PureOrMixtureData")):
        if not isinstance(dataset, dict):
            continue
        components = _dataset_components(dataset, compounds=compounds)
        if len(components) != 2:
            continue

        pair_smiles = sorted(component["smiles"] for component in components)
        pair_key_sorted = ">>".join(pair_smiles)
        if target_pairs is not None and pair_key_sorted not in target_pairs:
            continue

        component_by_smiles = {component["smiles"]: component for component in components}
        component_by_org_num = {
            int(component["org_num"]): component
            for component in components
            if component.get("org_num") is not None
        }
        constraints = _as_list(dataset.get("Constraint"))
        variables = _as_list(dataset.get("Variable"))
        composition_var_lookup = _composition_var_lookup(variables)

        for property_node in _as_list(dataset.get("Property")):
            if not isinstance(property_node, dict):
                continue

            property_labels = list(dict.fromkeys(_labels(property_node, "ePropName") or ["<none>"]))
            try:
                prop_number = int(property_node.get("nPropNumber"))
            except (TypeError, ValueError):
                continue

            method_block = property_node.get("Property-MethodID", {})
            target_org_num = _extract_org_num(method_block)
            if target_org_num is not None and target_org_num not in component_by_org_num:
                continue

            target_component = component_by_org_num.get(target_org_num, {})
            other_component = next(
                (
                    component
                    for component in components
                    if component.get("org_num") != target_org_num
                ),
                {},
            )
            property_method = _unique_join(_labels(property_node, "eMethodName")) or None
            property_phase = _unique_join(_labels(property_node, "ePropPhase")) or None
            property_standard_state = property_node.get("eStandardState")
            property_uncertainty = _first_numeric(
                property_node,
                "nCombUncertValue",
                "nStdUncertValue",
                "nExpandUncertValue",
                "nCombExpandUncertValue",
            )

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
                composition_component = (
                    compounds.get(selected_composition.get("composition_org_num"), {})
                    if selected_composition is not None
                    else {}
                )

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
                        value_prop_number = int(property_value.get("nPropNumber"))
                    except (TypeError, ValueError):
                        continue
                    if value_prop_number != prop_number:
                        continue

                    try:
                        raw_value = float(property_value.get("nPropValue"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(raw_value):
                        continue

                    raw_uncertainty = _first_numeric(
                        property_value,
                        "nCombUncertValue",
                        "nStdUncertValue",
                        "nExpandUncertValue",
                        "nCombExpandUncertValue",
                    )

                    for property_label in property_labels:
                        spec = _value_spec(property_label)
                        property_family = classify_property_family(property_label)
                        is_candidate_activity_signal = (
                            property_family in ACTIVITY_SIGNAL_CANDIDATE_FAMILIES
                        )
                        is_idac_like = False
                        if property_family == "direct_activity":
                            is_idac_like = _is_idac_like(
                                {"standard_state": property_standard_state},
                                composition_entries=composition_entries,
                            )
                            if is_idac_like:
                                continue

                        value = raw_value * float(spec["scale"])
                        if not math.isfinite(value):
                            continue
                        if bool(spec["require_positive"]) and value <= 0:
                            continue
                        if abs(value) > float(spec["abs_max"]):
                            continue

                        uncertainty = raw_uncertainty
                        if uncertainty is None:
                            uncertainty = property_uncertainty
                        if uncertainty is not None and float(spec["scale"]) != 1.0:
                            uncertainty = float(uncertainty) * float(spec["scale"])

                        property_target_smiles = target_component.get("smiles")
                        other_smiles = other_component.get("smiles")
                        targeted_pair_key = (
                            f"{property_target_smiles}>>{other_smiles}"
                            if isinstance(property_target_smiles, str)
                            and property_target_smiles
                            and isinstance(other_smiles, str)
                            and other_smiles
                            else None
                        )
                        reverse_targeted_pair_key = (
                            f"{other_smiles}>>{property_target_smiles}"
                            if targeted_pair_key is not None
                            else None
                        )
                        ln_value = (
                            float(math.log(value))
                            if bool(spec["log_value"]) and value > 0
                            else None
                        )
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
                                "property_label": property_label,
                                "property_key": spec["property_key"],
                                "property_family": property_family,
                                "property_method": property_method,
                                "property_phase": property_phase,
                                "property_standard_state": property_standard_state,
                                "reported_unit": _reported_unit(property_label),
                                "canonical_unit": spec["canonical_unit"],
                                "temperature": temperature,
                                "pressure_kpa": pressure,
                                "value": value,
                                "raw_value": raw_value,
                                "ln_value": ln_value,
                                "uncertainty": uncertainty,
                                "pair_component_smiles_a": pair_smiles[0],
                                "pair_component_smiles_b": pair_smiles[1],
                                "pair_component_name_a": component_by_smiles[pair_smiles[0]].get("name"),
                                "pair_component_name_b": component_by_smiles[pair_smiles[1]].get("name"),
                                "pair_key_sorted": pair_key_sorted,
                                "property_target_smiles": property_target_smiles,
                                "property_target_name": target_component.get("name"),
                                "other_smiles": other_smiles,
                                "other_name": other_component.get("name"),
                                "targeted_pair_key": targeted_pair_key,
                                "reverse_targeted_pair_key": reverse_targeted_pair_key,
                                "composition_basis": (
                                    selected_composition.get("composition_basis")
                                    if selected_composition is not None
                                    else None
                                ),
                                "composition_label": (
                                    selected_composition.get("composition_label")
                                    if selected_composition is not None
                                    else None
                                ),
                                "composition_source": (
                                    selected_composition.get("source")
                                    if selected_composition is not None
                                    else None
                                ),
                                "composition_value": (
                                    selected_composition.get("value")
                                    if selected_composition is not None
                                    else None
                                ),
                                "composition_smiles": composition_component.get("smiles"),
                                "composition_name": composition_component.get("name"),
                                "is_candidate_activity_signal": is_candidate_activity_signal,
                                "is_idac_like": is_idac_like,
                            }
                        )
    return rows


def measurement_rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TARGETED_MEASUREMENT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in TARGETED_MEASUREMENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[TARGETED_MEASUREMENT_COLUMNS].copy()


def aggregate_targeted_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact-state ThermoML targeted measurements."""
    columns = [
        "property_label",
        "property_key",
        "property_family",
        "pair_component_smiles_a",
        "pair_component_smiles_b",
        "pair_component_name_a",
        "pair_component_name_b",
        "pair_key_sorted",
        "property_target_smiles",
        "property_target_name",
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
        "property_phase",
        "property_standard_state",
        "reported_unit",
        "canonical_unit",
        "property_method",
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
        "is_candidate_activity_signal",
        "is_idac_like",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    work = frame.copy()
    numeric_columns = (
        "temperature",
        "pressure_kpa",
        "composition_value",
        "value",
        "ln_value",
        "uncertainty",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work["temperature_key"] = work["temperature"].round(6)
    work["pressure_key"] = work["pressure_kpa"].round(6)
    work["composition_key"] = work["composition_value"].round(6)

    group_columns = [
        "property_label",
        "property_key",
        "property_family",
        "pair_component_smiles_a",
        "pair_component_smiles_b",
        "pair_component_name_a",
        "pair_component_name_b",
        "pair_key_sorted",
        "property_target_smiles",
        "property_target_name",
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
        "property_phase",
        "property_standard_state",
        "reported_unit",
        "canonical_unit",
        "property_method",
        "is_candidate_activity_signal",
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
                "property_label": exemplar["property_label"],
                "property_key": exemplar["property_key"],
                "property_family": exemplar["property_family"],
                "pair_component_smiles_a": exemplar["pair_component_smiles_a"],
                "pair_component_smiles_b": exemplar["pair_component_smiles_b"],
                "pair_component_name_a": exemplar["pair_component_name_a"],
                "pair_component_name_b": exemplar["pair_component_name_b"],
                "pair_key_sorted": exemplar["pair_key_sorted"],
                "property_target_smiles": exemplar["property_target_smiles"],
                "property_target_name": exemplar["property_target_name"],
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
                "property_phase": exemplar["property_phase"],
                "property_standard_state": exemplar["property_standard_state"],
                "reported_unit": exemplar["reported_unit"],
                "canonical_unit": exemplar["canonical_unit"],
                "property_method": exemplar["property_method"],
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
                "uncertainty_median": (
                    float(uncertainties.median()) if not uncertainties.empty else pd.NA
                ),
                "n_measurements": int(len(values)),
                "n_dois": int(len(doi_values)),
                "available_dois": ";".join(doi_values),
                "example_doi": doi_values[0] if doi_values else pd.NA,
                "is_candidate_activity_signal": bool(exemplar["is_candidate_activity_signal"]),
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
