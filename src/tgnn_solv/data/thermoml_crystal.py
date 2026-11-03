"""Helpers for extracting pure-component crystal properties from ThermoML JSON."""

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

TM_PROP_NAME = "Normal melting temperature, K"
DH_PROP_NAME = "Molar enthalpy of transition or fusion, kJ/mol"

PROPERTY_SPECS = {
    TM_PROP_NAME: {
        "property_key": "T_m",
        "canonical_unit": "K",
        "scale": 1.0,
        "min_value": 50.0,
        "max_value": 1_500.0,
    },
    DH_PROP_NAME: {
        "property_key": "dH_fus",
        "canonical_unit": "J/mol",
        "scale": 1_000.0,
        "min_value": 100.0,
        "max_value": 200_000.0,
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


def _property_metadata(property_node: dict[str, Any]) -> dict[str, Any] | None:
    property_group = property_node.get("Property-MethodID", {}).get("PropertyGroup", {})
    prop_name = _first_label(property_group, "ePropName")
    if prop_name not in PROPERTY_SPECS:
        return None
    spec = PROPERTY_SPECS[prop_name]
    return {
        "prop_number": property_node.get("nPropNumber"),
        "property_name": prop_name,
        "property_key": spec["property_key"],
        "canonical_unit": spec["canonical_unit"],
        "scale": float(spec["scale"]),
        "min_value": float(spec["min_value"]),
        "max_value": float(spec["max_value"]),
        "method": _first_label(property_group, "sMethodName", "eMethodName"),
        "phase": _first_label(property_node.get("PropPhaseID", {}), "ePropPhase"),
        "uncertainty": _first_numeric(
            property_node,
            "nCombUncertValue",
            "nStdUncertValue",
            "nExpandUncertValue",
        ),
    }


def extract_crystal_measurements(
    record: dict[str, Any],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    """Extract pure-component T_m and dH_fus measurements from one ThermoML record."""
    citation = record.get("Citation", {})
    compounds = _compound_lookup(record)

    rows: list[dict[str, Any]] = []
    for dataset in _as_list(record.get("PureOrMixtureData")):
        component_ids = [
            org_num
            for component in _as_list(dataset.get("Component"))
            if (org_num := _extract_org_num(component)) is not None
        ]
        if len(component_ids) != 1:
            continue
        solute_org_num = component_ids[0]
        compound = compounds.get(solute_org_num, {})
        solute_smiles = compound.get("smiles")
        if not isinstance(solute_smiles, str) or not solute_smiles:
            continue

        property_metadata: dict[int, dict[str, Any]] = {}
        for property_node in _as_list(dataset.get("Property")):
            metadata = _property_metadata(property_node)
            if metadata is None:
                continue
            try:
                prop_number = int(metadata["prop_number"])
            except (TypeError, ValueError):
                continue
            property_metadata[prop_number] = metadata

        if not property_metadata:
            continue

        for num_values in _as_list(dataset.get("NumValues")):
            for property_value in _as_list(num_values.get("PropertyValue")):
                try:
                    prop_number = int(property_value.get("nPropNumber"))
                except (TypeError, ValueError):
                    continue
                metadata = property_metadata.get(prop_number)
                if metadata is None:
                    continue
                try:
                    raw_value = float(property_value.get("nPropValue"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(raw_value):
                    continue

                value = raw_value * float(metadata["scale"])
                if not (metadata["min_value"] <= value <= metadata["max_value"]):
                    continue

                uncertainty = _first_numeric(
                    property_value,
                    "nCombUncertValue",
                    "nStdUncertValue",
                    "nExpandUncertValue",
                )
                if uncertainty is None:
                    uncertainty = metadata["uncertainty"]
                # Scale the uncertainty into the same unit as ``value`` (kJ/mol ->
                # J/mol for dH_fus), mirroring thermoml_activity.py. Previously it
                # was stored in the raw unit -> a 1000x mismatch vs the J/mol value
                # column. Use `is None` (not `or`) so a reported uncertainty of
                # exactly 0.0 is not silently replaced by the property-level fallback.
                if uncertainty is not None and metadata["canonical_unit"] == "J/mol":
                    uncertainty = float(uncertainty) * float(metadata["scale"])

                row = {
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
                    "solute_smiles": solute_smiles,
                    "solute_name": compound.get("name"),
                    "formula": compound.get("formula"),
                    "property_name": metadata["property_name"],
                    "property_key": metadata["property_key"],
                    "method": metadata["method"],
                    "phase": metadata["phase"],
                    "value": value,
                    "raw_value": raw_value,
                    "canonical_unit": metadata["canonical_unit"],
                    "uncertainty": uncertainty,
                    "T_m": value if metadata["property_key"] == "T_m" else None,
                    "dH_fus": value if metadata["property_key"] == "dH_fus" else None,
                }
                rows.append(row)

    return rows


def measurements_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert extracted rows to a stable dataframe shape."""
    columns = [
        "source_label",
        "doi",
        "json_url",
        "title",
        "journal",
        "year",
        "dataset_number",
        "prop_number",
        "solute_smiles",
        "solute_name",
        "formula",
        "property_name",
        "property_key",
        "method",
        "phase",
        "value",
        "raw_value",
        "canonical_unit",
        "uncertainty",
        "T_m",
        "dH_fus",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns].copy()


def _aggregate_property(
    frame: pd.DataFrame,
    *,
    property_key: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["solute_smiles"])

    value_column = "T_m" if property_key == "T_m" else "dH_fus"
    subset = frame.loc[frame["property_key"] == property_key].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=["solute_smiles", value_column])
    if subset.empty:
        return pd.DataFrame(columns=["solute_smiles"])

    rows: list[dict[str, Any]] = []
    for solute_smiles, group in subset.groupby("solute_smiles", sort=True):
        values = pd.to_numeric(group[value_column], errors="coerce").dropna()
        values = values[values.map(math.isfinite)]
        if values.empty:
            continue
        doi_values = sorted(
            {
                str(value).strip()
                for value in group["doi"].dropna().tolist()
                if str(value).strip()
            }
        )
        row = {
            "solute_smiles": solute_smiles,
            "solute_name": group["solute_name"].dropna().astype(str).iloc[0]
            if group["solute_name"].notna().any()
            else None,
            "formula": group["formula"].dropna().astype(str).iloc[0]
            if group["formula"].notna().any()
            else None,
            value_column: float(values.median()),
            f"{value_column}_source": "thermoml",
            f"{value_column}_protocol": (
                "thermoml_median_same_doi_tm_paired"
                if property_key == "dH_fus"
                else "thermoml_median_all_measurements"
            ),
            f"{value_column}_n_measurements": int(len(values)),
            f"{value_column}_n_dois": int(len(doi_values)),
            f"{value_column}_example_doi": doi_values[0] if doi_values else None,
            f"{value_column}_std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
            f"{value_column}_min": float(values.min()),
            f"{value_column}_max": float(values.max()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_crystal_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ThermoML measurements to one solute-level record.

    `dH_fus` is kept only from DOI groups that also contain a melting-point
    measurement for the same solute, which removes many ambiguous
    transition-only records from the more weakly named enthalpy field.
    """
    frame = measurements_to_frame(frame.to_dict("records") if isinstance(frame, pd.DataFrame) else frame)
    if frame.empty:
        return pd.DataFrame(columns=["solute_smiles", "T_m", "dH_fus"])

    grouped = (
        frame.groupby(["solute_smiles", "doi"])["property_key"]
        .agg(lambda series: tuple(sorted(set(series))))
        .reset_index(name="property_keys")
    )
    paired_groups = {
        (row.solute_smiles, row.doi)
        for row in grouped.itertuples(index=False)
        if {"T_m", "dH_fus"}.issubset(set(row.property_keys))
    }

    tm_frame = _aggregate_property(frame, property_key="T_m")

    dh_frame = frame.loc[
        frame.apply(
            lambda row: (row["solute_smiles"], row["doi"]) in paired_groups
            if pd.notna(row["solute_smiles"])
            else False,
            axis=1,
        )
    ].copy()
    dh_frame = _aggregate_property(dh_frame, property_key="dH_fus")

    if tm_frame.empty and dh_frame.empty:
        return pd.DataFrame(columns=["solute_smiles", "T_m", "dH_fus"])
    if tm_frame.empty:
        result = dh_frame
    elif dh_frame.empty:
        result = tm_frame
    else:
        result = tm_frame.merge(
            dh_frame,
            on=["solute_smiles"],
            how="outer",
            suffixes=("", "__dh"),
        )
        for column in ("solute_name", "formula"):
            alt = f"{column}__dh"
            if alt in result.columns:
                result[column] = result[column].fillna(result[alt])
                result = result.drop(columns=[alt])

    result["thermoml_pair_doi_count"] = result["solute_smiles"].map(
        lambda smiles: sum(1 for group in paired_groups if group[0] == smiles)
    )
    return result.sort_values("solute_smiles").reset_index(drop=True)
