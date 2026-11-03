"""Shared helpers for experiment-report JSON schemas.

This module provides:
- a canonical report layout for evaluation-style outputs,
- normalization of legacy report payloads,
- JSON-safe conversion helpers for NumPy-heavy metrics payloads.
"""

from __future__ import annotations

import math
from typing import Mapping, TypeAlias

import numpy as np


REPORT_SCHEMA_VERSION = "1.0"
JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _is_mapping(value: object) -> bool:
    """Return True if a value behaves like a mapping."""
    return isinstance(value, Mapping)


def _to_builtin(value: object) -> object:
    """Convert NumPy scalars and arrays into builtin Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: object) -> JSONValue:
    """Recursively convert a structure into JSON-safe builtin types."""
    value = _to_builtin(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (type(None), bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def get_sample_count(metrics: Mapping[str, object] | None) -> int | None:
    """Return the unified sample count from a metric block."""
    if not _is_mapping(metrics):
        return None
    for key in ("n_samples", "n"):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def normalize_metric_block(metrics: Mapping[str, object] | None) -> dict[str, JSONValue]:
    """Normalize one metric block to a canonical shape.

    The helper preserves all existing keys, but ensures:
    - both `n_samples` and `n` are present when a count exists,
    - both `pearson_r` and `pearson` are present when one of them exists.
    """
    if not _is_mapping(metrics):
        return {}

    normalized = {str(key): value for key, value in metrics.items()}
    n_samples = get_sample_count(metrics)
    if n_samples is not None:
        normalized["n_samples"] = n_samples
        normalized["n"] = n_samples

    if "pearson_r" not in normalized and "pearson" in normalized:
        normalized["pearson_r"] = normalized["pearson"]
    if "pearson" not in normalized and "pearson_r" in normalized:
        normalized["pearson"] = normalized["pearson_r"]

    return json_safe(normalized)


def normalize_metric_mapping(
    mapping: Mapping[str, object] | None,
) -> dict[str, dict[str, JSONValue]]:
    """Normalize a mapping of named metric blocks."""
    if not _is_mapping(mapping):
        return {}
    return {
        str(name): normalize_metric_block(metrics)
        for name, metrics in mapping.items()
        if _is_mapping(metrics)
    }


def _select_section(
    payload: Mapping[str, object],
    stratified: Mapping[str, object],
    *aliases: str,
) -> dict[str, dict[str, JSONValue]]:
    """Select and normalize one stratified section from a payload."""
    for alias in aliases:
        candidate = stratified.get(alias)
        if _is_mapping(candidate):
            return normalize_metric_mapping(candidate)
    for alias in aliases:
        candidate = payload.get(alias)
        if _is_mapping(candidate):
            return normalize_metric_mapping(candidate)
    return {}


def normalize_predictions(payload: Mapping[str, object]) -> dict[str, list[JSONValue]]:
    """Extract prediction arrays from either canonical or legacy payloads."""
    predictions = payload.get("predictions", {})
    if not _is_mapping(predictions):
        predictions = {}

    true = predictions.get("true_ln_x2", payload.get("true_ln_x2"))
    pred = predictions.get("pred_ln_x2", payload.get("pred_ln_x2"))
    true_logS = predictions.get("true_logS", payload.get("true_logS"))
    pred_logS = predictions.get("pred_logS", payload.get("pred_logS"))
    row_indices = predictions.get("row_indices", payload.get("row_indices"))
    logS_eval_mask = predictions.get("logS_eval_mask", payload.get("logS_eval_mask"))

    result: dict[str, list[JSONValue]] = {}
    if true is not None:
        result["true_ln_x2"] = json_safe(list(true))
    if pred is not None:
        result["pred_ln_x2"] = json_safe(list(pred))
    if true_logS is not None:
        result["true_logS"] = json_safe(list(true_logS))
    if pred_logS is not None:
        result["pred_logS"] = json_safe(list(pred_logS))
    if row_indices is not None:
        result["row_indices"] = json_safe(list(row_indices))
    if logS_eval_mask is not None:
        result["logS_eval_mask"] = json_safe(list(logS_eval_mask))
    return result


def normalize_report_payload(payload: Mapping[str, object]) -> dict[str, JSONValue]:
    """Normalize evaluation or benchmark payloads to a canonical report view."""
    if not _is_mapping(payload):
        raise ValueError("Expected a report payload mapping.")

    metadata = payload.get("metadata", {})
    if not _is_mapping(metadata):
        metadata = {}
    metadata = {str(key): value for key, value in metadata.items()}

    for key in (
        "model",
        "checkpoint",
        "test_data",
        "config",
        "timestamp",
        "test_samples",
        "n_valid_predictions",
    ):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]

    stratified = payload.get("stratified", {})
    if not _is_mapping(stratified):
        stratified = {}

    temperature = _select_section(payload, stratified, "temperature", "by_temperature", "by_temp")
    solubility = _select_section(payload, stratified, "solubility", "by_solubility", "by_solubility_range", "by_range")
    solvent = _select_section(payload, stratified, "solvent", "by_solvent")
    solvent_type = _select_section(payload, stratified, "solvent_type", "by_solvent_type")
    aux_data = _select_section(payload, stratified, "aux_data", "by_aux_data", "by_aux")

    if not solvent_type and solvent:
        filtered = {
            name: metrics
            for name, metrics in solvent.items()
            if name in {"water", "organic"}
        }
        if filtered:
            solvent_type = filtered

    predictions = normalize_predictions(payload)
    n_valid_predictions = None
    if "true_ln_x2" in predictions and "pred_ln_x2" in predictions:
        n_valid_predictions = min(
            len(predictions["true_ln_x2"]),
            len(predictions["pred_ln_x2"]),
        )
        metadata.setdefault("n_valid_predictions", n_valid_predictions)

    normalized = {
        "schema_version": payload.get("schema_version", REPORT_SCHEMA_VERSION),
        "report_type": payload.get("report_type", "report"),
        "metadata": json_safe(metadata),
        "overall": normalize_metric_block(payload.get("overall", {})),
        "stratified": {
            "temperature": temperature,
            "solubility": solubility,
            "solvent_type": solvent_type,
            "solvent": solvent,
            "aux_data": aux_data,
        },
        "physics_summary": json_safe(payload.get("physics_summary", {})),
        "predictions": predictions,
        "evaluation_subsets": json_safe(payload.get("evaluation_subsets", {})),
    }

    # Compatibility aliases for existing downstream consumers.
    normalized["config"] = normalized["metadata"].get("config")
    normalized["checkpoint"] = normalized["metadata"].get("checkpoint")
    normalized["test_data"] = normalized["metadata"].get("test_data")
    normalized["model"] = normalized["metadata"].get("model")
    normalized["timestamp"] = normalized["metadata"].get("timestamp")
    normalized["test_samples"] = normalized["metadata"].get("test_samples")
    normalized["n_valid_predictions"] = normalized["metadata"].get("n_valid_predictions")
    normalized["split"] = normalized["metadata"].get("split")
    normalized["by_temperature"] = temperature
    normalized["by_solubility"] = solubility
    normalized["by_solubility_range"] = solubility
    normalized["by_solvent_type"] = solvent_type
    normalized["by_solvent"] = solvent
    normalized["by_aux_data"] = aux_data
    normalized["by_aux"] = aux_data
    if "true_ln_x2" in predictions:
        normalized["true_ln_x2"] = predictions["true_ln_x2"]
    if "pred_ln_x2" in predictions:
        normalized["pred_ln_x2"] = predictions["pred_ln_x2"]
    if "row_indices" in predictions:
        normalized["row_indices"] = predictions["row_indices"]
    if "true_logS" in predictions:
        normalized["true_logS"] = predictions["true_logS"]
    if "pred_logS" in predictions:
        normalized["pred_logS"] = predictions["pred_logS"]
    if "logS_eval_mask" in predictions:
        normalized["logS_eval_mask"] = predictions["logS_eval_mask"]

    return json_safe(normalized)


def build_report_payload(
    report_type: str,
    *,
    metadata: Mapping[str, object] | None = None,
    overall: Mapping[str, object] | None = None,
    stratified: Mapping[str, Mapping[str, object]] | None = None,
    predictions: Mapping[str, object] | None = None,
    physics_summary: Mapping[str, object] | None = None,
    evaluation_subsets: Mapping[str, object] | None = None,
) -> dict[str, JSONValue]:
    """Build a canonical report payload with compatibility aliases."""
    metadata = dict(metadata or {})
    stratified = stratified or {}

    def _section(name: str) -> dict[str, object]:
        value = stratified.get(name, {})
        return dict(value) if _is_mapping(value) else {}

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": report_type,
        "metadata": dict(metadata),
        "overall": dict(overall or {}),
        "stratified": {
            "temperature": _section("temperature"),
            "solubility": _section("solubility"),
            "solvent_type": _section("solvent_type"),
            "solvent": _section("solvent"),
            "aux_data": _section("aux_data"),
        },
        "physics_summary": dict(physics_summary or {}),
        "predictions": dict(predictions or {}),
        "evaluation_subsets": dict(evaluation_subsets or {}),
    }
    return normalize_report_payload(payload)
