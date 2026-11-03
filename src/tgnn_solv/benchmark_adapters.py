"""Formal adapter API for benchmarking arbitrary custom models on repo splits."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd

from .external_benchmarking import (
    BenchmarkArtifacts,
    build_benchmark_artifacts,
    clip_ln_x2_for_logS,
    ln_x2_from_logS,
    logS_from_ln_x2,
    merge_prediction_frame,
    prepare_pair_dataframe,
    write_benchmark_artifacts,
)


@dataclass(slots=True)
class BenchmarkAdapterContext:
    """Context passed to adapter implementations."""

    train_df: pd.DataFrame | None
    val_df: pd.DataFrame | None
    test_df: pd.DataFrame
    out_dir: Path


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Minimal contract for repo-native custom benchmarking."""

    def describe(self) -> Mapping[str, Any]:
        """Return metadata about the adapter and model family."""

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Optional train step before prediction."""

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe containing pair keys and predictions."""

    def save_state(self, out_dir: Path) -> Mapping[str, Any] | None:
        """Optional hook to export adapter-specific state."""


class BaseBenchmarkAdapter:
    """Convenience base class for adapters that only need `predict_frame`."""

    def describe(self) -> Mapping[str, Any]:
        return {
            "adapter_name": self.__class__.__name__,
            "model_family": "custom_adapter",
            "supports_fit": hasattr(self, "fit"),
        }

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        return None

    def save_state(self, out_dir: Path) -> Mapping[str, Any] | None:
        return None


def _parse_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    return dict(json.loads(value))


def load_adapter(ref: str, *, init_kwargs: Mapping[str, Any] | None = None) -> BenchmarkAdapter:
    """Load an adapter from `module:ClassOrFactory`."""
    if ":" not in ref:
        raise ValueError("Adapter reference must be in `module:object` form.")
    module_name, object_name = ref.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    kwargs = dict(init_kwargs or {})
    instance = obj(**kwargs) if callable(obj) else obj
    if not isinstance(instance, BenchmarkAdapter):
        missing = [name for name in ("describe", "fit", "predict_frame") if not hasattr(instance, name)]
        raise TypeError(
            f"Adapter `{ref}` does not satisfy the benchmark adapter contract. "
            f"Missing: {', '.join(missing) or 'unknown'}."
        )
    return instance


def _safe_pred_logS(predictions: pd.DataFrame, pred_ln_x2_col: str) -> pd.Series:
    clipped = clip_ln_x2_for_logS(
        pd.to_numeric(predictions[pred_ln_x2_col], errors="coerce")
    )
    return logS_from_ln_x2(
        pd.DataFrame(
            {
                "solvent_smiles": predictions["solvent_smiles"],
                "ln_x2": clipped,
            }
        ),
        ln_x2_col="ln_x2",
    )


def _normalize_predictions(
    base_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    pred_lnx2_col: str | None,
    pred_logs_col: str | None,
) -> tuple[pd.DataFrame, str, str]:
    normalized = merge_prediction_frame(base_df, predictions_df, prefer_row_index=True)
    if pred_lnx2_col and pred_lnx2_col not in normalized.columns:
        raise ValueError(f"Adapter predictions are missing `{pred_lnx2_col}`.")
    if pred_logs_col and pred_logs_col not in normalized.columns:
        raise ValueError(f"Adapter predictions are missing `{pred_logs_col}`.")
    if pred_lnx2_col is None and pred_logs_col is None:
        raise ValueError("At least one prediction column must be provided.")

    if pred_lnx2_col is None and pred_logs_col is not None:
        normalized["ln_x2_pred"] = ln_x2_from_logS(
            pd.DataFrame(
                {
                    "solvent_smiles": base_df["solvent_smiles"],
                    "logS": normalized[pred_logs_col],
                }
            ),
            logS_col="logS",
        )
        pred_lnx2_col = "ln_x2_pred"
    if pred_logs_col is None and pred_lnx2_col is not None:
        normalized["logS_pred"] = _safe_pred_logS(
            normalized if "solvent_smiles" in normalized.columns else base_df.assign(**normalized),
            pred_lnx2_col,
        )
        pred_logs_col = "logS_pred"
    return normalized, pred_lnx2_col, pred_logs_col


def run_adapter_benchmark(
    *,
    adapter_ref: str,
    test_data: str | Path,
    out_dir: str | Path,
    train_data: str | Path | None = None,
    val_data: str | Path | None = None,
    init_kwargs: Mapping[str, Any] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    predict_kwargs: Mapping[str, Any] | None = None,
    model_name: str | None = None,
    pred_lnx2_col: str | None = "ln_x2_pred",
    pred_logs_col: str | None = None,
    uncertainty_col: str | None = None,
) -> BenchmarkArtifacts:
    """Load a Python adapter, optionally fit it, and benchmark it on repo data."""
    adapter = load_adapter(adapter_ref, init_kwargs=init_kwargs)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    test_df = prepare_pair_dataframe(pd.read_csv(test_data), require_targets=True)
    train_df = prepare_pair_dataframe(pd.read_csv(train_data), require_targets=True) if train_data else None
    val_df = prepare_pair_dataframe(pd.read_csv(val_data), require_targets=True) if val_data else None

    if train_df is not None:
        adapter.fit(train_df, val_df, **dict(fit_kwargs or {}))

    raw_predictions = adapter.predict_frame(test_df.copy(), **dict(predict_kwargs or {}))
    merged_predictions, pred_lnx2_col, pred_logs_col = _normalize_predictions(
        test_df,
        raw_predictions,
        pred_lnx2_col=pred_lnx2_col,
        pred_logs_col=pred_logs_col,
    )
    uncertainty = None
    if uncertainty_col and uncertainty_col in merged_predictions.columns:
        uncertainty = pd.to_numeric(merged_predictions[uncertainty_col], errors="coerce").to_numpy(dtype=float)

    descriptor = dict(adapter.describe() or {})
    family = str(descriptor.get("model_family") or "custom_adapter")
    adapter_state = adapter.save_state(out_root)
    metadata = {
        **descriptor,
        "adapter_ref": adapter_ref,
        "model_family": family,
        "native_training": train_df is not None,
        "train_data": str(train_data) if train_data else None,
        "val_data": str(val_data) if val_data else None,
        "test_data": str(test_data),
    }
    if adapter_state:
        metadata["adapter_state"] = dict(adapter_state)

    artifacts = build_benchmark_artifacts(
        model_name=model_name or str(descriptor.get("model_name") or descriptor.get("adapter_name") or "custom_adapter"),
        eval_df=test_df,
        pred_ln_x2=pd.to_numeric(merged_predictions[pred_lnx2_col], errors="coerce").to_numpy(dtype=float),
        pred_logS=pd.to_numeric(merged_predictions[pred_logs_col], errors="coerce").to_numpy(dtype=float),
        uncertainty=uncertainty,
        metadata=metadata,
        split_mode="custom_adapter",
        test_data=str(test_data),
    )
    write_benchmark_artifacts(
        out_root,
        artifacts,
        input_paths={
            "train_data": str(train_data) if train_data else None,
            "val_data": str(val_data) if val_data else None,
            "test_data": str(test_data),
        },
        command=["python", "-m", "tgnn_solv.benchmark_adapters", adapter_ref],
    )
    (out_root / "adapter_description.json").write_text(
        json.dumps({"describe": descriptor, "state": adapter_state}, indent=2),
        encoding="utf-8",
    )
    return artifacts


__all__ = [
    "BaseBenchmarkAdapter",
    "BenchmarkAdapter",
    "BenchmarkAdapterContext",
    "load_adapter",
    "run_adapter_benchmark",
    "_parse_json_arg",
]
