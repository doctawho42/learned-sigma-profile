"""Shared benchmarking helpers for external and custom baseline models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from .artifacts import build_benchmark_card, build_run_manifest, write_json
from .data.sources import _density_map
from .data.split_registry import build_split_metadata
from .data.utils import canonicalize
from .reporting import build_report_payload, json_safe


REQUIRED_PAIR_COLUMNS = {"solute_smiles", "solvent_smiles", "temperature"}
DEFAULT_JOIN_COLUMNS = ("row_index", "solute_smiles", "solvent_smiles", "temperature")
DEFAULT_TEMPERATURE_BINS = (
    (0.0, 298.0, "T_0_to_298K"),
    (298.0, 323.0, "T_298_to_323K"),
    (323.0, 373.0, "T_323_to_373K"),
    (373.0, 500.0, "T_373_to_500K"),
)
MAX_FINITE_X2_FOR_LOGS = 0.999999
MAX_FINITE_LN_X2_FOR_LOGS = float(np.log(MAX_FINITE_X2_FOR_LOGS))


@dataclass(slots=True)
class BenchmarkArtifacts:
    """Small container for canonical benchmark outputs."""

    report: dict[str, Any]
    predictions: pd.DataFrame
    summary: pd.DataFrame


def _ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_pair_dataframe(
    df: pd.DataFrame,
    *,
    require_targets: bool = False,
) -> pd.DataFrame:
    """Canonicalize a pair dataframe to the repo's baseline format."""
    _ensure_required_columns(df, REQUIRED_PAIR_COLUMNS)
    out = df.copy()
    out["solute_smiles"] = out["solute_smiles"].astype(str).map(canonicalize)
    out["solvent_smiles"] = out["solvent_smiles"].astype(str).map(canonicalize)
    out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
    if require_targets:
        if "ln_x2" not in out.columns:
            raise ValueError("Expected `ln_x2` when require_targets=True.")
        out["ln_x2"] = pd.to_numeric(out["ln_x2"], errors="coerce")
    if "has_solubility" in out.columns:
        series = out["has_solubility"]
        if pd.api.types.is_bool_dtype(series):
            out["has_solubility"] = series.fillna(False).astype(bool)
        else:
            normalized = series.fillna(False).astype(str).str.strip().str.lower()
            out["has_solubility"] = normalized.isin({"true", "1", "yes", "y", "t"})
    else:
        out["has_solubility"] = True
    if "row_index" not in out.columns:
        out = out.reset_index(drop=False).rename(columns={"index": "row_index"})
    out = out.dropna(subset=["solute_smiles", "solvent_smiles", "temperature"])
    return out.reset_index(drop=True)


def merge_prediction_frame(
    base_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    prefer_row_index: bool = True,
    required_prediction_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Merge model predictions back onto a canonical pair dataframe.

    The preferred join key is `row_index` because it is stable across filtering.
    When it is not available on both sides, the function falls back to the
    canonical pair identity `(solute_smiles, solvent_smiles, temperature)`.
    """
    base = prepare_pair_dataframe(base_df, require_targets="ln_x2" in base_df.columns)
    preds = predictions_df.copy()
    if required_prediction_cols:
        _ensure_required_columns(preds, required_prediction_cols)

    if prefer_row_index and "row_index" in base.columns and "row_index" in preds.columns:
        merged = base.merge(preds, on="row_index", how="left", suffixes=("", "_pred"))
        if len(merged) == len(base):
            return merged

    join_cols = [col for col in ("solute_smiles", "solvent_smiles", "temperature") if col in preds.columns]
    _ensure_required_columns(preds, join_cols)
    preds = preds.copy()
    preds["solute_smiles"] = preds["solute_smiles"].astype(str).map(canonicalize)
    preds["solvent_smiles"] = preds["solvent_smiles"].astype(str).map(canonicalize)
    preds["temperature"] = pd.to_numeric(preds["temperature"], errors="coerce")
    merged = base.merge(preds, on=join_cols, how="left", suffixes=("", "_pred"))
    return merged


def estimate_solvent_molarity(smiles: str) -> float | None:
    """Estimate solvent molarity (mol/L) from density tables or molecular volume."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    can = canonicalize(smiles) or smiles
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return None

    density_map = _density_map()
    rho = density_map.get(can)
    if rho is not None:
        mw = Descriptors.MolWt(mol)
        if mw > 0:
            return 1000.0 * rho / mw

    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        vol_a3 = rdMolDescriptors.CalcMolVolume(mol_h)
        if not np.isfinite(vol_a3) or vol_a3 <= 0:
            return None
        v_m_cm3 = vol_a3 * 0.602214076
        if v_m_cm3 <= 0 or not np.isfinite(v_m_cm3):
            return None
        return 1000.0 / v_m_cm3
    except Exception:
        return None


def _solvent_molarity_series(solvent_smiles: pd.Series) -> pd.Series:
    cache: dict[str, float | None] = {}
    values: list[float] = []
    for smi in solvent_smiles.astype(str):
        if smi not in cache:
            cache[smi] = estimate_solvent_molarity(smi)
        value = cache[smi]
        values.append(float(value) if value is not None else np.nan)
    return pd.Series(values, index=solvent_smiles.index, dtype=float)


def logS_from_ln_x2(
    df: pd.DataFrame,
    *,
    ln_x2_col: str = "ln_x2",
    solvent_col: str = "solvent_smiles",
) -> pd.Series:
    """Convert ln(x2) mole fractions into log10 solubility in mol/L."""
    if ln_x2_col not in df.columns:
        raise ValueError(f"Missing `{ln_x2_col}` column for conversion.")
    x2 = np.exp(pd.to_numeric(df[ln_x2_col], errors="coerce"))
    c_solvent = _solvent_molarity_series(df[solvent_col])
    with np.errstate(divide="ignore", invalid="ignore"):
        solubility = x2 * c_solvent / (1.0 - x2)
        log_s = np.log10(solubility)
    return pd.Series(log_s, index=df.index, dtype=float)


def clip_ln_x2_for_logS(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Clip ln(x2) so predicted logS conversions stay finite near x2 -> 1.

    This helper is for prediction-side conversions only. It should not be used
    to silently modify ground-truth targets during ingestion.
    """
    return np.minimum(np.asarray(values, dtype=float), MAX_FINITE_LN_X2_FOR_LOGS)


def ln_x2_from_logS(
    df: pd.DataFrame,
    *,
    logS_col: str = "logS",
    solvent_col: str = "solvent_smiles",
) -> pd.Series:
    """Convert log10 solubility in mol/L into ln(x2) mole fractions."""
    if logS_col not in df.columns:
        raise ValueError(f"Missing `{logS_col}` column for conversion.")
    log_s = pd.to_numeric(df[logS_col], errors="coerce")
    solubility = np.power(10.0, log_s)
    c_solvent = _solvent_molarity_series(df[solvent_col])
    with np.errstate(divide="ignore", invalid="ignore"):
        x2 = solubility / (solubility + c_solvent)
        ln_x2 = np.log(x2)
    return pd.Series(ln_x2, index=df.index, dtype=float)


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute finite-only regression metrics."""
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "n_samples": 0,
        }
    y_true = np.asarray(true[mask], dtype=float)
    y_pred = np.asarray(pred[mask], dtype=float)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "pearson_r": pearson_r,
        "n_samples": int(mask.sum()),
    }


def _stratified_metrics(df: pd.DataFrame, pred_col: str, mask: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for low, high, name in DEFAULT_TEMPERATURE_BINS:
        bucket = mask & (df["temperature"].to_numpy(dtype=float) >= low) & (df["temperature"].to_numpy(dtype=float) < high)
        if np.any(bucket):
            out[name] = regression_metrics(
                df.loc[bucket, "ln_x2"].to_numpy(dtype=float),
                df.loc[bucket, pred_col].to_numpy(dtype=float),
            )
    return out


def _solubility_range_metrics(df: pd.DataFrame, pred_col: str, mask: np.ndarray) -> dict[str, dict[str, float]]:
    y_true = df["ln_x2"].to_numpy(dtype=float)
    ranges = [
        (np.nanmin(y_true), -6.0, "very_low_solubility"),
        (-6.0, -3.0, "low_solubility"),
        (-3.0, 0.0, "medium_solubility"),
        (0.0, np.nanmax(y_true) + 1e-9, "high_solubility"),
    ]
    out: dict[str, dict[str, float]] = {}
    for low, high, name in ranges:
        bucket = mask & (y_true >= low) & (y_true < high)
        if np.any(bucket):
            out[name] = regression_metrics(
                df.loc[bucket, "ln_x2"].to_numpy(dtype=float),
                df.loc[bucket, pred_col].to_numpy(dtype=float),
            )
    return out


def _solvent_metrics(df: pd.DataFrame, pred_col: str, mask: np.ndarray) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    solvent_smiles = df["solvent_smiles"].astype(str)
    y_true = df["ln_x2"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    water = canonicalize("O") or "O"
    solvent_type: dict[str, dict[str, float]] = {}
    water_mask = mask & (solvent_smiles == water).to_numpy()
    organic_mask = mask & ~(solvent_smiles == water).to_numpy()
    if np.any(water_mask):
        solvent_type["water"] = regression_metrics(y_true[water_mask], y_pred[water_mask])
    if np.any(organic_mask):
        solvent_type["organic"] = regression_metrics(y_true[organic_mask], y_pred[organic_mask])

    top_solvents = solvent_smiles[mask].value_counts().head(6)
    by_solvent: dict[str, dict[str, float]] = {}
    for smi in top_solvents.index:
        bucket = mask & (solvent_smiles == smi).to_numpy()
        if np.any(bucket):
            by_solvent[str(smi)] = regression_metrics(y_true[bucket], y_pred[bucket])
    return solvent_type, by_solvent


def _aux_data_metrics(df: pd.DataFrame, pred_col: str, mask: np.ndarray) -> dict[str, dict[str, float]]:
    if "has_T_m" not in df.columns:
        return {}
    has_tm = df["has_T_m"].fillna(False).astype(bool).to_numpy()
    y_true = df["ln_x2"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    out: dict[str, dict[str, float]] = {}
    if np.any(mask & has_tm):
        out["with_T_m"] = regression_metrics(y_true[mask & has_tm], y_pred[mask & has_tm])
    if np.any(mask & ~has_tm):
        out["without_T_m"] = regression_metrics(y_true[mask & ~has_tm], y_pred[mask & ~has_tm])
    return out


def build_predictions_frame(
    df: pd.DataFrame,
    *,
    model_name: str,
    pred_ln_x2: np.ndarray,
    true_logS: np.ndarray | None = None,
    pred_logS: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach predictions and basic errors to a dataframe copy."""
    out = df.copy().reset_index(drop=True)
    out["model"] = model_name
    out["ln_x2_pred"] = np.asarray(pred_ln_x2, dtype=float)
    if true_logS is not None:
        out["logS_true"] = np.asarray(true_logS, dtype=float)
    if pred_logS is not None:
        out["logS_pred"] = np.asarray(pred_logS, dtype=float)
    if uncertainty is not None:
        out["prediction_std"] = np.asarray(uncertainty, dtype=float)
    if "ln_x2" in out.columns:
        out["error"] = out["ln_x2_pred"] - pd.to_numeric(out["ln_x2"], errors="coerce")
        out["abs_error"] = np.abs(out["error"])
    if "logS_true" in out.columns and "logS_pred" in out.columns:
        out["logS_error"] = (
            pd.to_numeric(out["logS_pred"], errors="coerce")
            - pd.to_numeric(out["logS_true"], errors="coerce")
        )
        out["logS_abs_error"] = np.abs(out["logS_error"])
    return out


def _pred_logS_from_ln_x2(
    solvent_smiles: pd.Series,
    pred_ln_x2: np.ndarray,
) -> np.ndarray:
    clipped = clip_ln_x2_for_logS(pred_ln_x2)
    pred_log_s = logS_from_ln_x2(
        pd.DataFrame(
            {
                "solvent_smiles": solvent_smiles,
                "ln_x2": clipped,
            }
        ),
        ln_x2_col="ln_x2",
    )
    return pred_log_s.to_numpy(dtype=float)


def _build_evaluation_subsets(predictions_df: pd.DataFrame) -> dict[str, Any]:
    supervision_mask = predictions_df["has_solubility"].fillna(False).astype(bool).to_numpy()
    n_rows = len(predictions_df)
    ln_x2_true = (
        pd.to_numeric(predictions_df["ln_x2"], errors="coerce").to_numpy(dtype=float)
        if "ln_x2" in predictions_df.columns
        else np.full(n_rows, np.nan, dtype=float)
    )
    ln_x2_pred = (
        pd.to_numeric(predictions_df["ln_x2_pred"], errors="coerce").to_numpy(dtype=float)
        if "ln_x2_pred" in predictions_df.columns
        else np.full(n_rows, np.nan, dtype=float)
    )
    true_log_s = (
        pd.to_numeric(predictions_df["logS_true"], errors="coerce").to_numpy(dtype=float)
        if "logS_true" in predictions_df.columns
        else np.full(n_rows, np.nan, dtype=float)
    )
    pred_log_s = (
        pd.to_numeric(predictions_df["logS_pred"], errors="coerce").to_numpy(dtype=float)
        if "logS_pred" in predictions_df.columns
        else np.full(n_rows, np.nan, dtype=float)
    )

    finite_pred_ln_x2_mask = supervision_mask & np.isfinite(ln_x2_pred)
    finite_true_log_s_mask = supervision_mask & np.isfinite(true_log_s)
    finite_pred_log_s_mask = supervision_mask & np.isfinite(pred_log_s)
    finite_log_s_eval_mask = supervision_mask & finite_true_log_s_mask & finite_pred_log_s_mask
    exact_miscible_mask = supervision_mask & np.isfinite(ln_x2_true) & (ln_x2_true == 0.0)

    predictions_df["logS_true_finite"] = finite_true_log_s_mask
    predictions_df["logS_pred_finite"] = finite_pred_log_s_mask
    predictions_df["logS_eval_mask"] = finite_log_s_eval_mask
    predictions_df["ln_x2_eq_0_target"] = exact_miscible_mask

    subsets: dict[str, Any] = {
        "policy": {
            "headline_metric_space": "ln_x2",
            "headline_subset": "ln_x2_all_supervised",
            "logS_metric_subset": "finite_logS_only",
            "exclude_exact_ln_x2_eq_0_from_logS_metrics": True,
            "notes": (
                "logS metrics are computed only where both true and predicted logS are finite. "
                "Exact ln_x2 = 0 implies x2 = 1 and therefore infinite true logS, so those rows "
                "remain in ln_x2 metrics but are excluded from logS metrics."
            ),
            "coverage": {
                "n_total_rows": int(len(predictions_df)),
                "n_supervised_rows": int(supervision_mask.sum()),
                "n_finite_pred_ln_x2_rows": int(finite_pred_ln_x2_mask.sum()),
                "n_true_logS_finite_rows": int(finite_true_log_s_mask.sum()),
                "n_pred_logS_finite_rows": int(finite_pred_log_s_mask.sum()),
                "n_finite_logS_eval_rows": int(finite_log_s_eval_mask.sum()),
                "n_exact_ln_x2_eq_0_rows": int(exact_miscible_mask.sum()),
                "n_logS_rows_excluded": int(supervision_mask.sum() - finite_log_s_eval_mask.sum()),
            },
        },
        "ln_x2_all_supervised": regression_metrics(
            ln_x2_true[supervision_mask],
            ln_x2_pred[supervision_mask],
        ),
        "ln_x2_finite_logS_subset": regression_metrics(
            ln_x2_true[finite_log_s_eval_mask],
            ln_x2_pred[finite_log_s_eval_mask],
        ),
        "logS_finite_subset": regression_metrics(
            true_log_s[finite_log_s_eval_mask],
            pred_log_s[finite_log_s_eval_mask],
        ),
    }
    return json_safe(subsets)


def build_benchmark_artifacts(
    *,
    model_name: str,
    eval_df: pd.DataFrame,
    pred_ln_x2: np.ndarray,
    pred_logS: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    split_mode: str | None = None,
    test_data: str | None = None,
) -> BenchmarkArtifacts:
    """Build canonical report + predictions + summary outputs for one model."""
    eval_frame = eval_df.copy().reset_index(drop=True)
    true_log_s: np.ndarray | None = None
    if "ln_x2" in eval_frame.columns:
        true_log_s_series = logS_from_ln_x2(eval_frame)
        true_log_s = true_log_s_series.to_numpy(dtype=float)
        eval_frame["logS_true"] = true_log_s
        if "logS" not in eval_frame.columns:
            eval_frame["logS"] = true_log_s
    if pred_logS is None:
        pred_logS = _pred_logS_from_ln_x2(
            eval_frame["solvent_smiles"],
            np.asarray(pred_ln_x2, dtype=float),
        )
    predictions_df = build_predictions_frame(
        eval_frame,
        model_name=model_name,
        pred_ln_x2=pred_ln_x2,
        true_logS=true_log_s,
        pred_logS=pred_logS,
        uncertainty=uncertainty,
    )
    evaluation_subsets = _build_evaluation_subsets(predictions_df)
    supervision_mask = predictions_df["has_solubility"].fillna(False).astype(bool).to_numpy()
    finite_mask = supervision_mask & np.isfinite(predictions_df["ln_x2_pred"].to_numpy(dtype=float))
    overall = dict(evaluation_subsets.get("ln_x2_all_supervised", {}))
    overall["n_predictions"] = int(finite_mask.sum())
    overall["n_total"] = int(len(predictions_df))
    overall["n_supervised"] = int(supervision_mask.sum())

    by_temperature = _stratified_metrics(predictions_df, "ln_x2_pred", supervision_mask)
    by_solubility = _solubility_range_metrics(predictions_df, "ln_x2_pred", supervision_mask)
    by_solvent_type, by_solvent = _solvent_metrics(predictions_df, "ln_x2_pred", supervision_mask)
    by_aux_data = _aux_data_metrics(predictions_df, "ln_x2_pred", supervision_mask)

    report = build_report_payload(
        "evaluation",
        metadata={
            **dict(metadata or {}),
            "model": model_name,
            "split": build_split_metadata(split_mode=split_mode, test_data=test_data) if test_data or split_mode else None,
            "test_samples": int(len(predictions_df)),
            "logS_evaluation_policy": evaluation_subsets.get("policy", {}),
        },
        overall=overall,
        stratified={
            "temperature": by_temperature,
            "solubility": by_solubility,
            "solvent_type": by_solvent_type,
            "solvent": by_solvent,
            "aux_data": by_aux_data,
        },
        predictions={
            "true_ln_x2": predictions_df.loc[supervision_mask, "ln_x2"].to_numpy(dtype=float).tolist() if "ln_x2" in predictions_df.columns else [],
            "pred_ln_x2": predictions_df.loc[supervision_mask, "ln_x2_pred"].to_numpy(dtype=float).tolist(),
            "true_logS": predictions_df.loc[supervision_mask, "logS_true"].to_numpy(dtype=float).tolist() if "logS_true" in predictions_df.columns else [],
            "pred_logS": predictions_df.loc[supervision_mask, "logS_pred"].to_numpy(dtype=float).tolist() if "logS_pred" in predictions_df.columns else [],
            "logS_eval_mask": predictions_df.loc[supervision_mask, "logS_eval_mask"].astype(bool).tolist() if "logS_eval_mask" in predictions_df.columns else [],
            "row_indices": predictions_df.loc[supervision_mask, "row_index"].astype(int).tolist() if "row_index" in predictions_df.columns else list(range(int(supervision_mask.sum()))),
        },
        evaluation_subsets=evaluation_subsets,
    )

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "mae": report["overall"].get("mae"),
                "rmse": report["overall"].get("rmse"),
                "r2": report["overall"].get("r2"),
                "bias": report["overall"].get("bias"),
                "n_samples": report["overall"].get("n_samples"),
                "n_predictions": report["overall"].get("n_predictions"),
                "mae_ln_x2_logS_subset": (
                    ((report.get("evaluation_subsets") or {}).get("ln_x2_finite_logS_subset") or {}).get("mae")
                ),
                "rmse_ln_x2_logS_subset": (
                    ((report.get("evaluation_subsets") or {}).get("ln_x2_finite_logS_subset") or {}).get("rmse")
                ),
                "r2_ln_x2_logS_subset": (
                    ((report.get("evaluation_subsets") or {}).get("ln_x2_finite_logS_subset") or {}).get("r2")
                ),
                "n_samples_ln_x2_logS_subset": (
                    ((report.get("evaluation_subsets") or {}).get("ln_x2_finite_logS_subset") or {}).get("n_samples")
                ),
                "mae_logS_finite_subset": (
                    ((report.get("evaluation_subsets") or {}).get("logS_finite_subset") or {}).get("mae")
                ),
                "rmse_logS_finite_subset": (
                    ((report.get("evaluation_subsets") or {}).get("logS_finite_subset") or {}).get("rmse")
                ),
                "r2_logS_finite_subset": (
                    ((report.get("evaluation_subsets") or {}).get("logS_finite_subset") or {}).get("r2")
                ),
                "n_samples_logS_finite_subset": (
                    ((report.get("evaluation_subsets") or {}).get("logS_finite_subset") or {}).get("n_samples")
                ),
                "n_exact_ln_x2_eq_0": (
                    ((((report.get("evaluation_subsets") or {}).get("policy") or {}).get("coverage") or {}).get("n_exact_ln_x2_eq_0_rows"))
                ),
                "n_finite_logS_eval_rows": (
                    ((((report.get("evaluation_subsets") or {}).get("policy") or {}).get("coverage") or {}).get("n_finite_logS_eval_rows"))
                ),
                "split": (
                    (
                        ((report.get("metadata") or {}).get("split") or {}).get("split_mode")
                        or ((report.get("metadata") or {}).get("split") or {}).get("mode")
                    )
                    if isinstance((report.get("metadata") or {}).get("split"), dict)
                    else None
                ),
            }
        ]
    )
    return BenchmarkArtifacts(report=report, predictions=predictions_df, summary=summary)


def write_benchmark_artifacts(
    output_dir: str | Path,
    artifacts: BenchmarkArtifacts,
    *,
    report_name: str = "report.json",
    predictions_name: str = "predictions.csv",
    summary_name: str = "summary.csv",
    input_paths: Mapping[str, str | Path | None] | None = None,
    command: Sequence[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a benchmark bundle to disk."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / report_name
    predictions_path = root / predictions_name
    summary_path = root / summary_name
    report_path.write_text(json.dumps(artifacts.report, indent=2), encoding="utf-8")
    artifacts.predictions.to_csv(predictions_path, index=False)
    artifacts.summary.to_csv(summary_path, index=False)
    metadata = artifacts.report.get("metadata", {}) if isinstance(artifacts.report, dict) else {}
    model_name = None
    model_family = None
    if isinstance(metadata, Mapping):
        model_name = metadata.get("model")
        model_family = metadata.get("model_family")
    if not model_name and not artifacts.summary.empty and "model" in artifacts.summary.columns:
        model_name = artifacts.summary.iloc[0].get("model")
    summary_row = artifacts.summary.iloc[0].to_dict() if not artifacts.summary.empty else {}
    manifest = build_run_manifest(
        "benchmark_bundle",
        model_name=str(model_name) if model_name is not None else None,
        model_family=str(model_family) if model_family is not None else None,
        command=command,
        inputs=input_paths or {
            "test_data": metadata.get("test_data") if isinstance(metadata, Mapping) else None,
            "train_data": metadata.get("train_data") if isinstance(metadata, Mapping) else None,
            "val_data": metadata.get("val_data") if isinstance(metadata, Mapping) else None,
            "predictions_source": metadata.get("predictions_csv") if isinstance(metadata, Mapping) else None,
            "checkpoint": metadata.get("checkpoint") if isinstance(metadata, Mapping) else None,
        },
        outputs={
            "report": report_path,
            "predictions": predictions_path,
            "summary": summary_path,
        },
        metadata={
            "split": metadata.get("split") if isinstance(metadata, Mapping) else None,
            "report_type": artifacts.report.get("report_type") if isinstance(artifacts.report, Mapping) else None,
        },
    )
    card = build_benchmark_card(
        artifacts.report,
        summary_row=summary_row,
        predictions_columns=artifacts.predictions.columns.tolist(),
        metadata=metadata if isinstance(metadata, Mapping) else None,
    )
    write_json(root / "run_manifest.json", manifest)
    write_json(root / "benchmark_card.json", card)
    return report_path, predictions_path, summary_path
