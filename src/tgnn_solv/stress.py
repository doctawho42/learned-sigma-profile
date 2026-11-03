"""Thermodynamic stress-suite helpers for benchmark bundles and predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .external_benchmarking import regression_metrics


def _finite_mask(df: pd.DataFrame, pred_col: str) -> np.ndarray:
    return np.isfinite(pd.to_numeric(df["ln_x2"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(
        pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    )


def _metric_block(df: pd.DataFrame, pred_col: str, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"n_samples": 0}
    return regression_metrics(
        pd.to_numeric(df.loc[mask, "ln_x2"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df.loc[mask, pred_col], errors="coerce").to_numpy(dtype=float),
    )


def build_stress_suite(
    predictions_df: pd.DataFrame,
    *,
    pred_col: str = "ln_x2_pred",
    train_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute a compact stress-suite report for benchmark predictions."""
    df = predictions_df.copy()
    if pred_col not in df.columns:
        raise ValueError(f"Predictions dataframe is missing `{pred_col}`.")
    base_mask = _finite_mask(df, pred_col)
    temperatures = pd.to_numeric(df["temperature"], errors="coerce") if "temperature" in df.columns else pd.Series(np.nan, index=df.index)

    slices: dict[str, dict[str, Any]] = {
        "all": _metric_block(df, pred_col, base_mask),
    }
    if np.isfinite(temperatures.to_numpy(dtype=float)).any():
        q20 = float(np.nanquantile(temperatures, 0.2))
        q80 = float(np.nanquantile(temperatures, 0.8))
        slices["temperature_low"] = _metric_block(df, pred_col, base_mask & (temperatures <= q20).to_numpy())
        slices["temperature_high"] = _metric_block(df, pred_col, base_mask & (temperatures >= q80).to_numpy())

    ln_x2 = pd.to_numeric(df["ln_x2"], errors="coerce")
    slices["very_low_solubility"] = _metric_block(df, pred_col, base_mask & (ln_x2 <= -6.0).to_numpy())
    slices["moderate_to_high_solubility"] = _metric_block(df, pred_col, base_mask & (ln_x2 >= -3.0).to_numpy())

    if "has_T_m" in df.columns:
        has_tm = df["has_T_m"].fillna(False).astype(bool).to_numpy()
        slices["with_T_m"] = _metric_block(df, pred_col, base_mask & has_tm)
        slices["without_T_m"] = _metric_block(df, pred_col, base_mask & ~has_tm)
    if "has_dH_fus" in df.columns:
        has_dh = df["has_dH_fus"].fillna(False).astype(bool).to_numpy()
        slices["with_dH_fus"] = _metric_block(df, pred_col, base_mask & has_dh)
        slices["without_dH_fus"] = _metric_block(df, pred_col, base_mask & ~has_dh)
    elif "dH_mask" in df.columns:
        has_dh = df["dH_mask"].fillna(False).astype(bool).to_numpy()
        slices["with_dH_fus"] = _metric_block(df, pred_col, base_mask & has_dh)
        slices["without_dH_fus"] = _metric_block(df, pred_col, base_mask & ~has_dh)

    if "solvent_smiles" in df.columns:
        water_mask = df["solvent_smiles"].astype(str).eq("O").to_numpy()
        slices["water"] = _metric_block(df, pred_col, base_mask & water_mask)
        slices["organic"] = _metric_block(df, pred_col, base_mask & ~water_mask)

    gamma_columns = [name for name in ("ln_gamma_inf", "gamma_inf", "ln_gamma_2") if name in df.columns]
    if gamma_columns:
        gamma = pd.to_numeric(df[gamma_columns[0]], errors="coerce")
        if np.isfinite(gamma.to_numpy(dtype=float)).any():
            threshold = float(np.nanquantile(np.abs(gamma), 0.9))
            slices["extreme_gamma"] = _metric_block(df, pred_col, base_mask & (np.abs(gamma) >= threshold).to_numpy())

    if train_df is not None and {"solute_smiles", "solvent_smiles"} <= set(train_df.columns):
        train_solutes = set(train_df["solute_smiles"].astype(str))
        train_solvents = set(train_df["solvent_smiles"].astype(str))
        solute_seen = df["solute_smiles"].astype(str).isin(train_solutes).to_numpy()
        solvent_seen = df["solvent_smiles"].astype(str).isin(train_solvents).to_numpy()
        slices["unseen_solute"] = _metric_block(df, pred_col, base_mask & ~solute_seen)
        slices["unseen_solvent"] = _metric_block(df, pred_col, base_mask & ~solvent_seen)
        slices["both_seen"] = _metric_block(df, pred_col, base_mask & solute_seen & solvent_seen)

    return {
        "n_rows": int(len(df)),
        "pred_col": pred_col,
        "slices": slices,
    }

