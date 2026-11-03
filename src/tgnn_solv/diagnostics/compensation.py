from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

R_GAS = 8.31446261815324


def _bool_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    bool_mask = raw.isin({"true", "1", "1.0", "yes"})
    numeric = pd.to_numeric(series, errors="coerce")
    return bool_mask | (numeric.fillna(0.0) > 0.0)


def canonical_smiles(smiles: object) -> str:
    if pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def compute_phi(
    T: pd.Series | np.ndarray,
    Tm: pd.Series | np.ndarray,
    dH_fus: pd.Series | np.ndarray,
    dCp_fus: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    T_arr = pd.to_numeric(pd.Series(T), errors="coerce").to_numpy(dtype=float)
    Tm_arr = pd.to_numeric(pd.Series(Tm), errors="coerce").to_numpy(dtype=float)
    dH_arr = pd.to_numeric(pd.Series(dH_fus), errors="coerce").to_numpy(dtype=float)
    phi = (dH_arr / R_GAS) * (1.0 / T_arr - 1.0 / Tm_arr)
    if dCp_fus is not None:
        dCp_arr = (
            pd.to_numeric(pd.Series(dCp_fus), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ratio = Tm_arr / T_arr
        phi = phi - (dCp_arr / R_GAS) * ((ratio - 1.0) - np.log(ratio + 1.0e-10))
    return phi


def safe_corr(x: pd.Series, y: pd.Series) -> float | None:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    value = float(np.corrcoef(x_arr, y_arr)[0, 1])
    return value if math.isfinite(value) else None


def regression_metrics(
    true: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float | int | None]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if true.size == 0:
        return {
            "n_rows": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "bias": None,
            "target_std": None,
            "pred_std": None,
            "pred_std_ratio": None,
        }
    err = pred - true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    true_std = float(np.std(true, ddof=0))
    pred_std = float(np.std(pred, ddof=0))
    return {
        "n_rows": int(true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "bias": float(np.mean(err)),
        "target_std": true_std,
        "pred_std": pred_std,
        "pred_std_ratio": None if true_std == 0.0 else float(pred_std / true_std),
    }


def annotate_compensation_frame(
    df: pd.DataFrame,
    *,
    label: str = "run",
) -> pd.DataFrame:
    work = df.copy()
    required_cols = [
        "ln_x2_true",
        "ln_x2_pred",
        "T",
        "T_m",
        "dH_fus",
        "Phi",
        "ln_gamma_2",
        "solute_smiles",
        "solvent_smiles",
    ]
    missing = [col for col in required_cols if col not in work.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    work["solute_canonical"] = work["solute_smiles"].map(canonical_smiles)
    work["solvent_canonical"] = work["solvent_smiles"].map(canonical_smiles)
    work["pair_canonical"] = (
        work["solute_canonical"] + ">>" + work["solvent_canonical"]
    )
    dcp = work["dCp_fus_solver"] if "dCp_fus_solver" in work.columns else None
    work["Phi_true"] = compute_phi(work["T"], work["T_m"], work["dH_fus"], dcp)
    work["delta_phi"] = pd.to_numeric(work["Phi"], errors="coerce") - work["Phi_true"]
    work["ln_gamma_required"] = (
        -pd.to_numeric(work["ln_x2_true"], errors="coerce") - work["Phi_true"]
    )
    work["delta_gamma"] = (
        pd.to_numeric(work["ln_gamma_2"], errors="coerce") - work["ln_gamma_required"]
    )
    work["compensation_sum"] = work["delta_phi"] + work["delta_gamma"]
    work["compensation_opposite_sign"] = (
        np.sign(work["delta_phi"]).replace(0.0, np.nan)
        * np.sign(work["delta_gamma"]).replace(0.0, np.nan)
        < 0.0
    )
    work["physics_error_from_terms"] = -work["compensation_sum"]
    work["final_error"] = pd.to_numeric(work["ln_x2_pred"], errors="coerce") - pd.to_numeric(
        work["ln_x2_true"], errors="coerce"
    )
    work["abs_compensation_sum"] = work["compensation_sum"].abs()
    work["abs_magnitude_gap"] = (
        work["delta_phi"].abs() - work["delta_gamma"].abs()
    ).abs()
    work["run_label"] = label
    return work


def compensation_row_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    if "has_solubility" in df.columns:
        mask &= _bool_series(df["has_solubility"])
    for col in ("ln_x2_true", "ln_x2_pred", "T", "Phi", "ln_gamma_2"):
        if col in df.columns:
            mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce"))
    if "has_T_m" in df.columns:
        mask &= _bool_series(df["has_T_m"])
    else:
        mask &= pd.to_numeric(df.get("T_m"), errors="coerce") > 0.0
    if "has_dH_fus" in df.columns:
        mask &= _bool_series(df["has_dH_fus"])
    else:
        mask &= np.isfinite(pd.to_numeric(df.get("dH_fus"), errors="coerce"))
    return mask


def compensation_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    summary = regression_metrics(
        pd.to_numeric(df["ln_x2_true"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["ln_x2_pred"], errors="coerce").to_numpy(dtype=float),
    )
    if df.empty:
        summary.update(
            {
                "delta_phi_mean": None,
                "delta_phi_median": None,
                "delta_phi_std": None,
                "delta_gamma_mean": None,
                "delta_gamma_median": None,
                "delta_gamma_std": None,
                "delta_phi_delta_gamma_corr": None,
                "opposite_sign_fraction": None,
                "mean_abs_compensation_sum": None,
                "median_abs_compensation_sum": None,
                "mean_abs_magnitude_gap": None,
                "mean_abs_final_minus_physics_error_gap": None,
            }
        )
        return summary
    summary.update(
        {
            "delta_phi_mean": float(df["delta_phi"].mean()),
            "delta_phi_median": float(df["delta_phi"].median()),
            "delta_phi_std": float(df["delta_phi"].std(ddof=0)),
            "delta_gamma_mean": float(df["delta_gamma"].mean()),
            "delta_gamma_median": float(df["delta_gamma"].median()),
            "delta_gamma_std": float(df["delta_gamma"].std(ddof=0)),
            "delta_phi_delta_gamma_corr": safe_corr(
                df["delta_phi"], df["delta_gamma"]
            ),
            "opposite_sign_fraction": float(df["compensation_opposite_sign"].mean()),
            "mean_abs_compensation_sum": float(df["abs_compensation_sum"].mean()),
            "median_abs_compensation_sum": float(df["abs_compensation_sum"].median()),
            "mean_abs_magnitude_gap": float(df["abs_magnitude_gap"].mean()),
            "mean_abs_final_minus_physics_error_gap": float(
                (
                    df["final_error"].abs()
                    - df["physics_error_from_terms"].abs()
                ).abs().mean()
            ),
        }
    )
    return summary


def gc_reference_summary(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    cluster_key: str = "solute_smiles",
) -> dict[str, Any]:
    """Robust compensation diagnostic against a GC (Joback) crystal reference.

    The strict compensation metric needs measured T_m AND dH_fus per row, but the
    standard test split has no fusion-enthalpy labels, which made the published
    ``corr=-0.876`` an n=8-probe artifact. This variant uses per-solute Joback
    group-contribution priors (available for any SMILES) as the crystal
    reference, so it is computable on the full supervised test set.

    Expects columns: ``solute_smiles``, ``T``, ``Phi``, ``ln_gamma_2``,
    ``ln_x2_true`` and (optionally) ``has_solubility``. Returns ``delta_phi_mean``
    (the robust headline metric) plus ``corr`` and bootstrap CIs.

    IMPORTANT -- ``delta_phi_delta_gamma_corr`` / ``opposite_sign_fraction`` are
    trivial-by-construction and must NOT be read as evidence of learned
    compensation: ``delta_phi = phi_pred - phi_gc`` and
    ``delta_gamma = lng2 + ln_x2_true + phi_gc`` share the same high-variance
    per-solute ``phi_gc`` with opposite sign, which forces a strong negative
    correlation for ANY model (and corr -> -1 additionally as the ln x2 fit
    improves). Both a row-level and a solute-CLUSTER bootstrap CI are returned;
    the cluster CI is the honest one, since rows are repeated, phi_gc-sharing
    measurements of the same solute and a row bootstrap understates the width.
    """
    from ..group_contribution import compute_gc_priors

    work = df.copy()
    if "has_solubility" in work.columns:
        work = work[_bool_series(work["has_solubility"])]
    required = ("solute_smiles", "T", "Phi", "ln_gamma_2", "ln_x2_true")
    missing = [c for c in required if c not in work.columns]
    if missing:
        return {"n_rows": 0, "missing_columns": missing}

    cache: dict[str, dict[str, float | None]] = {}

    def _gc(smi: str) -> dict[str, float | None]:
        if smi not in cache:
            cache[smi] = compute_gc_priors(str(smi))
        return cache[smi]

    tm_gc = work["solute_smiles"].map(lambda s: _gc(s)["T_m_gc"])
    dh_gc = work["solute_smiles"].map(lambda s: _gc(s)["dH_fus_gc"])
    phi_gc = compute_phi(work["T"], tm_gc, dh_gc)

    phi_pred = pd.to_numeric(work["Phi"], errors="coerce").to_numpy(dtype=float)
    lng2 = pd.to_numeric(work["ln_gamma_2"], errors="coerce").to_numpy(dtype=float)
    ln_x2_true = pd.to_numeric(work["ln_x2_true"], errors="coerce").to_numpy(dtype=float)

    delta_phi = phi_pred - phi_gc
    delta_gamma = lng2 - (-ln_x2_true - phi_gc)
    mask = np.isfinite(delta_phi) & np.isfinite(delta_gamma)
    dp = delta_phi[mask]
    dg = delta_gamma[mask]
    if dp.size < 2:
        return {"n_rows": int(dp.size)}

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rng = np.random.default_rng(seed)

    def _bootstrap_ci(index_iter) -> tuple[float | None, float | None]:
        boots = [_corr(dp[idx], dg[idx]) for idx in index_iter]
        boots = [b for b in boots if math.isfinite(b)]
        if not boots:
            return (None, None)
        return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # Row-level bootstrap: OVERSTATES precision -- the rows are repeated
    # measurements of the same solutes that share an identical per-solute phi_gc.
    row_ci = _bootstrap_ci(
        rng.integers(0, dp.size, dp.size) for _ in range(int(n_bootstrap))
    )

    # Cluster (solute) bootstrap: the honest interval, resampling the correlated
    # unit (solute) rather than the row.
    cluster_ci: tuple[float | None, float | None] = (None, None)
    n_clusters: int | None = None
    if cluster_key in work.columns:
        clusters = work[cluster_key].astype(str).to_numpy()[mask]
        unique = np.unique(clusters)
        n_clusters = int(unique.size)
        idx_by_cluster = [np.flatnonzero(clusters == c) for c in unique]

        def _cluster_indices():
            for _ in range(int(n_bootstrap)):
                chosen = rng.integers(0, len(idx_by_cluster), len(idx_by_cluster))
                yield np.concatenate([idx_by_cluster[j] for j in chosen])

        cluster_ci = _bootstrap_ci(_cluster_indices())

    return {
        "reference": "GC(Joback) crystal term",
        "n_rows": int(dp.size),
        "n_clusters": n_clusters,
        "cluster_by": cluster_key,
        "delta_phi_mean": float(dp.mean()),
        "delta_phi_std": float(dp.std(ddof=0)),
        "delta_gamma_mean": float(dg.mean()),
        "delta_gamma_std": float(dg.std(ddof=0)),
        "delta_phi_delta_gamma_corr": _corr(dp, dg),
        "corr_bootstrap_ci95_low": row_ci[0],
        "corr_bootstrap_ci95_high": row_ci[1],
        "corr_cluster_bootstrap_ci95_low": cluster_ci[0],
        "corr_cluster_bootstrap_ci95_high": cluster_ci[1],
        "opposite_sign_fraction": float((np.sign(dp) * np.sign(dg) < 0).mean()),
    }


def _gc_decomposition_arrays(
    df: pd.DataFrame,
    *,
    cluster_key: str = "solute_smiles",
) -> tuple[dict[str, Any] | None, list[str]]:
    """Masked (delta_phi, delta_gamma, phi_gc, phi_pred, lng2, lnx, clusters).

    Shared by the GC-reference compensation diagnostics. Filters to supervised
    rows, computes the per-solute Joback crystal reference phi_gc, and returns
    finite-aligned arrays. Returns ``(None, missing_columns)`` if inputs are
    insufficient.
    """
    from ..group_contribution import compute_gc_priors

    work = df.copy()
    if "has_solubility" in work.columns:
        work = work[_bool_series(work["has_solubility"])]
    required = ("solute_smiles", "T", "Phi", "ln_gamma_2", "ln_x2_true")
    missing = [c for c in required if c not in work.columns]
    if missing:
        return None, missing

    cache: dict[str, dict[str, float | None]] = {}

    def _gc(smi: str) -> dict[str, float | None]:
        if smi not in cache:
            cache[smi] = compute_gc_priors(str(smi))
        return cache[smi]

    tm_gc = work["solute_smiles"].map(lambda s: _gc(s)["T_m_gc"])
    dh_gc = work["solute_smiles"].map(lambda s: _gc(s)["dH_fus_gc"])
    phi_gc = compute_phi(work["T"], tm_gc, dh_gc)
    phi_pred = pd.to_numeric(work["Phi"], errors="coerce").to_numpy(dtype=float)
    lng2 = pd.to_numeric(work["ln_gamma_2"], errors="coerce").to_numpy(dtype=float)
    lnx = pd.to_numeric(work["ln_x2_true"], errors="coerce").to_numpy(dtype=float)

    delta_phi = phi_pred - phi_gc
    delta_gamma = lng2 - (-lnx - phi_gc)
    mask = np.isfinite(delta_phi) & np.isfinite(delta_gamma) & np.isfinite(phi_gc)
    clusters = (
        work[cluster_key].astype(str).to_numpy()[mask]
        if cluster_key in work.columns
        else None
    )
    return (
        {
            "dphi": delta_phi[mask],
            "dgam": delta_gamma[mask],
            "phi_gc": phi_gc[mask],
            "phi_pred": phi_pred[mask],
            "lng2": lng2[mask],
            "lnx": lnx[mask],
            "clusters": clusters,
        },
        [],
    )


def decomposition_identifiability_summary(
    df: pd.DataFrame,
    *,
    n_perm: int = 1000,
    n_bootstrap: int = 2000,
    seed: int = 42,
    cluster_key: str = "solute_smiles",
) -> dict[str, Any]:
    """Honest crystal/activity decomposition (non-identifiability) diagnostic.

    Replaces the trivial ``corr(delta_phi, delta_gamma)`` headline. The raw
    correlation is an *accounting artifact*: ``delta_phi + delta_gamma`` equals
    the (small) physics-only prediction error, so corr is forced toward -1 when
    the model fits ln x2; and ``delta_phi``/``delta_gamma`` additionally share the
    high-variance ``phi_gc`` term with opposite sign. This function quantifies
    BOTH artifacts and reports the defensible signal:

    * ``delta_phi_mean`` / ``delta_phi_std`` with a SOLUTE-CLUSTER bootstrap CI --
      how far the model's crystal term sits from the physical (Joback) prior. This
      is the real evidence of non-identifiability (the crystal term is set to
      whatever the activity term can absorb).
    * ``corr_observed`` vs a PERMUTATION NULL (``phi_pred`` and ``lng2`` shuffled
      independently to destroy any learned coupling) -- the structural floor. The
      ``corr_excess_over_null`` is how much correlation survives beyond the
      artifact.
    * ``partial_corr_given_phi_gc`` and the model-internal
      ``internal_corr_phi_pred_lng2`` -- the residual association once the shared
      reference is removed.
    * ``mean_abs_compensation_sum`` -- the physics-only error that forces corr
      toward -1.
    """
    arrays, missing = _gc_decomposition_arrays(df, cluster_key=cluster_key)
    if arrays is None:
        return {"n_rows": 0, "missing_columns": missing}
    dphi = arrays["dphi"]
    dgam = arrays["dgam"]
    n = int(dphi.size)
    if n < 3:
        return {"n_rows": n}
    phi_gc = arrays["phi_gc"]
    phi_pred = arrays["phi_pred"]
    lng2 = arrays["lng2"]
    lnx = arrays["lnx"]
    clusters = arrays["clusters"]
    rng = np.random.default_rng(seed)

    def _c(a: np.ndarray, b: np.ndarray) -> float:
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    observed = _c(dphi, dgam)

    # Permutation null: independently shuffle phi_pred and lng2 (break any learned
    # crystal/activity coupling), keep phi_gc and ln_x2_true aligned per row, then
    # recompute the same corr. This is the structural floor from the shared phi_gc.
    null_vals = []
    for _ in range(int(n_perm)):
        pp = phi_pred[rng.permutation(n)]
        lg = lng2[rng.permutation(n)]
        null_vals.append(_c(pp - phi_gc, lg - (-lnx - phi_gc)))
    null_arr = np.array([v for v in null_vals if math.isfinite(v)])
    null_mean = float(null_arr.mean()) if null_arr.size else None
    null_lo = float(np.percentile(null_arr, 2.5)) if null_arr.size else None
    null_hi = float(np.percentile(null_arr, 97.5)) if null_arr.size else None

    def _pcorr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        rxy, rxz, ryz = _c(x, y), _c(x, z), _c(y, z)
        denom = math.sqrt(max((1.0 - rxz**2) * (1.0 - ryz**2), 1e-12))
        return (rxy - rxz * ryz) / denom

    # Solute-cluster bootstrap for delta_phi_mean (the headline) -- row bootstrap
    # understates the CI because rows are repeated measurements of the same solute.
    if clusters is not None:
        unique = np.unique(clusters)
        idx_by_cluster = [np.flatnonzero(clusters == c) for c in unique]

        def _resample() -> np.ndarray:
            chosen = rng.integers(0, len(idx_by_cluster), len(idx_by_cluster))
            return np.concatenate([idx_by_cluster[j] for j in chosen])

        n_clusters: int | None = int(unique.size)
    else:
        def _resample() -> np.ndarray:
            return rng.integers(0, n, n)

        n_clusters = None

    boots = np.array([float(dphi[_resample()].mean()) for _ in range(int(n_bootstrap))])
    dphi_ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots.size
        else (None, None)
    )

    return {
        "reference": "GC(Joback) crystal term",
        "n_rows": n,
        "n_clusters": n_clusters,
        # Defensible headline: crystal-term deviation from the physical prior.
        "delta_phi_mean": float(dphi.mean()),
        "delta_phi_std": float(dphi.std(ddof=0)),
        "delta_phi_mean_cluster_ci95_low": dphi_ci[0],
        "delta_phi_mean_cluster_ci95_high": dphi_ci[1],
        # The correlation and the evidence that it is non-evidence.
        "corr_observed": observed,
        "corr_permutation_null_mean": null_mean,
        "corr_permutation_null_ci95_low": null_lo,
        "corr_permutation_null_ci95_high": null_hi,
        "corr_excess_over_null": (
            observed - null_mean if (null_mean is not None and math.isfinite(observed)) else None
        ),
        "partial_corr_given_phi_gc": _pcorr(dphi, dgam, phi_gc),
        "internal_corr_phi_pred_lng2": _c(phi_pred, lng2),
        # Accounting identity: delta_phi + delta_gamma = -(physics-only error).
        "mean_abs_compensation_sum": float(np.abs(dphi + dgam).mean()),
        "interpretation": (
            "corr(delta_phi, delta_gamma) is an accounting/shared-reference "
            "artifact (a permutation null reproduces most of it); the defensible "
            "non-identifiability signal is the magnitude of delta_phi -- the "
            "crystal term's deviation from the physical prior."
        ),
    }


def grouped_compensation_summary(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {
            "run_label": label,
            "n_rows": int(len(group)),
            **compensation_summary(group),
        }
        for col, value in zip(group_cols, keys):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)
