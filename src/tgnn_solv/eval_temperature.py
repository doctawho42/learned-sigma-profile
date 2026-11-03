"""
Temperature-dependent evaluation for TGNN-Solv.

Key experiments:
1. Stratified metrics: T=298K vs other temperatures
2. Temperature extrapolation: train on T≤T_cut, test on T>T_cut
3. van't Hoff consistency: is d(ln x₂)/d(1/T) linear?
4. Per-pair temperature curves vs experimental data
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .model import TGNNSolv
from .inference import predict_solubility, temperature_scan


# ================================================================== #
#  Analyze temperature coverage in dataset                            #
# ================================================================== #

def temperature_coverage(df: pd.DataFrame) -> Dict:
    """
    Analyze how much temperature variation exists in the dataset.

    Returns statistics about multi-temperature pairs.
    """
    print("\n" + "=" * 60)
    print("Temperature Coverage Analysis")
    print("=" * 60)

    sol_df = df[df["has_solubility"]].copy()
    n_total = len(sol_df)

    # Unique temperatures
    T_unique = sol_df["temperature"].nunique()
    at_298 = (sol_df["temperature"] - 298.15).abs() < 1.0

    print(f"  Total records:       {n_total:,}")
    print(f"  Unique temperatures: {T_unique}")
    print(f"  T range:             {sol_df['temperature'].min():.1f} – "
          f"{sol_df['temperature'].max():.1f} K")
    print(f"  At 298.15K:          {at_298.sum():,} "
          f"({at_298.sum()/n_total*100:.1f}%)")
    print(f"  At other T:          {(~at_298).sum():,} "
          f"({(~at_298).sum()/n_total*100:.1f}%)")

    # Pairs with multiple temperatures
    pair_groups = sol_df.groupby(
        ["solute_smiles", "solvent_smiles"]
    )
    pair_T_counts = pair_groups["temperature"].nunique()

    multi_T = pair_T_counts[pair_T_counts > 1]
    n_multi = len(multi_T)
    n_pairs = len(pair_T_counts)

    print(f"\n  Unique (solute, solvent) pairs: {n_pairs:,}")
    print(f"  Pairs with >1 temperature:     {n_multi:,} "
          f"({n_multi/n_pairs*100:.1f}%)")

    if n_multi > 0:
        print(f"  Max temperatures per pair:      {multi_T.max()}")
        print(f"  Median T points (multi-T):      {multi_T.median():.0f}")

        # Records from multi-T pairs
        multi_T_pairs = set(multi_T.index)
        n_records_multi = sol_df.apply(
            lambda r: (r["solute_smiles"], r["solvent_smiles"])
            in multi_T_pairs, axis=1
        ).sum()
        print(f"  Records from multi-T pairs:     {n_records_multi:,}")

    # Temperature histogram
    print("\n  Temperature distribution:")
    bins = [200, 270, 290, 300, 310, 330, 360, 400, 500]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = ((sol_df["temperature"] >= lo) &
                 (sol_df["temperature"] < hi)).sum()
        bar = "█" * int(count / n_total * 100)
        print(f"    [{lo:3d}, {hi:3d}) K: {count:6,d} {bar}")

    return {
        "n_total": n_total,
        "n_at_298": int(at_298.sum()),
        "n_other_T": int((~at_298).sum()),
        "n_pairs": n_pairs,
        "n_multi_T_pairs": n_multi,
        "T_range": (
            float(sol_df["temperature"].min()),
            float(sol_df["temperature"].max()),
        ),
    }


# ================================================================== #
#  Temperature extrapolation experiment                               #
# ================================================================== #

def temperature_extrapolation_split(
    df: pd.DataFrame,
    T_cutoff: float = 310.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data for temperature extrapolation test.

    Train: all records at T ≤ T_cutoff
    Test:  records at T > T_cutoff, but ONLY for pairs
           that also appear in train (so we test T-extrapolation,
           not molecular generalization)

    Returns (train_df, test_df)
    """
    print(f"\n  Temperature extrapolation split (T_cut = {T_cutoff} K)")

    sol_df = df[df["has_solubility"]].copy()

    train_mask = sol_df["temperature"] <= T_cutoff
    test_mask = sol_df["temperature"] > T_cutoff

    # Only keep test pairs that are in train
    train_pairs = set(
        zip(
            sol_df.loc[train_mask, "solute_smiles"],
            sol_df.loc[train_mask, "solvent_smiles"],
        )
    )

    test_in_train = sol_df[test_mask].apply(
        lambda r: (r["solute_smiles"], r["solvent_smiles"])
        in train_pairs, axis=1
    )

    train_df = sol_df[train_mask].reset_index(drop=True)
    test_df = sol_df[test_mask & test_in_train].reset_index(drop=True)

    print(f"  Train (T ≤ {T_cutoff}K): {len(train_df):,} records")
    print(f"  Test  (T > {T_cutoff}K): {len(test_df):,} records "
          f"(pairs seen in train)")
    print(f"  Test T range: {test_df['temperature'].min():.1f} – "
          f"{test_df['temperature'].max():.1f} K")

    return train_df, test_df


# ================================================================== #
#  van't Hoff consistency check                                       #
# ================================================================== #

@torch.no_grad()
def vant_hoff_check(
    model: TGNNSolv,
    solute_smiles: str,
    solvent_smiles: str,
    T_range: Tuple[float, float] = (270, 350),
    n_points: int = 30,
    experimental: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Check van't Hoff consistency for a single system.

    ln(x₂) should be approximately linear in 1/T.
    Deviations indicate ΔCp effects or model artifacts.

    If experimental data is provided, compare predicted and
    measured temperature dependence.
    """
    scan = temperature_scan(
        model, solute_smiles, solvent_smiles,
        T_min=T_range[0], T_max=T_range[1], n_points=n_points,
    )

    inv_T = 1000.0 / scan["T"].values  # 1000/T for better scale
    ln_x2 = scan["ln_x2"].values

    # Linear fit to van't Hoff: ln(x₂) = A + B/T
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(inv_T, ln_x2, 1)  # [intercept, slope]
    ln_x2_fit = P.polyval(inv_T, coef)
    residuals = ln_x2 - ln_x2_fit
    r2_vH = 1.0 - (residuals ** 2).sum() / (
        (ln_x2 - ln_x2.mean()) ** 2
    ).sum()

    # Effective ΔH_sol from slope
    # ln(x₂) ≈ const - ΔH_sol/(R·T)
    # slope in 1000/T → ΔH_sol = -slope × R × 1000
    R = 8.314
    dH_sol_eff = -coef[1] * R * 1000  # J/mol

    # Monotonicity
    diffs = np.diff(ln_x2)
    is_monotonic = np.all(diffs >= -0.01)

    result = {
        "solute": solute_smiles,
        "solvent": solvent_smiles,
        "vH_r2": float(r2_vH),
        "dH_sol_effective": float(dH_sol_eff),
        "is_monotonic": bool(is_monotonic),
        "T_range": T_range,
        "scan": scan,
    }

    # Compare with experimental if available
    if experimental is not None:
        exp_match = experimental[
            (experimental["solute_smiles"] == solute_smiles) &
            (experimental["solvent_smiles"] == solvent_smiles)
        ].sort_values("temperature")

        if len(exp_match) >= 2:
            result["n_exp_points"] = len(exp_match)
            result["exp_T"] = exp_match["temperature"].values
            result["exp_ln_x2"] = exp_match["ln_x2"].values

            # Interpolated predictions at experimental temperatures
            pred_at_exp = np.interp(
                exp_match["temperature"].values,
                scan["T"].values,
                scan["ln_x2"].values,
            )
            result["pred_at_exp_T"] = pred_at_exp
            result["T_dep_mae"] = float(
                np.abs(pred_at_exp - exp_match["ln_x2"].values).mean()
            )

    return result


# ================================================================== #
#  Multi-T pair evaluation                                            #
# ================================================================== #

@torch.no_grad()
def evaluate_multi_T_pairs(
    model: TGNNSolv,
    df: pd.DataFrame,
    min_T_points: int = 3,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """
    Evaluate model on pairs with multiple experimental temperatures.

    This tests whether the model captures correct T-dependence,
    not just the value at 298K.

    Returns DataFrame with per-pair metrics.
    """
    print("\n" + "=" * 60)
    print("Multi-Temperature Pair Evaluation")
    print("=" * 60)

    sol_df = df[df["has_solubility"]].copy()
    pairs = sol_df.groupby(["solute_smiles", "solvent_smiles"])

    # Filter pairs with enough T points
    good_pairs = []
    for (sol, slv), group in pairs:
        if group["temperature"].nunique() >= min_T_points:
            good_pairs.append((sol, slv, group))

    good_pairs = good_pairs[:max_pairs]
    print(f"  Pairs with ≥{min_T_points} temperatures: {len(good_pairs)}")

    results = []
    for sol, slv, group in good_pairs:
        group_sorted = group.sort_values("temperature")
        exp_T = group_sorted["temperature"].values
        exp_ln_x2 = group_sorted["ln_x2"].values

        # Predict at experimental temperatures
        pred_ln_x2 = []
        for T_val in exp_T:
            try:
                r = predict_solubility(model, sol, slv, float(T_val))
                pred_ln_x2.append(r["ln_x2"])
            except Exception:
                pred_ln_x2.append(np.nan)

        pred_ln_x2 = np.array(pred_ln_x2)
        valid = ~np.isnan(pred_ln_x2)

        if valid.sum() < 2:
            continue

        errors = pred_ln_x2[valid] - exp_ln_x2[valid]
        mae = float(np.abs(errors).mean())

        # T-dependence: slope comparison
        # Experimental slope
        exp_slope = np.polyfit(1.0 / exp_T[valid], exp_ln_x2[valid], 1)[0]
        pred_slope = np.polyfit(1.0 / exp_T[valid], pred_ln_x2[valid], 1)[0]
        slope_error = abs(pred_slope - exp_slope) / (abs(exp_slope) + 1e-6)

        results.append({
            "solute": sol[:40],
            "solvent": slv[:20],
            "n_T_points": int(valid.sum()),
            "T_range": f"{exp_T.min():.0f}-{exp_T.max():.0f}",
            "mae": mae,
            "exp_slope": exp_slope,
            "pred_slope": pred_slope,
            "slope_rel_error": slope_error,
        })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print(f"\n  Mean MAE across pairs: "
              f"{results_df['mae'].mean():.3f}")
        print(f"  Mean slope rel. error: "
              f"{results_df['slope_rel_error'].mean():.2%}")
        print(f"  Pairs with slope error <20%: "
              f"{(results_df['slope_rel_error'] < 0.2).sum()}"
              f"/{len(results_df)}")

    return results_df
