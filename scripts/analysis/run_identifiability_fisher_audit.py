#!/usr/bin/env python3
"""Numerical identifiability / Fisher audit for the SLE decomposition.

This is the empirical companion to the load-bearing identifiability theorem
(T1) and the semiparametric / class-dependent bound (T2). It does *not* train
any network. It builds the per-pair sensitivity (Jacobian) matrix of the SLE
forward model

    ln x2(T) = -Phi(T; dH_fus, T_m) - ln gamma2(x2, T)

with the analytic ``gamma_inf`` activity actually used by the model:

    ln gamma_inf(T) = a + b * (T_ref / T - 1)        (affine in 1/T)
    ln gamma2       = ln gamma_inf(T) * (1 - x2)^2

and reports how well the *physical decomposition* is determined under three
supervision regimes:

  1. ``solubility_only``      : only ln x2(T) observations.
  2. ``plus_Tm``             : ln x2(T) + a direct melting-point label.
  3. ``plus_Tm_dH``          : ln x2(T) + melting-point + enthalpy labels.

The Fisher information is I = J^T J on *scaled* (dimensionless) parameters so
the condition number reflects rank structure, not unit choices. Two sweeps make
the theory falsifiable:

  * a temperature-window sweep (cond(I) grows as Delta T shrinks);
  * a with / without free ``b`` (activity 1/T slope) comparison that shows the
    crystal enthalpy dH_fus is identifiable from solubility alone *only* when the
    activity class has no free 1/T term -- the honest version of the
    "(Delta T)^{-k}" bound, which silently assumed exactly that restriction.

Run (seconds, CPU):

    python scripts/analysis/run_identifiability_fisher_audit.py \
        --out-dir results/identifiability_audit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tgnn_solv.diagnostics.compensation import compute_phi

# Parameter order used everywhere below.
PARAM_NAMES = ("dH_fus", "T_m", "gamma_a", "gamma_b")

# Characteristic scales used to non-dimensionalize parameters. These mirror the
# scale factors in src/tgnn_solv/config.py (S_H, a 50 K melting-point band, unit
# gamma offset, S_tau_b) so the Fisher condition number measures identifiability,
# not unit mismatch.
PARAM_SCALES = {
    "dH_fus": 5000.0,    # J/mol  (config.S_H)
    "T_m": 50.0,         # K
    "gamma_a": 1.0,      # dimensionless ln gamma_inf offset
    "gamma_b": 300.0,    # K      (config.S_tau_b)
}

R_GAS = 8.31446261815324


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerical identifiability / Fisher audit for SLE decomposition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default="results/identifiability_audit")
    parser.add_argument("--dH-fus", type=float, default=25_000.0, help="True dH_fus [J/mol].")
    parser.add_argument("--T-m", type=float, default=400.0, help="True melting point [K].")
    parser.add_argument("--gamma-a", type=float, default=1.0, help="True ln gamma_inf offset.")
    parser.add_argument("--gamma-b", type=float, default=200.0, help="True ln gamma_inf 1/T slope.")
    parser.add_argument("--T-ref", type=float, default=298.15)
    parser.add_argument("--n-points", type=int, default=9, help="Temperatures per pair.")
    parser.add_argument("--T-min", type=float, default=280.0)
    parser.add_argument("--delta-T", type=float, default=40.0, help="Width of the base T window [K].")
    parser.add_argument("--sigma-lnx2", type=float, default=0.3, help="Noise std on ln x2.")
    parser.add_argument("--sigma-Tm", type=float, default=10.0, help="Noise std on T_m label [K].")
    parser.add_argument("--sigma-dH", type=float, default=2000.0, help="Noise std on dH_fus label [J/mol].")
    parser.add_argument(
        "--delta-T-grid",
        default="10,20,40,80,120",
        help="Comma-separated Delta T values (K) for the window sweep.",
    )
    parser.add_argument("--fd-rel-step", type=float, default=1.0e-4, help="Finite-difference step.")
    parser.add_argument("--n-iter", type=int, default=64, help="Fixed-point iterations.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ln_gamma_inf(T: np.ndarray, a: float, b: float, T_ref: float) -> np.ndarray:
    return a + b * (T_ref / T - 1.0)


def solve_ln_x2(
    T: np.ndarray,
    dH_fus: float,
    T_m: float,
    gamma_a: float,
    gamma_b: float,
    *,
    T_ref: float,
    n_iter: int,
) -> np.ndarray:
    """Forward SLE solve in the gamma_inf activity mode (numpy mirror of solver)."""
    phi = compute_phi(T, np.full_like(T, T_m), np.full_like(T, dH_fus))
    lng_inf = _ln_gamma_inf(T, gamma_a, gamma_b, T_ref)
    x2 = np.clip(np.exp(-phi), 1.0e-10, 1.0 - 1.0e-10)
    for _ in range(n_iter):
        lng2 = lng_inf * (1.0 - x2) ** 2
        x2_new = np.clip(np.exp(-phi - lng2), 1.0e-10, 1.0 - 1.0e-10)
        if np.max(np.abs(np.log(x2_new) - np.log(x2))) < 1.0e-9:
            x2 = x2_new
            break
        x2 = x2_new
    lng2 = lng_inf * (1.0 - x2) ** 2
    return -phi - lng2  # == ln x2


def solubility_jacobian(
    T: np.ndarray,
    theta: dict[str, float],
    *,
    T_ref: float,
    n_iter: int,
    fd_rel_step: float,
) -> np.ndarray:
    """d ln x2(T_k) / d theta_j by central finite differences. Shape (n_T, 4)."""
    base = {k: float(theta[k]) for k in PARAM_NAMES}
    cols = []
    for name in PARAM_NAMES:
        scale = abs(base[name]) if abs(base[name]) > 0 else 1.0
        h = fd_rel_step * scale
        up = dict(base)
        dn = dict(base)
        up[name] += h
        dn[name] -= h
        f_up = solve_ln_x2(T, **up, T_ref=T_ref, n_iter=n_iter)
        f_dn = solve_ln_x2(T, **dn, T_ref=T_ref, n_iter=n_iter)
        cols.append((f_up - f_dn) / (2.0 * h))
    return np.stack(cols, axis=1)


def _scale_columns(J: np.ndarray) -> np.ndarray:
    """Convert d/d theta into d/d (theta / scale) so Fisher cond is unit-free."""
    s = np.array([PARAM_SCALES[name] for name in PARAM_NAMES], dtype=float)
    return J * s[None, :]


def _fisher_report(J_scaled: np.ndarray) -> dict[str, Any]:
    fisher = J_scaled.T @ J_scaled
    eig = np.linalg.eigvalsh(fisher)
    eig = np.clip(eig, 0.0, None)
    lam_min = float(eig[0])
    lam_max = float(eig[-1])
    cond = float(lam_max / lam_min) if lam_min > 0 else float("inf")
    # CRLB std on the scaled dH_fus parameter (index 0), back to J/mol.
    try:
        cov = np.linalg.pinv(fisher, rcond=1.0e-12)
        sd_dH_scaled = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sd_dH_scaled = float("inf")
    sd_dH = sd_dH_scaled * PARAM_SCALES["dH_fus"]
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": cond,
        "eigenvalues": eig.tolist(),
        "crlb_sd_dH_fus_Jmol": sd_dH,
    }


def _augment(J_sol_scaled: np.ndarray, label_rows: list[tuple[str, float]]) -> np.ndarray:
    """Append external single-component label rows (scaled Fisher contribution)."""
    rows = [J_sol_scaled]
    for name, sigma in label_rows:
        e = np.zeros((1, len(PARAM_NAMES)), dtype=float)
        idx = PARAM_NAMES.index(name)
        # Row of the scaled Jacobian for a direct observation theta_name = label:
        # d(theta_name)/d(theta_name/scale) = scale; divided by sigma.
        e[0, idx] = PARAM_SCALES[name] / sigma
        rows.append(e)
    return np.vstack(rows)


def run_regimes(
    T: np.ndarray,
    theta: dict[str, float],
    *,
    T_ref: float,
    n_iter: int,
    fd_rel_step: float,
    sigma_lnx2: float,
    sigma_Tm: float,
    sigma_dH: float,
) -> dict[str, Any]:
    J = solubility_jacobian(T, theta, T_ref=T_ref, n_iter=n_iter, fd_rel_step=fd_rel_step)
    J_sol = _scale_columns(J) / sigma_lnx2
    regimes = {
        "solubility_only": _fisher_report(J_sol),
        "plus_Tm": _fisher_report(_augment(J_sol, [("T_m", sigma_Tm)])),
        "plus_Tm_dH": _fisher_report(
            _augment(J_sol, [("T_m", sigma_Tm), ("dH_fus", sigma_dH)])
        ),
    }
    return regimes


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta = {
        "dH_fus": args.dH_fus,
        "T_m": args.T_m,
        "gamma_a": args.gamma_a,
        "gamma_b": args.gamma_b,
    }
    common = dict(
        T_ref=args.T_ref,
        n_iter=args.n_iter,
        fd_rel_step=args.fd_rel_step,
        sigma_lnx2=args.sigma_lnx2,
        sigma_Tm=args.sigma_Tm,
        sigma_dH=args.sigma_dH,
    )

    def grid(delta_T: float) -> np.ndarray:
        return np.linspace(args.T_min, args.T_min + delta_T, args.n_points)

    # --- 1. Base regimes at the nominal window. ---
    base_regimes = run_regimes(grid(args.delta_T), theta, **common)

    # --- 2. Delta T sweep (theory: cond grows as Delta T -> 0). ---
    delta_T_values = [float(x) for x in str(args.delta_T_grid).split(",") if x.strip()]
    delta_T_sweep = []
    for dT in delta_T_values:
        reg = run_regimes(grid(dT), theta, **common)
        delta_T_sweep.append(
            {
                "delta_T": dT,
                "cond_solubility_only": reg["solubility_only"]["cond"],
                "cond_plus_Tm": reg["plus_Tm"]["cond"],
                "cond_plus_Tm_dH": reg["plus_Tm_dH"]["cond"],
                "crlb_sd_dH_solubility_only_Jmol": reg["solubility_only"]["crlb_sd_dH_fus_Jmol"],
                "crlb_sd_dH_plus_Tm_dH_Jmol": reg["plus_Tm_dH"]["crlb_sd_dH_fus_Jmol"],
            }
        )

    # --- 3. Activity-class comparison (free 1/T slope vs known), solubility-only. ---
    #     The honest version of T2, done as a *nested* comparison at the SAME
    #     operating point (true theta): is dH_fus identifiable from solubility if
    #     the activity 1/T slope b is free vs. if it is known/constrained? We take
    #     one Jacobian and either keep or drop the gamma_b column.
    J_base = solubility_jacobian(
        grid(args.delta_T), theta,
        T_ref=args.T_ref, n_iter=args.n_iter, fd_rel_step=args.fd_rel_step,
    )
    J_base_scaled = _scale_columns(J_base) / args.sigma_lnx2

    def _crlb_dH(J_scaled: np.ndarray) -> tuple[float, float]:
        fisher = J_scaled.T @ J_scaled
        eig = np.clip(np.linalg.eigvalsh(fisher), 0.0, None)
        cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
        # No eigenvalue truncation: a genuinely degenerate model returns a huge sd.
        cov = np.linalg.pinv(fisher, rcond=1.0e-15)
        sd = float(np.sqrt(max(cov[0, 0], 0.0)) * PARAM_SCALES["dH_fus"])
        return cond, sd

    keep_known_b = [PARAM_NAMES.index(n) for n in PARAM_NAMES if n != "gamma_b"]
    cond_free, sd_free = _crlb_dH(J_base_scaled)
    cond_known, sd_known = _crlb_dH(J_base_scaled[:, keep_known_b])
    activity_class = {
        "free_activity_slope_b": {"cond": cond_free, "crlb_sd_dH_fus_Jmol": sd_free},
        "known_activity_slope_b": {"cond": cond_known, "crlb_sd_dH_fus_Jmol": sd_known},
    }

    summary = {
        "config": {
            "theta_true": theta,
            "param_scales": PARAM_SCALES,
            "n_points": args.n_points,
            "T_min": args.T_min,
            "delta_T": args.delta_T,
            "sigma_lnx2": args.sigma_lnx2,
            "sigma_Tm": args.sigma_Tm,
            "sigma_dH": args.sigma_dH,
        },
        "base_regimes": base_regimes,
        "delta_T_sweep": delta_T_sweep,
        "activity_class_comparison": activity_class,
        "interpretation": {
            "T1": "solubility_only Fisher is rank-deficient (lambda_min ~ 0, cond huge): "
            "the crystal enthalpy is collinear with the activity 1/T slope.",
            "T1_closure": "adding a direct dH_fus label (plus_Tm_dH) collapses cond and "
            "makes the CRLB on dH_fus finite -- a melting point alone (plus_Tm) does not.",
            "T2": "without a free activity 1/T term (gamma_b=0) dH_fus becomes identifiable "
            "from solubility alone -- the bound is class-dependent, not universal.",
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    # Console table.
    print("== Base regimes (Delta T = %.0f K) ==" % args.delta_T)
    print(f"{'regime':<18}{'lambda_min':>14}{'cond':>16}{'CRLB sd(dH) J/mol':>22}")
    for name, rep in base_regimes.items():
        cond = rep["cond"]
        cond_s = "inf" if not math.isfinite(cond) else f"{cond:.3e}"
        print(f"{name:<18}{rep['lambda_min']:>14.4e}{cond_s:>16}{rep['crlb_sd_dH_fus_Jmol']:>22.1f}")
    print("\n== Delta T sweep (cond, solubility-only) ==")
    for row in delta_T_sweep:
        c = row["cond_solubility_only"]
        c_s = "inf" if not math.isfinite(c) else f"{c:.3e}"
        print(f"  Delta T = {row['delta_T']:>6.0f} K  ->  cond = {c_s}")
    print("\n== Activity class (solubility-only) ==")
    for k, v in activity_class.items():
        c = v["cond"]
        c_s = "inf" if not math.isfinite(c) else f"{c:.3e}"
        print(f"  {k:<32} cond={c_s:>12}  CRLB sd(dH)={v['crlb_sd_dH_fus_Jmol']:.1f} J/mol")
    print(f"\nWrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
