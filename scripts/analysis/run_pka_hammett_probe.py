#!/usr/bin/env python3
"""pKa / Hammett-LFER probe — Phase 3 local scaffolding (no GPU, no external labels).

Instantiates the general theory (memo ``paper/GENERAL_THEORY_MEMO.md``) in a SECOND
real chemical domain: pKa via a FIXED Hammett linear-free-energy closure
``g(sigma) = pKa0 - rho * sigma`` over the physical intermediate ``sigma`` (electronic
substituent effect), with a TABULATED oracle (Hansch-Leo sigma constants).

Why this is the right analog of COSMO-SAC-over-sigma-profile (not Henderson-Hasselbalch):
  - g is a fixed, physically-motivated, *misspecifiable* map;
  - sigma is a learnable physical descriptor with a *tabulated oracle* (grounding test);
  - the TWO classic LFER failure modes map exactly onto the theory's two channels:
        resonance saturation (a NONLINEARITY in sigma the linear g cannot represent)
            -> closure mis-maps a function OF z*  -> B_clos
        ortho / steric field effects (NOT captured by tabulated sigma)
            -> intermediate z* is insufficient      -> B_insuff

This script (a) embeds the Hansch-Leo table and a fixed Hammett closure; (b) validates
the full grounding-test + closure/insufficiency-decomposition HARNESS on a
semi-synthetic Hammett series whose ground-truth B_clos / B_insuff are known in closed
form, so the estimators and one-sided bounds can be checked against truth; (c) shows the
two-cause conflation of the grounding gap Gamma in the pKa domain (the pKa version of the
referee's Counterexample A). The REAL experiment reuses this harness with real pKa labels
and the shared graph encoder in place of the semi-synthetic generator (Modal-gated).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/run_pka_hammett_probe.py \
        --out-json results/pka_hammett/probe.json --fig-dir paper/figs --seed 0
numpy + scikit-learn + matplotlib (+ optional rdkit for the oracle-extraction demo).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# --------------------------------------------------------------------------- #
# Hansch-Leo-Taft sigma constants (sigma_meta, sigma_para).
# Standard tabulated values (Hansch, Leo, Taft, Chem. Rev. 1991, 91, 165).
# --------------------------------------------------------------------------- #
SIGMA: dict[str, tuple[float, float]] = {
    "H": (0.00, 0.00),      "CH3": (-0.07, -0.17),   "C2H5": (-0.07, -0.15),
    "tBu": (-0.10, -0.20),  "OCH3": (0.12, -0.27),   "OH": (0.12, -0.37),
    "NH2": (-0.16, -0.66),  "NMe2": (-0.16, -0.83),  "F": (0.34, 0.06),
    "Cl": (0.37, 0.23),     "Br": (0.39, 0.23),      "I": (0.35, 0.18),
    "CN": (0.56, 0.66),     "NO2": (0.71, 0.78),     "COOH": (0.37, 0.45),
    "COCH3": (0.38, 0.50),  "CF3": (0.43, 0.54),     "SO2CH3": (0.60, 0.72),
    "OCF3": (0.38, 0.35),   "CHO": (0.35, 0.42),     "COOCH3": (0.37, 0.45),
    "SCH3": (0.15, 0.00),   "Ph": (0.06, -0.01),     "NHAc": (0.21, 0.00),
    "SF5": (0.61, 0.68),
}
SUBSTITUENTS = [s for s in SIGMA if s != "H"]
PKA0_BENZOIC = 4.20      # benzoic acid; rho == 1.0 by the definition of sigma
RHO = 1.0


def hammett_closure(sigma_total: np.ndarray) -> np.ndarray:
    """Fixed physical closure g: additive sigma -> pKa (linear Hammett LFER)."""
    return PKA0_BENZOIC - RHO * sigma_total


# --------------------------------------------------------------------------- #
def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _std(v):
    s = v.std(); return (v - v.mean()) / s if s > 0 else v - v.mean()


def _oof_rf(X, y, seed=0):
    X = np.asarray(X); X = X.reshape(-1, 1) if X.ndim == 1 else X
    pred = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        rf.fit(X[tr], y[tr]); pred[te] = rf.predict(X[te])
    return pred


# --------------------------------------------------------------------------- #
# semi-synthetic Hammett series with KNOWN B_clos / B_insuff
# --------------------------------------------------------------------------- #
def build_series(rng, n, fidelity, loss_frac, snr, kappa_clos=0.9, kappa_ins=0.9):
    """Disubstituted benzoic acids. Physical intermediate z* = additive tabulated sigma.

    True pKa = g(z*) - kappa_clos*(1-F)*c_nl(z*)   [resonance saturation, function of z* -> B_clos]
                     - kappa_ins*loss*c_orth(x)    [ortho/steric, orthogonal to z*  -> B_insuff]
                     + noise.
    Returns m (pKa), z_true (sigma_total), x_feat (structure one-hot+ortho), and closed-form truths.
    """
    names = np.array(SUBSTITUENTS)
    idx1 = rng.integers(0, len(names), n)
    idx2 = rng.integers(0, len(names), n)
    pos1 = rng.integers(0, 2, n)   # 0=meta, 1=para  (well-defined for tabulated sigma)
    pos2 = rng.integers(0, 2, n)
    sig = np.array([SIGMA[s] for s in names])          # (S,2): meta,para
    z_true = sig[idx1, pos1] + sig[idx2, pos2]         # additive Hammett sigma (the oracle intermediate)

    # closure-channel error: a nonlinearity in z* (resonance saturation / curved Hammett plot)
    c_nl = _std(z_true ** 2)
    # insufficiency-channel error: ortho steric field, ORTHOGONAL to z* by construction
    ortho = rng.standard_normal(n)                      # a structural feature absent from tabulated sigma
    c_orth = _std(ortho - np.polyval(np.polyfit(z_true, ortho, 1), z_true))  # residualize vs z*

    signal = hammett_closure(z_true)
    noise_sd = signal.std() / np.sqrt(snr)
    b_clos_term = kappa_clos * (1.0 - fidelity) * c_nl
    b_ins_term = kappa_ins * loss_frac * c_orth
    m = signal - b_clos_term - b_ins_term + rng.standard_normal(n) * noise_sd

    # closed-form truths (standardized channels have unit variance)
    b_clos_true = float(np.mean(b_clos_term ** 2))
    b_insuff_true = float(np.mean(b_ins_term ** 2) + noise_sd ** 2)

    # structural features x (what a model must learn sigma from): substituent one-hots + positions + ortho
    S = len(names)
    x = np.zeros((n, 2 * S + 2))
    x[np.arange(n), idx1] += 1; x[np.arange(n), idx2] += 1
    x[:, 2 * S] = pos1; x[:, 2 * S + 1] = pos2
    x = np.column_stack([x, ortho])     # ortho IS in the full structure (so free head can use it)
    return m, z_true, x, b_clos_true, b_insuff_true


@dataclass
class ProbeRecord:
    sweep: str            # "fidelity" | "insufficiency"
    knob: float
    r_orc: float
    r_free: float
    gamma: float
    b_clos_true: float
    b_insuff_true: float
    b_clos_est: float
    b_insuff_est: float
    b_clos_jensen_lb: float     # (E[m]-E[g])^2  (assumption-free lower bound)
    b_insuff_lotv_ub: float     # E[Var(m|bin(g))]  (law-of-total-variance upper bound)


def run_probe_point(rng, n, fidelity, loss_frac, snr, sweep, knob):
    m, z, x, b_clos_true, b_insuff_true = build_series(rng, n, fidelity, loss_frac, snr)
    g = hammett_closure(z)

    r_orc = float(np.mean((m - g) ** 2))               # = B_insuff + B_clos (Lemma 3)
    # free head (DirectGNN analog): OOF RF on full structure x
    free_pred = _oof_rf(x, m, seed=0)
    r_free = float(np.mean((m - free_pred) ** 2))
    gamma = r_orc - r_free                             # oracle-through-closure minus best free predictor

    # estimators (mirror run_b_insuff_decomposition.py)
    ez = _oof_rf(z, m, seed=1)                          # E[m|z*] estimate
    b_insuff_est = float(np.mean((m - ez) ** 2))       # E[Var(m|z*)]
    b_clos_est = float(np.mean((ez - g) ** 2))         # plug-in (biased; reported w/ bounds)
    b_clos_jensen_lb = float((m.mean() - g.mean()) ** 2)
    # LOTV upper bound on B_insuff: bin g into deciles, within-bin variance of m
    bins = np.quantile(g, np.linspace(0, 1, 11))
    which = np.clip(np.digitize(g, bins[1:-1]), 0, 9)
    lotv = float(np.mean([m[which == b].var() * np.sum(which == b) for b in range(10)]) * 10 / len(m)
                 ) if len(m) else float("nan")
    return ProbeRecord(sweep, float(knob), r_orc, r_free, gamma, b_clos_true, b_insuff_true,
                       b_clos_est, b_insuff_est, b_clos_jensen_lb, lotv)


# --------------------------------------------------------------------------- #
def rdkit_oracle_demo():
    """Prove the real-molecule oracle path exists: map substituent fragments -> tabulated sigma."""
    try:
        from rdkit import Chem
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": str(e)}
    # a few real para-substituted benzoic acids (SMILES -> canonical parse check)
    examples = {
        "benzoic acid": "OC(=O)c1ccccc1",
        "p-nitrobenzoic acid": "OC(=O)c1ccc([N+](=O)[O-])cc1",
        "p-methoxybenzoic acid": "OC(=O)c1ccc(OC)cc1",
        "p-aminobenzoic acid": "OC(=O)c1ccc(N)cc1",
    }
    out = {"available": True, "rdkit_parsed": {}}
    for label, smi in examples.items():
        mol = Chem.MolFromSmiles(smi)
        out["rdkit_parsed"][label] = {"valid": mol is not None,
                                      "canonical": Chem.MolToSmiles(mol) if mol else None}
    out["note"] = ("SMILES parse OK; production substituent->sigma parser (RGroupDecomposition "
                   "on the benzoic-acid scaffold + fragment matching) is Phase-3 data-pipeline work.")
    out["sigma_table_size"] = len(SIGMA)
    return out


def make_figure(records, fig_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # the journal's graphics specification; see scripts/analysis/acs_figure_style.py
    import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parent))
    from acs_figure_style import apply as _acs; _acs()
    # soft-pastel base palette (shared with make_paradox_figures.py / fig_overview)
    SALMON, TEAL, BLUE = "#E8A98C", "#7FB5A6", "#8FB3DA"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fid = sorted([r for r in records if r.sweep == "fidelity"], key=lambda r: -r.knob)
    ins = sorted([r for r in records if r.sweep == "insufficiency"], key=lambda r: r.knob)
    # The canvas is the width the figure is SET at: \textwidth of the two-column measure,
    # 502.7 pt = 6.98 in, so the PDF prints at 1:1 and its type prints at the size chosen
    # here.  It used to be 9.5 in inside a 3.33 in column -- a 0.35x reduction that put the
    # legend on the page at 2.8 pt.  If this figure is ever moved back into one column,
    # shrink the canvas with it; do not rely on \includegraphics to scale it.
    # LABELS NAME QUANTITIES, NOT SYMBOLS. The two curves are the error the fixed relation
    # makes when it is handed the true constant, and the cost of handing it that constant in
    # place of a free fit. The sign convention is the one the text uses throughout: the cost is
    # the error with the reference minus the error without it, so a positive value means
    # supplying the reference makes the prediction worse.
    COST = "cost of supplying the reference"
    OWN = "error of the relation itself"
    # Type sizes are printed sizes: the canvas is the \textwidth measure, so the scale is 1.00.
    # 8.8/9.0 pt are the smallest bases that keep a mathtext subscript (0.7x) above 6 pt.
    FS, FS_LEG, FS_TICK = 9.0, 8.0, 7.6
    fig, ax = plt.subplots(1, 2, figsize=(6.98, 3.05))
    ax[0].plot([r.knob for r in fid], [r.b_clos_true for r in fid], "s-", ms=4, lw=1.4,
               color=SALMON, label=OWN)
    ax[0].plot([r.knob for r in fid], [r.gamma for r in fid], "o-", ms=4, lw=1.4,
               color=TEAL, label=COST)
    ax[0].plot([r.knob for r in fid], [r.b_clos_jensen_lb for r in fid], "^--", ms=4, lw=1.4,
               color=BLUE, label="lower bound on that error", alpha=.7)
    ax[0].axhline(0, color="0.6", lw=.8, ls="--", zorder=0)
    ax[0].set_xlabel("adequacy of the relation  $F$", fontsize=FS)
    ax[0].set_ylabel("squared error (pK$_a^2$)", fontsize=FS)
    ax[0].set_title("(a) relation degraded, constant sufficient", fontsize=FS)
    ax[0].invert_xaxis()
    ax[0].legend(fontsize=FS_LEG)
    ax[1].plot([r.knob for r in ins], [r.b_clos_true for r in ins], "s-", ms=4, lw=1.4,
               color=SALMON, label=OWN + " (zero here)")
    ax[1].plot([r.knob for r in ins], [r.gamma for r in ins], "o-", ms=4, lw=1.4,
               color=TEAL, label=COST)
    ax[1].axhline(0, color="0.6", lw=.8, ls="--", zorder=0)
    ax[1].set_xlabel("fraction of records the constant cannot describe", fontsize=FS)
    ax[1].set_ylabel("squared error (pK$_a^2$)", fontsize=FS)
    ax[1].set_title("(b) relation exact, constant insufficient", fontsize=FS)
    ax[1].legend(fontsize=FS_LEG, loc="upper left")
    for a in ax:
        a.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    outs = []
    for ext in ("pdf", "png"):
        p = fig_dir / f"fig_pka_hammett.{ext}"; fig.savefig(p, dpi=150); outs.append(str(p))
    plt.close(fig); return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=Path("results/pka_hammett/probe.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("paper/figs"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    records = []
    for F in [1.0, 0.85, 0.7, 0.5, 0.3, 0.0]:                     # fidelity sweep, loss=0 (A2 holds)
        records.append(run_probe_point(rng, args.n, F, 0.0, args.snr, "fidelity", F))
    for loss in [0.0, 0.15, 0.3, 0.5, 0.7]:                       # insufficiency sweep, F=1 (well-spec closure)
        records.append(run_probe_point(rng, args.n, 1.0, loss, args.snr, "insufficiency", loss))

    figs = [] if args.no_figures else make_figure(records, args.fig_dir)
    demo = rdkit_oracle_demo()

    out = {"config": {"n": args.n, "snr": args.snr, "seed": args.seed, "pka0": PKA0_BENZOIC, "rho": RHO},
           "records": [asdict(r) for r in records], "rdkit_oracle_demo": demo, "figures": figs,
           "note": "Phase-3 local scaffolding; real pKa labels + shared GNN encoder are Modal-gated."}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    print(f"[pka] wrote {args.out_json}")
    print("[pka] decomposition harness vs closed-form truth + one-sided bounds:")
    print("  sweep         knob   R_orc  B_clos(true/est/JensenLB)   B_insuff(true/est/LOTV-UB)   Gamma")
    for r in records:
        print(f"  {r.sweep:<12} {r.knob:5.2f}  {r.r_orc:5.2f}  "
              f"{r.b_clos_true:5.2f}/{r.b_clos_est:5.2f}/{r.b_clos_jensen_lb:5.2f}       "
              f"{r.b_insuff_true:5.2f}/{r.b_insuff_est:5.2f}/{r.b_insuff_lotv_ub:5.2f}      {r.gamma:5.2f}")
    print(f"[pka] rdkit oracle demo: available={demo.get('available')} sigma_table={demo.get('sigma_table_size')}")
    if figs:
        print("[pka] figures:", ", ".join(figs))


if __name__ == "__main__":
    main()
