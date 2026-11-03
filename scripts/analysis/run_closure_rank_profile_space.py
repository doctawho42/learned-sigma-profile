#!/usr/bin/env python
"""Spectrum of the closure's Jacobian in the space of the sigma-profile.

For each solvent the script collects the records in which it appears, differentiates the
predicted infinite-dilution activity coefficient with respect to that solvent's profile,
and stacks the gradients into a matrix. The eigenvalues of its Gram matrix say how many
directions of the profile the closure exposes to the observations, and how many remain
flat and therefore undetermined.

Normalisation of the profile already fixes one direction, so the constant vector is
projected out and the effective dimension is n_bins - 1.

The activity coefficient is the right surface: solubility reads the profile only through
it, so a direction the closure does not expose is not exposed to solubility either, and
scoring through the equilibrium solver would add an implicit-function factor that does not
change the rank.

A control repeats the calculation on random profiles of equal area. Agreement between the
two means the deficiency belongs to the exchange kernel and the available partners rather
than to the trained network, which is the stronger reading.

Outputs summary.json with per-solvent spectra and an aggregate.

    python scripts/analysis/run_closure_rank_profile_space.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from run_sigma_profile_residual import build_model, learned_profiles  # noqa: E402
from tgnn_solv.data.utils import canonicalize                  # noqa: E402
from tgnn_solv.layers import CosmoSacLayer                      # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles          # noqa: E402

PROFILES = ROOT / "results/sigma_profile_artifact/sigma_profiles.csv"
RECORDS = ROOT / "results/published_idac_check/scored_records.csv"
CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
OUT = ROOT / "results/closure_rank_profile_space"


def jacobian(layer, p2, A2, p1, A1, T, batch: int = 256) -> np.ndarray:
    """d ln gamma_2^inf / d p_solvent, (n_rows, n_bins)."""
    out = []
    for i in range(0, len(T), batch):
        sl = slice(i, i + batch)
        q = torch.tensor(p1[sl], dtype=torch.double, requires_grad=True)
        g = layer.ln_gamma_inf(torch.tensor(p2[sl], dtype=torch.double), q,
                               torch.tensor(A2[sl], dtype=torch.double),
                               torch.tensor(A1[sl], dtype=torch.double), None, None,
                               torch.tensor(T[sl], dtype=torch.double))
        g.sum().backward()
        out.append(q.grad.detach().numpy().copy())
    return np.concatenate(out)


def spectrum(J: np.ndarray, n_bins: int, tol_rel: float = 1e-8) -> dict:
    """Spectrum of J^T J with the normalisation direction removed."""
    # Project out the constant vector: the profile's normalisation closes that direction
    one = np.ones((n_bins, 1)) / np.sqrt(n_bins)
    Jp = J - (J @ one) @ one.T
    s = np.linalg.svd(Jp, compute_uv=False)
    ev = s ** 2
    ev = ev[ev > 0]
    rank = int((ev > tol_rel * ev.max()).sum())
    c = np.cumsum(ev) / ev.sum()
    return {
        "n_rows": int(J.shape[0]), "n_bins": n_bins,
        "effective_dim": n_bins - 1,
        "eigenvalue_max": float(ev.max()), "eigenvalue_min": float(ev.min()),
        "condition_number": float(ev.max() / ev.min()),
        "numerical_rank": rank,
        "dirs_for_99pct": int(np.searchsorted(c, 0.99) + 1),
        "dirs_for_999pct": int(np.searchsorted(c, 0.999) + 1),
        "spectrum_head": ev[:8].tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--min-rows", type=int, default=40,
                    help="minimum rows per solvent, below which the rank is limited by the row count")
    ap.add_argument("--max-solvents", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    layer = CosmoSacLayer()
    layer.eval().double()
    n_bins = int(layer.n_bins)
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    d = pd.read_csv(RECORDS)
    k2 = [canonicalize(str(s)) for s in d.solute_smiles]
    ok = np.array([x in table for x in k2])
    d = d[ok].reset_index(drop=True)
    counts = d.solvent_smiles.astype(str).value_counts()
    solvents = [s for s, n in counts.items() if n >= a.min_rows][:a.max_solvents]
    print(f"solvents with >= {a.min_rows} rows: {len(solvents)}; bins {n_bins}, "
          f"effective dimension {n_bins - 1}")

    tpl = pd.read_csv(ROOT / "notebooks/data/processed/test.csv", nrows=1, low_memory=False)
    res: dict = {"n_bins": n_bins, "min_rows": a.min_rows, "solvents": solvents,
                 "seeds": [], "per_seed": {}}

    for s in a.seeds:
        ck = CKPT_DIR / f"grounded_a_seed{s}.pt"
        if not ck.exists():
            continue
        model, _, cfg = build_model(ck)
        _, as_solvent = learned_profiles(model, cfg, solvents, tpl)
        V = {m: np.asarray(as_solvent[i], float) for i, m in enumerate(solvents)}
        per = {}
        for sv in solvents:
            rows = d[d.solvent_smiles.astype(str) == sv]
            k2r = [canonicalize(str(x)) for x in rows.solute_smiles]
            p2 = np.stack([table[k][0] for k in k2r])
            A2 = np.array([table[k][1] for k in k2r], float)
            p1 = np.repeat(V[sv][None, :], len(rows), axis=0)
            A1 = p1.sum(1)
            T = rows["T_K"].to_numpy(float)
            per[sv] = spectrum(jacobian(layer, p2, A2, p1, A1, T), n_bins)
        res["per_seed"][str(s)] = per
        res["seeds"].append(s)
        rk = np.array([v["numerical_rank"] for v in per.values()])
        d99 = np.array([v["dirs_for_99pct"] for v in per.values()])
        print(f"  seed {s}: rank {rk.min()}..{rk.max()} (median {int(np.median(rk))}); "
              f"directions for 99% of the information {int(np.median(d99))}")

    # --- THE NULL ARM: random profiles of the same area ---------------------------------
    rng = np.random.default_rng(a.seed)
    nullp = {}
    for sv in solvents:
        rows = d[d.solvent_smiles.astype(str) == sv]
        k2r = [canonicalize(str(x)) for x in rows.solute_smiles]
        p2 = np.stack([table[k][0] for k in k2r])
        A2 = np.array([table[k][1] for k in k2r], float)
        area = float(np.mean([table[k][1] for k in k2r]))
        base = rng.dirichlet(np.ones(n_bins)) * area
        p1 = np.repeat(base[None, :], len(rows), axis=0)
        nullp[sv] = spectrum(jacobian(layer, p2, A2, p1, p1.sum(1),
                                      rows["T_K"].to_numpy(float)), n_bins)
    res["random_profile_null"] = nullp
    rkn = np.array([v["numerical_rank"] for v in nullp.values()])
    d99n = np.array([v["dirs_for_99pct"] for v in nullp.values()])
    print(f"  NULL (random profiles): median rank {int(np.median(rkn))}; "
          f"directions for 99% {int(np.median(d99n))}")

    if res["seeds"]:
        d99 = np.array([v["dirs_for_99pct"] for p in res["per_seed"].values()
                        for v in p.values()])
        cond = np.array([v["condition_number"] for p in res["per_seed"].values()
                         for v in p.values()])
        res["summary"] = {
            "dirs_for_99pct_median": int(np.median(d99)),
            "dirs_for_99pct_range": [int(d99.min()), int(d99.max())],
            "condition_number_median": float(np.median(cond)),
            "effective_dim": n_bins - 1,
            "null_dirs_for_99pct_median": int(np.median(d99n)),
        }
        print(f"\n{'=' * 72}\nSUMMARY")
        print(f"  of the {n_bins - 1} available profile directions the closure exposes "
              f"{int(np.median(d99))} (range {d99.min()}..{d99.max()}) at 99% of the information")
        print(f"  median condition number of J^T J: {np.median(cond):.2e}")
        same = abs(np.median(d99) - np.median(d99n)) < 2
        verdict = ("THE SAME, so the deficiency is structural rather than a property of the checkpoint" if same
                   else "different, so the rank depends on where the map is linearised")
        print(f"  the null arm (random profiles) gives {int(np.median(d99n))}: {verdict}")
        res["summary"]["deficiency_is_structural"] = bool(same)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
