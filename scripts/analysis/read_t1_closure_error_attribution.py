#!/usr/bin/env python
"""Attribute the activity-coefficient error to thirds of the charge-density grid.

The grid is split into thirds ordered by the placement cost measured in
run_per_bin_sensitivity_weights. For each third the script substitutes the reference shape
into those bins alone, holds the cavity area fixed, renormalises, and rescores the
infinite-dilution activity coefficients on the same records and with the same metric as
the arm comparison.

A control substitutes into a randomly chosen third of the same size, which is what any
partial substitution recovers on average.

The scoring helpers are copied verbatim from run_idac_learned_vs_reference so that the
numbers sit on the same surface as the deposited arm errors, and an unperturbed arm is
rescored first and compared against the deposit before any substitution is read.

Outputs closure_error_attribution.json.

    python scripts/analysis/read_t1_closure_error_attribution.py
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
WEIGHTS = ROOT / "results/per_bin_sensitivity_weights/summary.json"
DEPOSIT = ROOT / "results/idac_learned_vs_reference/summary.json"
CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
OUT = ROOT / "results/per_bin_sensitivity_weights"


def metrics(m: np.ndarray, g: np.ndarray) -> dict:
    """Copied verbatim from run_idac_learned_vs_reference.py."""
    ok = np.isfinite(m) & np.isfinite(g)
    m, g = np.asarray(m, float)[ok], np.asarray(g, float)[ok]
    if m.size == 0:
        return {"n": 0}
    d = g - m
    return {"n": int(m.size), "aad": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean()),
            "r2": float(1.0 - (d ** 2).sum() / ((m - m.mean()) ** 2).sum())}


def score(layer, p2, A2, p1, A1, T, batch: int = 2048) -> np.ndarray:
    """Copied verbatim from run_idac_learned_vs_reference.py."""
    out = []
    with torch.no_grad():
        for i in range(0, len(T), batch):
            sl = slice(i, i + batch)
            out.append(layer.ln_gamma_inf(
                torch.tensor(p2[sl], dtype=torch.float),
                torch.tensor(p1[sl], dtype=torch.float),
                torch.tensor(A2[sl], dtype=torch.float),
                torch.tensor(A1[sl], dtype=torch.float),
                None, None,
                torch.tensor(T[sl], dtype=torch.float)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def substitute(learned: dict, ref_shape: dict, bins: np.ndarray) -> dict:
    """Reference shape in a subset of bins, learned area, learned shape elsewhere.

    Renormalised to sum to one BEFORE multiplying by the learned area: otherwise the
    substitution would move the area as well, floating two differences instead of one.
    """
    out = {}
    for k, p in learned.items():
        A = float(p.sum())
        s = p / max(A, 1e-30)
        s = s.copy()
        s[bins] = ref_shape[k][bins]
        out[k] = (s / max(s.sum(), 1e-30)) * A
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--weight-vector", default="train", choices=["train", "idac"])
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    layer = CosmoSacLayer()
    layer.eval()
    n_bins = int(layer.n_bins)
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    w = np.asarray(json.loads(WEIGHTS.read_text(encoding="utf8"))
                   ["by_rows"][a.weight_vector]["w_seed_mean"], float)
    q1, q2 = np.quantile(w, 1 / 3), np.quantile(w, 2 / 3)
    thirds = {"expensive": w >= q2, "middle": (w > q1) & (w < q2), "cheap": w <= q1}
    for nm, b in thirds.items():
        print(f"  {nm} third: {int(b.sum())} bins, weight {w[b].min():.3f}...{w[b].max():.3f}")

    d = pd.read_csv(RECORDS)
    T = d["T_K"].to_numpy(float)
    m = d["m"].to_numpy(float)
    s2 = d.solute_smiles.astype(str).to_numpy()
    s1 = d.solvent_smiles.astype(str).to_numpy()
    mols = sorted(set(s2) | set(s1))
    have = {x for x in mols if canonicalize(x) in table}
    print(f"records {len(d)}, molecules {len(mols)}, of which {len(have)} have a reference profile")
    tpl = pd.read_csv(ROOT / "notebooks/data/processed/test.csv", nrows=1, low_memory=False)
    dep = json.loads(DEPOSIT.read_text(encoding="utf8"))["arms"]["learned"]["per_seed"]

    rng = np.random.default_rng(a.seed)
    null_sets = [rng.permutation(n_bins)[:int(thirds["expensive"].sum())] for _ in range(a.n_null)]
    res: dict = {"n_records": int(len(d)), "n_bins": n_bins, "weight_vector": a.weight_vector,
                 "thirds_n_bins": {k: int(v.sum()) for k, v in thirds.items()},
                 "n_null": a.n_null, "per_seed": {}}

    for s in a.seeds:
        ck = CKPT_DIR / f"grounded_a_seed{s}.pt"
        if not ck.exists():
            print(f"seed {s}: no checkpoint, skipping")
            continue
        model, _, cfg = build_model(ck)
        as_solute, as_solvent = learned_profiles(model, cfg, mols, tpl)
        P_u = {x: np.asarray(as_solute[i], float) for i, x in enumerate(mols)}
        P_v = {x: np.asarray(as_solvent[i], float) for i, x in enumerate(mols)}
        R = {}
        for x in mols:
            if x in have:
                r = np.asarray(table[canonicalize(x)][0], float)
                R[x] = r / max(r.sum(), 1e-30)
            else:                      # no reference: the shape is left untouched in every substitution
                R[x] = None
        subst_ok = {x for x in mols if R[x] is not None}
        Rs = {x: R[x] for x in subst_ok}

        def pack(src2, src1):
            p2 = np.stack([src2[x] for x in s2])
            p1 = np.stack([src1[x] for x in s1])
            return p2, p2.sum(1), p1, p1.sum(1)

        def aad_with(bins) -> float:
            u, v = dict(P_u), dict(P_v)
            u.update(substitute({x: P_u[x] for x in subst_ok}, Rs, bins))
            v.update(substitute({x: P_v[x] for x in subst_ok}, Rs, bins))
            return metrics(m, score(layer, *pack(u, v), T))["aad"]

        base = metrics(m, score(layer, *pack(P_u, P_v), T))["aad"]
        dep_aad = float(dep[str(s)]["aad"])
        gate = abs(base - dep_aad)
        full = aad_with(np.ones(n_bins, bool))
        row = {"base_aad": base, "deposited_aad": dep_aad, "gate_abs_diff": gate,
               "full_shape_substituted_aad": full,
               "n_substitutable_molecules": len(subst_ok)}
        print(f"\nseed {s}: unperturbed AAD {base:.4f} against the deposited {dep_aad:.4f} "
              f"(|d| = {gate:.2e} {'ok' if gate < 5e-3 else 'DISAGREES'})")
        print(f"  whole shape from the reference (learned area): {full:.4f}")
        for nm, b in thirds.items():
            v = aad_with(b)
            row[f"aad_{nm}"] = v
            row[f"recovered_{nm}"] = (base - v) / max(base - full, 1e-30)
            print(f"  {nm:8s} third repaired: AAD {v:.4f}  "
                  f"{100 * row[f'recovered_{nm}']:+.1f}% of the gap to full substitution recovered")
        nulls = [aad_with(np.isin(np.arange(n_bins), ns)) for ns in null_sets]
        row["null_aad"] = nulls
        row["null_recovered"] = [float((base - x) / max(base - full, 1e-30)) for x in nulls]
        nr = np.asarray(row["null_recovered"])
        print(f"  NULL (random third, {a.n_null} draws): recovered "
              f"{100 * nr.mean():+.1f}% on average, spread {100 * nr.min():+.1f}...{100 * nr.max():+.1f}%")
        p = float((nr >= row["recovered_expensive"]).mean())
        row["p_expensive_vs_null"] = p
        print(f"  expensive third against the null: p = {p:.3f}")
        res["per_seed"][str(s)] = row

    ks = list(res["per_seed"])
    if ks:
        print(f"\n{'=' * 78}\nSUMMARY over {len(ks)} seeds (share of the gap recovered)")
        for nm in list(thirds) + ["null"]:
            if nm == "null":
                vals = np.array([np.mean(res["per_seed"][k]["null_recovered"]) for k in ks])
            else:
                vals = np.array([res["per_seed"][k][f"recovered_{nm}"] for k in ks])
            print(f"  {nm:8s}: {100 * vals.mean():+6.1f}%  per seed "
                  f"{100 * vals.min():+.1f}...{100 * vals.max():+.1f}%")
        ps = [res["per_seed"][k]["p_expensive_vs_null"] for k in ks]
        print(f"  p(expensive >= null) per seed: {np.round(ps, 3)}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "closure_error_attribution.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\nwritten: {a.out / 'closure_error_attribution.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
