#!/usr/bin/env python
"""Test whether the output layer is what keeps mass in bins the reference leaves empty.

The sigma head ends in a normalisation that cannot return an exact zero, so a profile that
should be empty in a bin is instead slightly populated there. This script asks whether
that limit is what the trained network is up against.

The comparison is internal to one checkpoint and needs no external floor. The theoretical
floor of a normalised exponential depends on the spread of the logits, which is itself a
property of the trained network, so comparing against it would compare against a number
from a different regime. Instead the script asks how small a value the same head reaches
anywhere on the corpus, and how large a value it leaves in the bins where the reference is
exactly zero. A gap of several orders between the two means the remaining population is
not the layer's limit.

A control repeats the statistics on an untrained network of the same architecture, which
separates what training produces from what the parameterisation imposes.

Outputs summary.json and bin_mass_distributions.npz, the latter so that figures can be
drawn from the deposit without loading checkpoints.

    python scripts/analysis/read_t2_softmax_floor_is_binding.py
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
OUT = ROOT / "results/t2_softmax_floor"


def stats(shape: np.ndarray, zero_mask: np.ndarray) -> dict:
    """shape: (n_mol, n_bins), normalised to one. zero_mask: where the reference is EXACTLY zero."""
    ped = np.array([shape[i, zero_mask[i]].sum() for i in range(len(shape))])
    in_zero = np.concatenate([shape[i, zero_mask[i]] for i in range(len(shape))])
    in_nz = np.concatenate([shape[i, ~zero_mask[i]] for i in range(len(shape))])
    return {
        "pedestal_median": float(np.median(ped)),
        "pedestal_mean": float(ped.mean()),
        "pedestal_p95": float(np.quantile(ped, 0.95)),
        "pedestal_max": float(ped.max()),
        "min_bin_mass_anywhere": float(shape.min()),
        "median_bin_mass_in_zero_bins": float(np.median(in_zero)),
        "p01_bin_mass_in_zero_bins": float(np.quantile(in_zero, 0.01)),
        "median_bin_mass_in_nonzero_bins": float(np.median(in_nz)),
        "orders_between_min_and_zero_bin_median": float(
            np.log10(max(np.median(in_zero), 1e-300) / max(shape.min(), 1e-300))),
        "share_of_zero_bins_below_1e_5": float((in_zero < 1e-5).mean()),
        "share_of_zero_bins_below_1e_3": float((in_zero < 1e-3).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="directory holding grounded_a_seed*.pt; defaults to the reference arm")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    n_bins = int(CosmoSacLayer().n_bins)
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    d = pd.read_csv(RECORDS)
    mols = sorted({str(s) for s in pd.concat([d.solute_smiles, d.solvent_smiles])
                   if canonicalize(str(s)) in table})
    ref = np.stack([np.asarray(table[canonicalize(m)][0], float) for m in mols])
    ref = ref / np.maximum(ref.sum(1, keepdims=True), 1e-30)
    zero = ref == 0.0
    print(f"molecules with a reference profile: {len(mols)}; bins {n_bins}")
    print(f"reference-zero bins: {100 * zero.mean():.1f}% of all cells "
          f"(median {int(np.median(zero.sum(1)))} of {n_bins} per molecule)")

    tpl = pd.read_csv(ROOT / "notebooks/data/processed/test.csv", nrows=1, low_memory=False)
    res: dict = {"n_molecules": len(mols), "n_bins": n_bins,
                 "share_reference_zero_cells": float(zero.mean()),
                 "median_zero_bins_per_molecule": int(np.median(zero.sum(1))),
                 "per_seed": {}}
    # The mass arrays are written beside the summary so that the figure is drawn FROM THE
    # DEPOSIT rather than recomputing five checkpoints of its own, which would let the figure and
    # the summary diverge silently.
    dists: dict[str, np.ndarray] = {}

    for s in a.seeds:
        ck = a.ckpt_dir / f"grounded_a_seed{s}.pt"
        if not ck.exists():
            print(f"seed {s}: no checkpoint, skipping")
            continue
        model, _, cfg = build_model(ck)
        for role, prof in zip(("solute", "solvent"), learned_profiles(model, cfg, mols, tpl)):
            p = np.asarray(prof, float)
            p = p / np.maximum(p.sum(1, keepdims=True), 1e-30)
            st = stats(p, zero)
            res["per_seed"].setdefault(str(s), {})[role] = st
            dists[f"zero_bins_seed{s}_{role}"] = np.concatenate(
                [p[i, zero[i]] for i in range(len(p))]).astype(np.float32)
            dists[f"nonzero_bins_seed{s}_{role}"] = np.concatenate(
                [p[i, ~zero[i]] for i in range(len(p))]).astype(np.float32)
            dists[f"pedestal_seed{s}_{role}"] = np.array(
                [p[i, zero[i]].sum() for i in range(len(p))], np.float32)
            print(f"  seed {s} {role:8s}: pedestal median {st['pedestal_median']:.4f}  "
                  f"min bin mass {st['min_bin_mass_anywhere']:.2e}  "
                  f"median in zero bins {st['median_bin_mass_in_zero_bins']:.2e}  "
                  f"orders between them {st['orders_between_min_and_zero_bin_median']:.1f}")

    # --- THE NULL ARM: an untrained head of the same architecture ------------------------
    from tgnn_solv.config import TGNNSolvConfig
    from tgnn_solv.heads import SigmaProfileHead
    torch.manual_seed(0)
    cfg0 = TGNNSolvConfig()
    head = SigmaProfileHead(input_dim=256, cfg=cfg0)
    head.eval()
    with torch.no_grad():
        g = torch.randn(len(mols), 256)
        p0 = head(g)["p_shape"].numpy().astype(float)
    res["untrained_null"] = stats(p0, zero)
    dists["zero_bins_untrained"] = np.concatenate(
        [p0[i, zero[i]] for i in range(len(p0))]).astype(np.float32)
    dists["pedestal_untrained"] = np.array(
        [p0[i, zero[i]].sum() for i in range(len(p0))], np.float32)
    n0 = res["untrained_null"]
    print(f"\n  NULL, untrained head: pedestal median {n0['pedestal_median']:.4f}  "
          f"min mass {n0['min_bin_mass_anywhere']:.2e}")

    ks = list(res["per_seed"])
    if ks:
        print(f"\n{'=' * 78}\nGATE: is the softmax binding")
        for role in ("solute", "solvent"):
            ped = np.array([res["per_seed"][k][role]["pedestal_median"] for k in ks])
            mn = np.array([res["per_seed"][k][role]["min_bin_mass_anywhere"] for k in ks])
            med0 = np.array([res["per_seed"][k][role]["median_bin_mass_in_zero_bins"]
                             for k in ks])
            orders = np.array([res["per_seed"][k][role]
                               ["orders_between_min_and_zero_bin_median"] for k in ks])
            below = np.array([res["per_seed"][k][role]["share_of_zero_bins_below_1e_5"]
                              for k in ks])
            print(f"\n  {role}")
            print(f"   pedestal (share of mass in reference-zero bins): median over seeds "
                  f"{np.median(ped):.4f}, spread {ped.min():.4f}...{ped.max():.4f}")
            print(f"   against the untrained head {n0['pedestal_median']:.4f}  "
                  f"{'ok, training moves it' if ped.max() < 0.8 * n0['pedestal_median'] else 'NO DIFFERENT from initialisation'}")
            print(f"   the same head reaches {mn.min():.2e} somewhere on the corpus "
                  f"while placing {np.median(med0):.2e} in the zero bins")
            print(f"   difference: {np.median(orders):.1f} orders (per seed "
                  f"{orders.min():.1f}...{orders.max():.1f})")
            print(f"   share of reference-zero cells where the head is ALREADY below 1e-5: "
                  f"{100 * np.median(below):.2f}%")
            verdict = ("THE SOFTMAX IS NOT BINDING: the pedestal is the head's own choice, and a "
                       "zero-capable output would lift a constraint that does not exist"
                       if np.median(orders) > 2.0 else
                       "the softmax is binding: the head is against a floor, and a zero-capable output is worth trying")
            print(f"   VERDICT: {verdict}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    np.savez_compressed(a.out / "bin_mass_distributions.npz", **dists)
    print(f"\nwritten: {a.out} (+ bin_mass_distributions.npz, {len(dists)} arrays)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
