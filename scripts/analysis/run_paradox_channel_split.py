#!/usr/bin/env python3
"""Row-class split of the sigma-oracle substitution: which molecule the reference reached.

Discharges the Section S1 sentence that puts the enrolled +0.41 on the solvent channel:

    "the MAE difference is -0.000+-0.000 on the 37 neither-matched rows -- the two are the same
     checkpoint forwarded twice, and agree to 5e-6 there -- +0.431+-0.036 on the 5274 solvent-only
     rows, and +0.006+-0.025 on the 297 both-matched ones."

and the match-rule counts that precede it (5571 solvent matches, 99.3%, 64 of 70 distinct solvents;
297 solute matches, 5.3%, 7 of 147 distinct solutes), which also feed the "~7 solutes" of Sec. 3.4.

Row classes come from the SAME rule the substitution itself used
(`tgnn_solv.sigma_oracle.build_oracle_tensors`: canonical SMILES against the reference profile
table). When the per-row export carries `sigma_oracle_mask_solute` / `sigma_oracle_mask_solvent`
-- newer exports do, the published tree does not -- those columns are used instead and the
recomputed rule is checked against them.

The results root is a flag. Default is the published tree; point `--root` at
results/e5_sigma_grounding_leakfree to discharge the leak-free re-run with the same command.

Reproduces, on the published tree at three seeds (verified 2026-08-10):
  neither -0.000 +- 0.000 (n=37), solvent-only +0.431 +- 0.036 (n=5274), both +0.006 +- 0.025 (n=297)
  agreement bound on the neither rows: max |MAE difference| = 3.7e-6 (<= 5e-6)
`--expect-published` asserts exactly that and exits non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tgnn_solv.data.utils import canonicalize  # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles  # noqa: E402

DEFAULT_ROOT = REPO / "results" / "e5_sigma_grounding"
DEFAULT_PROFILES = REPO / "results" / "sigma_profile_artifact" / "sigma_profiles.csv"

CLASSES = ("neither", "solvent_only", "solute_only", "both")

# The values Section S1 prints today, off the published tree at seeds 42/43/44.
PUBLISHED = {
    "n_rows": 5608,
    "n_solvent_match": 5571,
    "n_solute_match": 297,
    "n_distinct_solvents": 70,
    "n_distinct_solvents_matched": 64,
    "n_distinct_solutes": 147,
    "n_distinct_solutes_matched": 7,
    "class_n": {"neither": 37, "solvent_only": 5274, "both": 297, "solute_only": 0},
    "class_mean": {"neither": -0.000, "solvent_only": +0.431, "both": +0.006},
    "class_sd": {"neither": 0.000, "solvent_only": 0.036, "both": 0.025},
    "neither_agreement_bound": 5e-6,
}


def discover_seeds(root: Path, grounded: str, oracle: str) -> list[int]:
    seeds = []
    for d in sorted(root.glob("seed_*")):
        m = re.fullmatch(r"seed_(\d+)", d.name)
        if m and (d / grounded).exists() and (d / oracle).exists():
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def side_masks(g: pd.DataFrame, o: pd.DataFrame, vt: set[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """(solute_matched, solvent_matched, provenance). Prefer exported masks; else recompute."""
    recomputed = {}
    uniq = {}
    for s in pd.unique(pd.concat([g["solute_smiles"], g["solvent_smiles"]], ignore_index=True)):
        key = canonicalize(str(s))
        uniq[s] = key is not None and key in vt
    recomputed["solute"] = g["solute_smiles"].map(uniq).to_numpy(dtype=bool)
    recomputed["solvent"] = g["solvent_smiles"].map(uniq).to_numpy(dtype=bool)

    if {"sigma_oracle_mask_solute", "sigma_oracle_mask_solvent"} <= set(o.columns):
        su = o["sigma_oracle_mask_solute"].to_numpy(dtype=bool)
        sv = o["sigma_oracle_mask_solvent"].to_numpy(dtype=bool)
        agree = bool((su == recomputed["solute"]).all() and (sv == recomputed["solvent"]).all())
        return su, sv, f"exported columns (recomputed rule agrees: {agree})"
    return recomputed["solute"], recomputed["solvent"], "recomputed from the reference table"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"results root holding seed_*/ (default: {DEFAULT_ROOT.relative_to(REPO)})")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seeds to use; default is every seed_* under --root carrying both CSVs")
    ap.add_argument("--grounded-name", default="grounded_a_predictions.csv")
    ap.add_argument("--oracle-name", default="oracle_predictions.csv")
    ap.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES,
                    help="reference sigma-profile table defining the match rule")
    ap.add_argument("--out", type=Path, default=None, help="write the full result as JSON here")
    ap.add_argument("--expect-published", action="store_true",
                    help="assert the values Section S1 prints today; non-zero exit on mismatch")
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else (REPO / args.root)
    profiles = args.profiles if args.profiles.is_absolute() else (REPO / args.profiles)
    if not root.is_dir():
        ap.error(f"--root does not exist: {root}")
    seeds = args.seeds if args.seeds else discover_seeds(root, args.grounded_name, args.oracle_name)
    if not seeds:
        ap.error(f"no seed_* directory under {root} carries both "
                 f"{args.grounded_name} and {args.oracle_name}")

    table = load_sigma_profiles(str(profiles))
    vt = set(table)
    print(f"root      : {root}")
    print(f"seeds     : {' '.join(str(s) for s in seeds)}")
    print(f"reference : {profiles.name}, {len(vt)} canonical SMILES")

    per_seed = []
    for seed in seeds:
        d = root / f"seed_{seed}"
        g = pd.read_csv(d / args.grounded_name)
        o = pd.read_csv(d / args.oracle_name)
        g = g[g["has_solubility"].astype(bool)].reset_index(drop=True)
        o = o[o["has_solubility"].astype(bool)].reset_index(drop=True)
        if len(g) != len(o):
            raise SystemExit(f"seed {seed}: {len(g)} grounded rows vs {len(o)} oracle rows")
        if not (g["solute_smiles"].values == o["solute_smiles"].values).all():
            raise SystemExit(f"seed {seed}: solute row order differs between the two arms")
        if not (g["solvent_smiles"].values == o["solvent_smiles"].values).all():
            raise SystemExit(f"seed {seed}: solvent row order differs between the two arms")

        su, sv, prov = side_masks(g, o, vt)
        applied = o["sigma_oracle_applied"].astype(bool).to_numpy()
        if not (applied == (su | sv)).all():
            raise SystemExit(
                f"seed {seed}: the deposited substitution flag is not the union of the two side "
                f"matches ({int(applied.sum())} flagged vs {int((su | sv).sum())} matched) -- the "
                f"reference table given by --profiles is not the one the run substituted from")

        cls = np.where(su & sv, "both", np.where(sv, "solvent_only", np.where(su, "solute_only", "neither")))
        ae_g = g["abs_error"].to_numpy(dtype=float)
        ae_o = o["abs_error"].to_numpy(dtype=float)

        rec = {
            "seed": seed,
            "mask_provenance": prov,
            "n_rows": int(len(g)),
            "n_solvent_match": int(sv.sum()),
            "n_solute_match": int(su.sum()),
            "n_applied_flag": int(applied.sum()),
            "n_distinct_solutes": int(g["solute_smiles"].nunique()),
            "n_distinct_solutes_matched": int(g.loc[su, "solute_smiles"].nunique()),
            "n_distinct_solvents": int(g["solvent_smiles"].nunique()),
            "n_distinct_solvents_matched": int(g.loc[sv, "solvent_smiles"].nunique()),
            "classes": {},
        }
        for c in CLASSES:
            m = cls == c
            if m.sum() == 0:
                rec["classes"][c] = {"n": 0}
                continue
            rec["classes"][c] = {
                "n": int(m.sum()),
                "mae_grounded": float(ae_g[m].mean()),
                "mae_oracle": float(ae_o[m].mean()),
                "mae_difference": float(ae_o[m].mean() - ae_g[m].mean()),
                "max_abs_row_difference": float(np.abs(ae_o[m] - ae_g[m]).max()),
            }
        per_seed.append(rec)

        print(f"\nseed {seed}  ({prov})")
        print(f"  rows {rec['n_rows']}   solvent matched {rec['n_solvent_match']} "
              f"({100 * rec['n_solvent_match'] / rec['n_rows']:.1f}%; "
              f"{rec['n_distinct_solvents_matched']} of {rec['n_distinct_solvents']} distinct)   "
              f"solute matched {rec['n_solute_match']} "
              f"({100 * rec['n_solute_match'] / rec['n_rows']:.1f}%; "
              f"{rec['n_distinct_solutes_matched']} of {rec['n_distinct_solutes']} distinct)")
        for c in CLASSES:
            e = rec["classes"][c]
            if e["n"] == 0:
                continue
            print(f"  {c:13s} n={e['n']:5d}  grounded={e['mae_grounded']:.4f}  "
                  f"oracle={e['mae_oracle']:.4f}  difference={e['mae_difference']:+.6f}")

    summary = {"classes": {}}
    for c in CLASSES:
        vals = [r["classes"][c]["mae_difference"] for r in per_seed if r["classes"][c]["n"] > 0]
        ns = sorted({r["classes"][c]["n"] for r in per_seed})
        if not vals:
            summary["classes"][c] = {"n_per_seed": [0], "n_seeds": 0}
            continue
        summary["classes"][c] = {
            "n_per_seed": ns,
            "n_seeds": len(vals),
            "mean": float(np.mean(vals)),
            "sd_population": float(np.std(vals, ddof=0)),
            "sd_sample": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "per_seed": {str(r["seed"]): r["classes"][c]["mae_difference"] for r in per_seed},
            "max_abs_mae_difference": float(np.max(np.abs(vals))),
        }

    print("\n=== MAE difference (oracle minus grounded), mean +- population sd over "
          f"{len(per_seed)} seeds ===")
    for c in CLASSES:
        e = summary["classes"][c]
        if e["n_seeds"] == 0:
            print(f"  {c:13s} n=0 in every seed")
            continue
        ns = "/".join(str(n) for n in e["n_per_seed"])
        tail = "" if e["sd_sample"] is None else f"   (sample sd {e['sd_sample']:.3f})"
        print(f"  {c:13s} n={ns:>5s}  {e['mean']:+.3f} +- {e['sd_population']:.3f}{tail}")
    nb = summary["classes"]["neither"]
    if nb["n_seeds"]:
        print(f"\n  the two arms agree on the neither-matched rows to "
              f"{nb['max_abs_mae_difference']:.1e} in MAE "
              f"(largest per-row |difference| "
              f"{max(r['classes']['neither']['max_abs_row_difference'] for r in per_seed):.1e})")

    out = {
        "root": str(root),
        "profiles": str(profiles),
        "seeds": seeds,
        "grounded_name": args.grounded_name,
        "oracle_name": args.oracle_name,
        "per_seed": per_seed,
        "summary": summary,
    }
    if args.out:
        outp = args.out if args.out.is_absolute() else (REPO / args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {outp}")

    if args.expect_published:
        bad = []
        r0 = per_seed[0]
        for k in ("n_rows", "n_solvent_match", "n_solute_match", "n_distinct_solvents",
                  "n_distinct_solvents_matched", "n_distinct_solutes", "n_distinct_solutes_matched"):
            if r0[k] != PUBLISHED[k]:
                bad.append(f"{k}: {r0[k]} != {PUBLISHED[k]}")
        for c, n in PUBLISHED["class_n"].items():
            got = summary["classes"][c]["n_per_seed"]
            if got != [n] and not (n == 0 and got == [0]):
                bad.append(f"n({c}): {got} != [{n}]")
        for c, mu in PUBLISHED["class_mean"].items():
            e = summary["classes"][c]
            if round(e["mean"], 3) != round(mu, 3):
                bad.append(f"mean({c}): {e['mean']:+.4f} != {mu:+.3f}")
            if round(e["sd_population"], 3) != round(PUBLISHED["class_sd"][c], 3):
                bad.append(f"sd({c}): {e['sd_population']:.4f} != {PUBLISHED['class_sd'][c]:.3f}")
        if nb["max_abs_mae_difference"] > PUBLISHED["neither_agreement_bound"]:
            bad.append(f"neither agreement {nb['max_abs_mae_difference']:.2e} exceeds "
                       f"{PUBLISHED['neither_agreement_bound']:.0e}")
        if bad:
            print("\nEXPECT-PUBLISHED FAILED:")
            for b in bad:
                print(f"  {b}")
            return 1
        print("\nexpect-published: every value Section S1 prints is reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
