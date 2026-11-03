#!/usr/bin/env python
"""What the mass in reference-empty bins costs the closure.

A profile can depart from the tabulation in two places: in the bins the reference
occupies, and in the bins it leaves empty. This script substitutes the reference into one
part at a time, holding the learned cavity area fixed and renormalising the shape, and
rescores the activity coefficients on the full record set.

Setting exact zeros where the reference is zero removes the spurious population. Supplying
reference values where the reference is nonzero corrects the occupied part. Both
substitutions replace a part in full, so the pair is a decomposition of two complete
operations rather than an attribution over individual bins.

A control zeroes the same number of randomly chosen cells, which distinguishes the effect
of the location from the effect of removing mass and renormalising.

The script also reports the discrepancy between the sum of the parts and the whole, since
the profile-to-activity map is not additive over bins and the gap is the size of that
non-additivity.

Outputs pedestal_costs.json.

    python scripts/analysis/read_t2_pedestal_costs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from read_t1_closure_error_attribution import metrics, score  # noqa: E402
from run_sigma_profile_residual import build_model, learned_profiles  # noqa: E402
from tgnn_solv.data.utils import canonicalize                  # noqa: E402
from tgnn_solv.layers import CosmoSacLayer                      # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles          # noqa: E402

PROFILES = ROOT / "results/sigma_profile_artifact/sigma_profiles.csv"
RECORDS = ROOT / "results/published_idac_check/scored_records.csv"
CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
OUT = ROOT / "results/t2_softmax_floor"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    layer = CosmoSacLayer()
    layer.eval()
    n_bins = int(layer.n_bins)
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    d = pd.read_csv(RECORDS)
    T, m = d["T_K"].to_numpy(float), d["m"].to_numpy(float)
    s2 = d.solute_smiles.astype(str).to_numpy()
    s1 = d.solvent_smiles.astype(str).to_numpy()
    mols = sorted(set(s2) | set(s1))
    have = [x for x in mols if canonicalize(x) in table]
    REF = {}
    for x in have:
        r = np.asarray(table[canonicalize(x)][0], float)
        REF[x] = r / max(r.sum(), 1e-30)
    print(f"записей {len(d)}, молекул {len(mols)}, с эталоном {len(have)}")
    tpl = pd.read_csv(ROOT / "notebooks/data/processed/test.csv", nrows=1, low_memory=False)
    rng = np.random.default_rng(a.seed)
    res = {"n_records": int(len(d)), "n_null": a.n_null, "per_seed": {}}

    for s in a.seeds:
        ck = CKPT_DIR / f"grounded_a_seed{s}.pt"
        if not ck.exists():
            continue
        model, _, cfg = build_model(ck)
        as_solute, as_solvent = learned_profiles(model, cfg, mols, tpl)
        P = {"solute": {x: np.asarray(as_solute[i], float) for i, x in enumerate(mols)},
             "solvent": {x: np.asarray(as_solvent[i], float) for i, x in enumerate(mols)}}

        def edit(mode: str, draw=None) -> dict:
            """mode: base | zero_pedestal | fix_nonzero | full | null"""
            out = {}
            for role in ("solute", "solvent"):
                q = dict(P[role])
                for x in have:
                    p = P[role][x]
                    A = float(p.sum())
                    sh = (p / max(A, 1e-30)).copy()
                    z = REF[x] == 0.0
                    if mode == "zero_pedestal":
                        sh[z] = 0.0
                    elif mode == "fix_nonzero":
                        sh[~z] = REF[x][~z]
                    elif mode == "full":
                        sh = REF[x].copy()
                    elif mode == "null":
                        # снять массу из того же ЧИСЛА случайных клеток
                        idx = draw[:int(z.sum())]
                        sh[idx] = 0.0
                    q[x] = (sh / max(sh.sum(), 1e-30)) * A
                out[role] = q
            return out

        def aad(e) -> float:
            p2 = np.stack([e["solute"][x] for x in s2])
            p1 = np.stack([e["solvent"][x] for x in s1])
            return metrics(m, score(layer, p2, p2.sum(1), p1, p1.sum(1), T))["aad"]

        base = aad(edit("base"))
        row = {"base_aad": base}
        for mode in ("zero_pedestal", "fix_nonzero", "full"):
            row[f"aad_{mode}"] = aad(edit(mode))
        nulls = [aad(edit("null", rng.permutation(n_bins))) for _ in range(a.n_null)]
        row["null_aad"] = nulls
        gap = base - row["aad_full"]
        row["recovered_zero_pedestal"] = (base - row["aad_zero_pedestal"]) / max(gap, 1e-30)
        row["recovered_fix_nonzero"] = (base - row["aad_fix_nonzero"]) / max(gap, 1e-30)
        row["non_additivity"] = (row["recovered_zero_pedestal"]
                                 + row["recovered_fix_nonzero"] - 1.0)
        row["null_recovered"] = [float((base - x) / max(gap, 1e-30)) for x in nulls]
        res["per_seed"][str(s)] = row
        nr = np.asarray(row["null_recovered"])
        print(f"\nсид {s}: база {base:.4f}, вся форма {row['aad_full']:.4f} (разрыв {gap:.4f})")
        print(f"  обнулён пьедестал: {row['aad_zero_pedestal']:.4f}  "
              f"возвращено {100 * row['recovered_zero_pedestal']:+.1f}%")
        print(f"  починены НЕнулевые: {row['aad_fix_nonzero']:.4f}  "
              f"возвращено {100 * row['recovered_fix_nonzero']:+.1f}%")
        print(f"  сумма частей минус целое (неаддитивность): {100 * row['non_additivity']:+.1f}%")
        print(f"  НУЛЬ (столько же случайных клеток): {100 * nr.mean():+.1f}% в среднем, "
              f"{100 * nr.min():+.1f}...{100 * nr.max():+.1f}%; "
              f"p = {float((nr >= row['recovered_zero_pedestal']).mean()):.3f}")

    ks = list(res["per_seed"])
    if ks:
        print(f"\n{'=' * 78}\nСВОДКА по {len(ks)} сидам")
        for k in ("recovered_zero_pedestal", "recovered_fix_nonzero", "non_additivity"):
            v = np.array([res["per_seed"][x][k] for x in ks])
            print(f"  {k:26s}: {100 * v.mean():+6.1f}%  посидово "
                  f"{100 * v.min():+.1f}...{100 * v.max():+.1f}%")
        nv = np.array([np.mean(res["per_seed"][x]["null_recovered"]) for x in ks])
        print(f"  {'null':26s}: {100 * nv.mean():+6.1f}%  посидово "
              f"{100 * nv.min():+.1f}...{100 * nv.max():+.1f}%")
        ad = np.array([res["per_seed"][x]["aad_zero_pedestal"] - res["per_seed"][x]["base_aad"]
                       for x in ks])
        print(f"\n  ЦЕНА ПЬЕДЕСТАЛА в единицах AAD: {ad.mean():+.4f} "
              f"(посидово {ad.min():+.4f}...{ad.max():+.4f}); "
              f"сидовый пол референсного плеча 0.0632")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pedestal_costs.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                               encoding="utf8")
    print(f"\nзаписано: {a.out / 'pedestal_costs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
