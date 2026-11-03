#!/usr/bin/env python3
"""Departure from the pure-component reference state under a role-dependent profile.

An activity coefficient model must return unity for a molecule dissolved in itself. In
COSMO-SAC the residual term is a difference between a segment's activity in the mixture
and in the pure compound, so the property holds identically when one profile is used on
both sides of the calculation.

The network supplies a different profile depending on whether a molecule enters as solute
or as solvent, so pairing a molecule with itself pairs two different profiles and the
predicted activity departs from unity. This script measures that departure over the
molecules of the reference tabulation.

A control supplies a single profile to both sides, which must return exactly zero; the
measurement is not read unless it does. A second control repeats the calculation at a
higher segment-iteration count, which separates the departure from incomplete convergence
of the segment fixed point.

Outputs summary.json.

    python scripts/analysis/run_reference_state_violation.py
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

from tgnn_solv.data.utils import canonicalize  # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles  # noqa: E402

CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
PROFILES = ROOT / "results/sigma_profile_artifact/sigma_profiles.csv"
TEST = ROOT / "notebooks/data/processed/test.csv"
OUT = ROOT / "results/reference_state_violation"
COUNTS = (30, 300)
T_K = 298.15


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def ln_gamma_self(layer, p2: np.ndarray, p1: np.ndarray, n_iter: int) -> float:
    """ln gamma_inf(i в i) при профиле p2 в слоте растворяемого и p1 в слоте растворителя."""
    layer.n_iter_eval = n_iter
    t2 = torch.tensor(p2, dtype=torch.float).unsqueeze(0)
    t1 = torch.tensor(p1, dtype=torch.float).unsqueeze(0)
    A2 = t2.sum(dim=1)
    A1 = t1.sum(dim=1)
    with torch.no_grad():
        v = layer.ln_gamma_inf(t2, t1, A2, A1, None, None,
                               torch.tensor([T_K], dtype=torch.float))
    return float(v.item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--arm-glob", default="grounded_a_seed*.pt")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cks = sorted(a.ckpt_dir.glob(a.arm_glob))
    if not cks:
        print(f"нет чекпойнтов в {a.ckpt_dir}/{a.arm_glob}")
        return 1

    # ВОРОТА 3: набор по has_solubility, не по ln_x2.notna()
    test = pd.read_csv(TEST, low_memory=False)
    if "has_solubility" not in test.columns:
        print("нет колонки has_solubility -- отказываюсь угадывать набор")
        return 2
    scored = test[test["has_solubility"].astype(bool)]
    assert not (scored["ln_x2"] == 0.0).all(), "отобраны строки-заполнители"
    table = load_sigma_profiles(str(PROFILES), n_bins=51)
    seen = set(scored["solute_smiles"].astype(str)) | set(scored["solvent_smiles"].astype(str))
    mols = sorted({c for s in seen if (c := canonicalize(str(s))) is not None and c in table})
    if a.limit:
        mols = mols[: a.limit]
    print(f"ВОРОТА 3: молекул в наборе {len(mols)} (предзаявка: 71)")
    gate3 = len(mols) == 71 or bool(a.limit)

    res: dict = {"n_molecules": len(mols), "counts": list(COUNTS), "T_K": T_K,
                 "gate3_set_size": gate3, "per_seed": {}}
    tmpl = pd.read_csv(TEST, nrows=1, low_memory=False)
    gate1_worst = 0.0

    for ck in cks:
        model, ckd, cfg = build_model(ck)
        seed = int(ckd.get("seed", -1))
        layer = model.sle_solver.cosmo_sac_layer
        if layer is None:
            print(f"{ck.name}: плечо не на COSMO-SAC -- пропускаю")
            continue
        p_sol, p_slv = learned_profiles(model, cfg, mols, tmpl)
        rows = []
        for i, smi in enumerate(mols):
            rec = {"smiles": smi, "seed": seed,
                   "hellinger_role_gap": hellinger(p_sol[i], p_slv[i]),
                   "area_solute": float(p_sol[i].sum()),
                   "area_solvent": float(p_slv[i].sum())}
            for n in COUNTS:
                rec[f"ln_gamma_self_n{n}"] = ln_gamma_self(layer, p_sol[i], p_slv[i], n)
                # ВОРОТА 1: один профиль в оба слота обязан дать ноль
                rec[f"ctrl_solute_n{n}"] = ln_gamma_self(layer, p_sol[i], p_sol[i], n)
                rec[f"ctrl_solvent_n{n}"] = ln_gamma_self(layer, p_slv[i], p_slv[i], n)
            rows.append(rec)
        d = pd.DataFrame(rows)
        worst_ctrl = float(np.max([d[f"ctrl_{r}_n{n}"].abs().max()
                                   for r in ("solute", "solvent") for n in COUNTS]))
        gate1_worst = max(gate1_worst, worst_ctrl)
        res["per_seed"][str(seed)] = {
            "hellinger_role_gap_median": float(d.hellinger_role_gap.median()),
            "control_max_abs": worst_ctrl,
            **{f"median_abs_n{n}": float(d[f"ln_gamma_self_n{n}"].abs().median()) for n in COUNTS},
            **{f"p95_abs_n{n}": float(d[f"ln_gamma_self_n{n}"].abs().quantile(0.95)) for n in COUNTS},
            **{f"max_abs_n{n}": float(d[f"ln_gamma_self_n{n}"].abs().max()) for n in COUNTS},
            **{f"median_signed_n{n}": float(d[f"ln_gamma_self_n{n}"].median()) for n in COUNTS},
        }
        print(f"  сид {seed}: ролевой разрыв {d.hellinger_role_gap.median():.4f}  "
              f"контроль max|.| {worst_ctrl:.2e}  "
              + "  ".join(f"n={n} медиана|lnγ| {d[f'ln_gamma_self_n{n}'].abs().median():.4f}"
                          for n in COUNTS))
        a.out.mkdir(parents=True, exist_ok=True)
        d.to_csv(a.out / f"per_molecule_seed{seed}.csv", index=False)

    print(f"\nВОРОТА 1: одно-профильный контроль, худшее |.| по всем сидам = {gate1_worst:.3e}")
    res["gate1_control_is_zero"] = bool(gate1_worst < 1e-5)
    if not res["gate1_control_is_zero"]:
        print("  ПРОВАЛЕНЫ -- сломана оснастка, а не термодинамика. ЧТЕНИЕ ОТМЕНЕНО.")
    (a.out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    print(f"записано: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
