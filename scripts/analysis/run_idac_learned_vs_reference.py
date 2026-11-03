#!/usr/bin/env python
"""Score infinite-dilution activity coefficients from learned and from reference profiles.

The quantity is the residual term of COSMO-SAC evaluated on a pair of sigma-profiles. No
crystal branch enters it: neither the melting temperature, nor the enthalpy of fusion, nor
the ideal term. The comparison therefore isolates the profile itself, separately from any
joint fitting of the profile with the crystal properties that solubility training couples.

Four arms are scored on the same records. The reference arm uses the tabulated profile for
both components. The learned arm uses the network's solute-role profile for the solute and
its solvent-role profile for the solvent. The two role arms use one role on both sides,
which separates the effect of the role-dependent representation from the effect of
learning the profile at all.

A gate recomputes the reference arm from the tabulation and compares it against the
deposited residual term; the learned arms are not read unless the two agree to 1e-6.

Outputs summary.json with per-seed and aggregated errors.

    python scripts/analysis/run_idac_learned_vs_reference.py
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

from run_sigma_profile_residual import build_model, learned_profiles          # noqa: E402
from tgnn_solv.data.utils import canonicalize                                 # noqa: E402
from tgnn_solv.layers import CosmoSacLayer                                    # noqa: E402
from tgnn_solv.sigma_oracle import load_sigma_profiles                        # noqa: E402

CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
PROFILES = ROOT / "results/sigma_profile_artifact/sigma_profiles.csv"
RECORDS = ROOT / "results/published_idac_check/scored_records.csv"
TEST = ROOT / "notebooks/data/processed/test.csv"
OUT = ROOT / "results/idac_learned_vs_reference"

ARMS = ["reference_vt2005", "learned", "learned_role_solute", "learned_role_solvent"]


def per_seed_index(df: pd.DataFrame) -> list[int]:
    return sorted(int(s) for s in df["seed"].unique())


def metrics(m: np.ndarray, g: np.ndarray) -> dict:
    ok = np.isfinite(m) & np.isfinite(g)
    m, g = np.asarray(m, float)[ok], np.asarray(g, float)[ok]
    if m.size == 0:
        return {"n": 0}
    d = g - m
    return {"n": int(m.size), "aad": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean()),
            "r2": float(1.0 - (d ** 2).sum() / ((m - m.mean()) ** 2).sum())}


def score(layer, p2, A2, p1, A1, T, batch: int = 2048) -> np.ndarray:
    """ln gamma_2^inf по паре профилей, конвенция `res` (V=None), как в депозите."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=RECORDS,
                    help="таблица записей; нужны solute_smiles, solvent_smiles, T_K, m")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--arm-glob", default="grounded_a_seed*.pt")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="разрешить замену депозита с бОльшим набором сидов")
    a = ap.parse_args()

    d = pd.read_csv(a.records)
    template = pd.read_csv(TEST, nrows=1, low_memory=False)
    layer = CosmoSacLayer()
    layer.eval()
    n_bins = int(getattr(layer, "n_bins", 51))
    n_iter = int(layer.n_iter_eval)
    print(f"записей {len(d)}, слой COSMO-SAC-2002, сегментных итераций при eval: {n_iter}")

    # --- эталон: тот же артефакт, но ключ -- канонический SMILES, а не InChIKey -------------
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    keys2 = [canonicalize(str(s)) for s in d.solute_smiles]
    keys1 = [canonicalize(str(s)) for s in d.solvent_smiles]
    have = np.array([(k2 in table) and (k1 in table) for k2, k1 in zip(keys2, keys1)])
    print(f"  эталонный профиль с обеих сторон по SMILES: {have.sum()} из {len(d)}")
    d = d[have].reset_index(drop=True)
    keys2 = [k for k, h in zip(keys2, have) if h]
    keys1 = [k for k, h in zip(keys1, have) if h]

    T = d["T_K"].to_numpy(float)
    m = d["m"].to_numpy(float)
    ref_p2 = np.stack([table[k][0] for k in keys2])
    ref_A2 = np.array([table[k][1] for k in keys2], dtype=float)
    ref_p1 = np.stack([table[k][0] for k in keys1])
    ref_A1 = np.array([table[k][1] for k in keys1], dtype=float)
    g_ref = score(layer, ref_p2, ref_A2, ref_p1, ref_A1, T)

    # СТЕРЕО-НЕОДНОЗНАЧНЫЕ СТРОКИ ИСКЛЮЧАЮТСЯ, И ЭТО НЕ КОСМЕТИКА. Депозит ключует VT-2005 по
    # InChIKey, здесь ключ -- канонический SMILES. Там, где два маршрута выбирают РАЗНЫЕ записи,
    # молекула не определена однозначно своим SMILES: на этом корпусе это декалин, чьи цис- и
    # транс-формы делят SMILES без стереохимии, но несут разные InChIKey и разные профили.
    # Выученная голова тоже читает SMILES, поэтому на такой строке её профиль неоднозначен ровно
    # так же. Строка выбрасывается из ОБЕИХ сторон, и её число печатается.
    # ВОРОТА ЕСТЬ ТОЛЬКО ТАМ, ГДЕ ЕСТЬ С ЧЕМ СВЕРЯТЬСЯ. У депозита PGL6ed есть столбец g_res,
    # посчитанный run_published_idac_closure_check.py, и пересчёт обязан его воспроизвести. У
    # внешнего набора (Brouwer) такого столбца нет; тогда ворота не «пройдены», а ОТСУТСТВУЮТ, и
    # это печатается, а не замалчивается.
    has_gate = "g_res" in d.columns
    ambiguous: list[str] = []
    gate = float("nan")
    amb = np.zeros(len(d), dtype=bool) if not has_gate else (
        np.abs(g_ref - d["g_res"].to_numpy(float)) > 1e-6)
    if not has_gate:
        print("  ВОРОТА ОТСУТСТВУЮТ: столбца g_res в наборе нет, пересчёт эталона сверять не с чем")
    if amb.any():
        names = sorted({f"{r.solute_smiles} | {r.solvent_smiles}"
                        for r in d[amb].itertuples()})
        print(f"  стерео-неоднозначных строк отброшено: {int(amb.sum())} из {len(d)}")
        for nm in names[:6]:
            print(f"      {nm}")
        d = d[~amb].reset_index(drop=True)
        keys2 = [k for k, bad in zip(keys2, amb) if not bad]
        keys1 = [k for k, bad in zip(keys1, amb) if not bad]
        ref_p2, ref_A2 = ref_p2[~amb], ref_A2[~amb]
        ref_p1, ref_A1 = ref_p1[~amb], ref_A1[~amb]
        g_ref = g_ref[~amb]
        T = d["T_K"].to_numpy(float)
        m = d["m"].to_numpy(float)
        ambiguous = names

    if has_gate:
        gate = float(np.nanmax(np.abs(g_ref - d["g_res"].to_numpy(float))))
        print(f"  ВОРОТА: пересчёт эталона против депонированного g_res, max|d| = {gate:.2e}"
              f"  {'ok' if gate < 1e-6 else 'РАСХОДИТСЯ -- читать выученное плечо нельзя'}")
        if gate >= 1e-6:
            return 1

    mols = sorted(set(d.solute_smiles.astype(str)) | set(d.solvent_smiles.astype(str)))
    print(f"  уникальных молекул для выученной головы: {len(mols)}")

    cks = sorted(a.ckpt_dir.glob(a.arm_glob))
    rows, per_seed = [], {}
    for ck in cks:
        model, ckd, cfg = build_model(ck)
        seed = int(ckd.get("seed", -1))
        if a.seeds and seed not in a.seeds:
            continue
        print(f"\nсид {seed} ...", flush=True)
        as_solute, as_solvent = learned_profiles(model, cfg, mols, template)
        P_u = {s: np.asarray(as_solute[i], float) for i, s in enumerate(mols)}
        P_v = {s: np.asarray(as_solvent[i], float) for i, s in enumerate(mols)}
        s2 = d.solute_smiles.astype(str).to_numpy()
        s1 = d.solvent_smiles.astype(str).to_numpy()

        def pack(src2, src1):
            p2 = np.stack([src2[s] for s in s2])
            p1 = np.stack([src1[s] for s in s1])
            return p2, p2.sum(1), p1, p1.sum(1)

        g = {"reference_vt2005": g_ref,
             "learned": score(layer, *pack(P_u, P_v), T),
             "learned_role_solute": score(layer, *pack(P_u, P_u), T),
             "learned_role_solvent": score(layer, *pack(P_v, P_v), T)}
        per_seed[seed] = {arm: metrics(m, g[arm]) for arm in ARMS}
        for arm in ARMS:
            e = per_seed[seed][arm]
            print(f"  {arm:22s} AAD {e['aad']:.4f}  RMSE {e['rmse']:.4f}  "
                  f"R2 {e['r2']:+.4f}  смещение {e['bias']:+.4f}")
        for arm in ARMS:
            rows.append(pd.DataFrame({"seed": seed, "arm": arm, "m": m, "g": g[arm],
                                      "solute_smiles": s2, "solvent_smiles": s1,
                                      "aqueous": d["aqueous"].to_numpy()}))

    if not per_seed:
        print("нет чекпойнтов")
        return 1
    per_row = pd.concat(rows, ignore_index=True)

    # НЕ ЗАТИРАТЬ МНОГОСИДОВЫЙ ДЕПОЗИТ ОДНОСИДОВЫМ ПРОГОНОМ. Регрессионная проверка с --seeds 42
    # после правки ворот записалась в тот же --out и уничтожила пятисидовый депозит; числа из него
    # потом едва не были прочитаны как пятисидовые. Односидовый прогон -- это проверка тракта, а
    # депозит он заменять не должен: либо --out в отдельную папку, либо --force.
    prev = a.out / "summary.json"
    if prev.exists() and not a.force:
        try:
            old_seeds = set(json.loads(prev.read_text(encoding="utf8")).get("seeds", []))
        except Exception:
            old_seeds = set()
        new_seeds = set(int(s) for s in per_seed_index(per_row))
        if old_seeds - new_seeds:
            print(f"ОТКАЗ: в {a.out} лежит депозит на сидах {sorted(old_seeds)}, "
                  f"а этот прогон несёт {sorted(new_seeds)}. Затирать не буду.\n"
                  f"  либо --out в другую папку, либо --force, если замена намеренна.")
            return 1
    a.out.mkdir(parents=True, exist_ok=True)
    per_row.to_csv(a.out / "per_row.csv", index=False)

    print(f"\n{'плечо':<22}{'AAD':>9}{'RMSE':>9}{'R2':>9}{'смещение':>11}")
    summary = {"n_records": int(len(d)), "n_molecules": len(mols),
               "dropped_stereo_ambiguous": ambiguous,
               "seeds": sorted(per_seed), "segment_iterations_eval": n_iter,
               "convention": "residual only (no Staverman-Guggenheim; V=None both sides)",
               "closure": "tgnn_solv.layers.CosmoSacLayer() defaults, identical for every arm",
               "reference_gate_max_abs_diff_vs_deposited_g_res": gate,
               "published_reference_aad_cosmo_sac_2002_HF_TZVP": 1.7457,
               "arms": {}}
    for arm in ARMS:
        aad = [per_seed[s][arm]["aad"] for s in per_seed]
        r2 = [per_seed[s][arm]["r2"] for s in per_seed]
        summary["arms"][arm] = {
            "aad_mean": float(np.mean(aad)),
            "aad_sd": float(np.std(aad, ddof=1)) if len(aad) > 1 else 0.0,
            "r2_mean": float(np.mean(r2)),
            "per_seed": {int(s): per_seed[s][arm] for s in per_seed}}
        e = summary["arms"][arm]
        print(f"{arm:<22}{e['aad_mean']:>9.4f}{np.mean([per_seed[s][arm]['rmse'] for s in per_seed]):>9.4f}"
              f"{e['r2_mean']:>+9.4f}{np.mean([per_seed[s][arm]['bias'] for s in per_seed]):>+11.4f}")

    ref, lrn = summary["arms"]["reference_vt2005"]["aad_mean"], summary["arms"]["learned"]["aad_mean"]
    summary["learned_minus_reference_aad"] = float(lrn - ref)
    summary["learned_over_reference_aad"] = float(lrn / ref) if ref else None
    print(f"\n  выученный минус эталон, AAD: {lrn - ref:+.4f}  ({lrn / ref:.2f}x)")
    print("  для масштаба: опубликованный COSMO-SAC-2002 на HF-TZVP даёт AAD 1.7457")
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    print(f"\nзаписано: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
