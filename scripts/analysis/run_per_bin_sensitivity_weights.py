#!/usr/bin/env python
"""Cost of placing sigma-profile mass in each bin, and the transport weight it implies.

For each destination bin the script moves a fixed fraction of the core's mass into that
bin, holds the cavity area fixed, and records the mean change in the predicted activity
coefficient. The source and the moved mass are the same for every destination, so the
destination is the only quantity that varies. The resulting profile R(k) is the cost of
placing mass at bin k.

The measurement uses finite differences rather than gradients. A gradient overstates the
cost by a factor of about four because of curvature in the step size, and its
edge-to-core ratio varies several-fold across training seeds, while the finite-difference
ratio reproduces to within about a sixth.

A weighted sum of absolute cumulative differences is the one-dimensional Wasserstein
distance under the metric in which adjacent grid points sit at distance w_k. If R is read
as a potential, the weight that prices displacement correctly is its discrete gradient,
w_k = |R(k+1) - R(k)|, and the unweighted distance is the special case of constant w.

Two row sets are used. The train rows give the weight; the scoring rows give a comparison
against the deposited sensitivity map on the surface that map was measured on. A control
repeats the calculation on random profiles of equal area.

Outputs summary.json and weights_t1.npy.

    python scripts/analysis/run_per_bin_sensitivity_weights.py
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
IDAC_RECORDS = ROOT / "results/published_idac_check/scored_records.csv"
TRAIN_ROWS = ROOT / "notebooks/data/processed/train.csv"
CKPT_DIR = ROOT / "checkpoints/e5_leakfree"
OUT = ROOT / "results/per_bin_sensitivity_weights"


def load_rows(which: str, table: dict, max_rows: int, seed: int) -> pd.DataFrame:
    """Строки (solute, solvent, T), у которых ОБА компонента есть в эталонной таблице."""
    if which == "idac":
        d = pd.read_csv(IDAC_RECORDS)
        tcol = "T_K"
    else:
        d = pd.read_csv(TRAIN_ROWS, low_memory=False)
        tcol = "temperature"
    k2 = [canonicalize(str(s)) for s in d.solute_smiles]
    k1 = [canonicalize(str(s)) for s in d.solvent_smiles]
    ok = np.array([(x in table) and (y in table) for x, y in zip(k2, k1)])
    d = d[ok].reset_index(drop=True)
    print(f"  набор {which}: {int(ok.sum())} из {len(ok)} строк имеют оба профиля в эталоне")
    if len(d) > max_rows:
        # Подвыборка СЛУЧАЙНАЯ с фиксированным сидом, а не head(): у train.csv строки упорядочены
        # по источнику, и head() взял бы один источник целиком.
        d = d.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return d.rename(columns={tcol: "T_K"})


def ln_gamma(layer, p2, p1, A2, A1, T) -> np.ndarray:
    with torch.no_grad():
        return layer.ln_gamma_inf(
            torch.tensor(p2, dtype=torch.float), torch.tensor(p1, dtype=torch.float),
            torch.tensor(A2, dtype=torch.float), torch.tensor(A1, dtype=torch.float),
            None, None, torch.tensor(T, dtype=torch.float)).numpy()


def residence_map(layer, p2, p1, A2, A1, T, core, frac: float) -> np.ndarray:
    """R[k] = средний |d ln gamma| от переноса frac массы ядра ЦЕЛИКОМ в бин k. Площадь сохранена."""
    n_bins = p1.shape[1]
    base = ln_gamma(layer, p2, p1, A2, A1, T)
    R = np.zeros(n_bins)
    moved = frac * p1[:, core].sum(1, keepdims=True)
    for k in range(n_bins):
        q = p1.copy()
        q[:, core] *= (1.0 - frac)
        q[:, k] += moved[:, 0]
        q *= (A1[:, None] / q.sum(1, keepdims=True))
        R[k] = float(np.abs(ln_gamma(layer, p2, q, A2, A1, T) - base).mean())
    return R


def weights_from_potential(R: np.ndarray, floor_frac: float) -> np.ndarray:
    """w_k = |R(k+1) - R(k)| с полом, нормированный на mean(w) = 1.

    Пол нужен не для красоты: по измерению R почти плоский внутри ядра, и |dR| там близок к нулю.
    Нулевой вес означал бы, что супервизия ВООБЩЕ не получает сигнала о том, в какой бин ядра
    положить массу, а в ядре лежит её основная часть -- форма ядра уехала бы свободно. Пол задаётся
    долей от средней величины |dR| и объявляется в предзаявке вместе с ней.

    Длина вектора -- n_bins, последний элемент дублирует предыдущий: слагаемых |C_k - C_k| в EMD
    ровно n_bins, а границ между бинами n_bins-1, и последнее слагаемое (k = n_bins-1) есть полная
    разность масс, которая на нормированных формах тождественно нуль. Значение веса там ни на что
    не влияет, но длина должна совпадать с длиной суммы.
    """
    dR = np.abs(np.diff(R))
    w = np.maximum(dR, floor_frac * dR.mean())
    w = np.append(w, w[-1])
    return w / w.mean()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", choices=["train", "idac", "both"], default="both")
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--floor-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    layer = CosmoSacLayer()
    layer.eval()
    n_bins = int(layer.n_bins)
    grid = layer.sigma_grid.numpy()
    core = np.abs(grid) <= float(layer.sigma_hb)
    corner = np.abs(grid) >= 0.016
    table = load_sigma_profiles(str(PROFILES), n_bins=n_bins)
    print(f"бинов {n_bins}; ядро {int(core.sum())}, угол {int(corner.sum())}; "
          f"перенос {100 * a.frac:.0f}% массы ядра, пол веса {100 * a.floor_frac:.0f}% от mean|dR|")

    res: dict = {"n_bins": n_bins, "fraction_moved": a.frac, "floor_frac": a.floor_frac,
                 "sigma_grid": grid.tolist(), "core": core.tolist(), "corner": corner.tolist(),
                 "seeds": list(a.seeds), "by_rows": {}}
    tpl = pd.read_csv(ROOT / "notebooks/data/processed/test.csv", nrows=1, low_memory=False)
    row_sets = ["train", "idac"] if a.rows == "both" else [a.rows]

    for which in row_sets:
        print(f"\n{'=' * 78}\nНАБОР СТРОК: {which}")
        d = load_rows(which, table, a.max_rows, a.seed)
        T = d["T_K"].to_numpy(float)
        k2 = [canonicalize(str(s)) for s in d.solute_smiles]
        A2 = np.array([table[k][1] for k in k2], float)
        p2 = np.stack([table[k][0] for k in k2])
        mols = sorted(set(d.solvent_smiles.astype(str)))
        print(f"  строк {len(d)}, растворителей {len(mols)}")

        per_seed, Rs = {}, []
        for s in a.seeds:
            ck = CKPT_DIR / f"grounded_a_seed{s}.pt"
            if not ck.exists():
                print(f"  сид {s}: чекпойнта нет, пропуск")
                continue
            model, _, cfg = build_model(ck)
            _, as_solvent = learned_profiles(model, cfg, mols, tpl)
            V = {m: np.asarray(as_solvent[i], float) for i, m in enumerate(mols)}
            p1 = np.stack([V[m] for m in d.solvent_smiles.astype(str)])
            A1 = p1.sum(1)
            R = residence_map(layer, p2, p1, A2, A1, T, core, a.frac)
            w = weights_from_potential(R, a.floor_frac)
            cc = float(R[corner].mean() / max(R[core].mean(), 1e-30))
            per_seed[str(s)] = {"R": R.tolist(), "w": w.tolist(),
                                "corner_over_core": cc,
                                "R_core_mean": float(R[core].mean()),
                                "R_corner_mean": float(R[corner].mean()),
                                "w_core_mean": float(w[:-1][core[:-1]].mean()),
                                "w_corner_mean": float(w[:-1][corner[:-1]].mean())}
            Rs.append(R)
            print(f"  сид {s}: R ядро {R[core].mean():.4f}  угол {R[corner].mean():.4f}  "
                  f"угол/ядро {cc:.1f}x;  вес ядро {w[:-1][core[:-1]].mean():.3f}  "
                  f"угол {w[:-1][corner[:-1]].mean():.3f}")

        # --- нулевое плечо: случайные профили той же площади -------------------------------
        rng = np.random.default_rng(a.seed)
        A1n = np.full(len(d), float(np.mean([table[k][1] for k in k2])))
        p1n = rng.dirichlet(np.ones(n_bins), size=len(d)) * A1n[:, None]
        Rn = residence_map(layer, p2, p1n, A2, A1n, T, core, a.frac)
        wn = weights_from_potential(Rn, a.floor_frac)
        cc_n = float(Rn[corner].mean() / max(Rn[core].mean(), 1e-30))
        print(f"  НУЛЬ (случайные профили): угол/ядро {cc_n:.2f}x")

        Rm = np.stack(Rs)
        wm = np.stack([np.asarray(v["w"]) for v in per_seed.values()])
        # посидовая устойчивость: по каждому бину разброс нормированного веса, и ранговое согласие
        from scipy.stats import spearmanr
        pairs = [float(spearmanr(wm[i], wm[j]).statistic)
                 for i in range(len(wm)) for j in range(i + 1, len(wm))]
        rho_null = float(spearmanr(wm.mean(0), wn).statistic)
        w_mean = wm.mean(0)
        res["by_rows"][which] = {
            "n_rows": int(len(d)), "n_solvents": len(mols),
            "per_seed": per_seed,
            "R_seed_mean": Rm.mean(0).tolist(),
            "w_seed_mean": w_mean.tolist(),
            "w_seed_sd": wm.std(0, ddof=1).tolist() if len(wm) > 1 else None,
            "corner_over_core_per_seed": [v["corner_over_core"] for v in per_seed.values()],
            "null": {"R": Rn.tolist(), "w": wn.tolist(), "corner_over_core": cc_n},
            "spearman_between_seeds": pairs,
            "spearman_weights_vs_null": rho_null,
        }
        ccs = np.array([v["corner_over_core"] for v in per_seed.values()])
        print(f"\n  ВОРОТА, набор {which}")
        print(f"   карта не плоская:      угол/ядро {ccs.min():.1f}...{ccs.max():.1f}x  "
              f"{'ok' if ccs.min() > 1.5 else 'ПЛОСКАЯ'}")
        print(f"   отличается от нуля:    медиана {np.median(ccs):.1f}x против {cc_n:.2f}x; "
              f"ранговое согласие с нулём rho = {rho_null:+.3f}  "
              f"{'ok' if abs(rho_null) < 0.5 else 'ПОХОЖА НА НУЛЬ -- вес про сетку'}")
        if pairs:
            print(f"   устойчива по сидам:    rho между сидами "
                  f"{min(pairs):+.3f}...{max(pairs):+.3f}  "
                  f"{'ok' if min(pairs) > 0.5 else 'НЕУСТОЙЧИВ -- вес сидозависим'}")
        print(f"   вес нормирован:        mean(w) = {w_mean.mean():.6f} (обязан быть 1)")
        print(f"   сверка с выложенной картой (12.9...15.0 на наборе idac, другой стимул): "
              f"{np.median(ccs):.1f}x")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf8")
    # вектор для T1 берётся с TRAIN-строк: он не должен выводиться на поверхности, на которой
    # арм будут читать.
    src = "train" if "train" in res["by_rows"] else row_sets[0]
    np.save(a.out / "weights_t1.npy", np.asarray(res["by_rows"][src]["w_seed_mean"], np.float64))
    print(f"\nзаписано: {a.out}\nвектор весов для T1 взят с набора {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
