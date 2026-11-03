#!/usr/bin/env python3
"""fig_donor_window -- the learned sigma-profile's area in the window COSMO-SAC's hydrogen-bond
term reads, against the reference tabulation's, ON THE ARMS OF RECORD.

WHY THIS FIGURE EXISTS
----------------------
Sec. 3.3 carries the manuscript's most chemistry-legible result and carried it in prose alone.
Because the 2002 kernel assigns a segment to the donor side by the threshold |sigma| > sigma_hb
alone, with no atom typing, mass there makes the hydrogen-bond term live on a pair whose chemistry
does not call for it.  A reader can check "the reference is exactly zero and the learned profile is
not" in one glance and cannot check it in a sentence, which is what the figure is for.

WHICH ARM, AND WHY IT CHANGED (2026-09-28)
------------------------------------------
This figure was first drawn from results/closure_ladder/placebo_profile_diagnosis.csv, a RETIRED
run whose sigma-supervision stream ran for zero steps.  Sec. 3.3 named that scope, and also named
the control it lacked: "a supervised arm separates the two".  That arm has now been measured --
grounded_a, seeds 42-46, the arms that carry the 1.93 -> 2.34 substitution contrast -- and it moves
the quantity by more than an order of magnitude:

    median donor-window area where the reference is exactly 0   36.9 -> 2.3 A^2
    median donor-window share of surface                        31.0% -> 1.5%
    Spearman(learned, reference) over the 31 solvents           -0.055 (p=0.77) -> +0.649 (p=8e-05)

So supervision recovers the ORDERING the unsupervised head did not have at all, and shrinks the
mass twentyfold, without closing the window.  Drawing the retired head would overstate the defect
by that factor, so the default table is now the rescore deposit.  Producer:
scripts/analysis/run_donor_window_rescore.py, which reproduces the retired schema row for row.

WHAT IS DRAWN
-------------
One row per solvent, ordered by how many scored rows it carries, so the solvents the substitution
contrast actually rests on are at the top.

  left of the axis   the REFERENCE tabulation's donor-window area, drawn in teal, labelled where it
                     is non-zero, because "the reference is empty" is a claim about this corpus and
                     not a law.
  right of the axis  the LEARNED profile's area in the same window, in salmon, seed-averaged over
                     42-46, with the fraction of that molecule's total surface printed at the bar
                     end.

Nothing is hard-coded, including which solvents are non-zero on the reference side.

Usage
-----
    MPLBACKEND=Agg python scripts/analysis/make_donor_window_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# THE JOURNAL'S GRAPHICS SPECIFICATION, applied before any figure is created. Without it matplotlib
# emits DejaVu Sans in Type 3, and both are violations; see acs_figure_style for what and why.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from acs_figure_style import apply as _acs_apply  # noqa: E402
_acs_apply()

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# House palette, shared with make_parity_figure.py and make_paradox_figures.py.
SALMON = "#E8A98C"   # the learned profile -- the arm the substitution replaces
TEAL = "#7FB5A6"     # the reference tabulation
INK = "#4D4D4D"
_STYLE = Path.home() / ".claude/skills/repo-to-paper/assets/softpastel.mplstyle"

#: Common names for the solvents the manuscript names in prose, so the figure reads as chemistry
#: rather than as SMILES.  Anything absent falls back to its SMILES, which is the honest default.
NAMES = {
    "C1CCCCC1": "cyclohexane", "CC(C)=O": "acetone", "Cc1ccccc1": "toluene",
    "C1CCOC1": "tetrahydrofuran", "O": "water", "CCO": "ethanol", "CC#N": "acetonitrile",
    "CN(C)C=O": "N,N-dimethylformamide", "CCOC(C)=O": "ethyl acetate", "C1COCCO1": "1,4-dioxane",
    "CO": "methanol", "CS(C)=O": "dimethyl sulfoxide", "CC(C)O": "propan-2-ol",
    "CCCCO": "butan-1-ol", "CCCO": "propan-1-ol", "ClCCl": "dichloromethane",
    "ClC(Cl)Cl": "chloroform", "CCCCCC": "hexane", "CC(=O)N(C)C": "N,N-dimethylacetamide",
    "c1ccccc1": "benzene", "CCCCCCC": "heptane", "CC(C)(C)O": "tert-butanol",
    "COC(C)=O": "methyl acetate", "CN1CCCC1=O": "N-methyl-2-pyrrolidone",
    "CCCOC(C)=O": "propyl acetate", "ClC(Cl)(Cl)Cl": "carbon tetrachloride",
    "CCOCC": "diethyl ether", "CC(C)=CC": "2-methyl-2-butene",
    "CCC(C)=O": "butan-2-one", "CCCCOC(C)=O": "butyl acetate", "ClCCCl": "1,2-dichloroethane",
    "CC(=O)OC(C)C": "isopropyl acetate", "Clc1ccccc1": "chlorobenzene",
    "CC(C)OC(C)C": "diisopropyl ether", "CC(=O)CC(C)=O": "pentane-2,4-dione",
    "CC(=O)c1ccccc1": "acetophenone", "O=C1CCCCC1": "cyclohexanone",
    "O=C1CCCO1": "gamma-butyrolactone", "COCOC": "dimethoxymethane",
    "CC1COC(=O)O1": "propylene carbonate", "CCCCCOC(C)=O": "pentyl acetate",
    # CCC(C)(C)C(C)C НАМЕРЕННО без названия: разбор его локантов от руки -- не то место, где
    # стоит рисковать, а SMILES и есть честное умолчание этой таблицы.
}


def load(path: Path, top: int) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.sort_values("n_rows", ascending=False).head(top).copy()
    d["name"] = [NAMES.get(s, s) for s in d["solvent_smiles"]]
    return d.iloc[::-1].reset_index(drop=True)   # bottom-up for barh


def draw(d: pd.DataFrame, out_dir: Path, stem: str) -> list[str]:
    if _STYLE.exists():
        plt.style.use(str(_STYLE))
        # AFTER style.use, NOT BEFORE: the shared style file resets rcParams wholesale, so a
        # typeface set earlier is silently discarded. The specification has to be applied last.
        _acs_apply()
    n = len(d)
    fig, ax = plt.subplots(figsize=(7.0, 0.22 * n + 1.05))
    y = range(n)
    ax.barh(y, -d["reference_donor_window_area"], color=TEAL, height=0.62,
            label="reference tabulation", zorder=3)
    ax.barh(y, d["learned_donor_window_area"], color=SALMON, height=0.62,
            label="learned profile", zorder=3)
    ax.axvline(0, color=INK, lw=0.9, zorder=4)

    for i, r in d.iterrows():
        ax.text(r["learned_donor_window_area"] + 1.4, i,
                f"{100 * r['learned_donor_fraction']:.1f}%", va="center", ha="left",
                fontsize=6.8, color=INK)
        if r["reference_donor_window_area"] > 0:
            ax.text(-r["reference_donor_window_area"] - 1.4, i,
                    f"{r['reference_donor_window_area']:.2f}", va="center", ha="right",
                    fontsize=6.8, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels(d["name"], fontsize=7.4)
    ax.set_xlabel(r"donor-window area, $\AA^2$   "
                  r"($\leftarrow$ reference tabulation $\;|\;$ learned profile $\rightarrow$)",
                  fontsize=8.2)
    ax.tick_params(axis="x", labelsize=7.2)
    # ОДНА шкала на обе половины. На отозванной голове выученные бары были в 4-5 раз длиннее
    # эталонных, и левой половине давался свой множитель просто чтобы поместились подписи. На
    # плечах записи величины сравнимы (5.4 против 9.8 A^2), и раздельные шкалы теперь искажали бы
    # ровно то сравнение, ради которого рисунок back-to-back.
    m = max(float(d["reference_donor_window_area"].max()),
            float(d["learned_donor_window_area"].max()))
    ax.set_xlim(-m * 1.45, m * 1.45)
    ax.set_ylim(-0.8, n - 0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.4, frameon=False)

    n_zero = int((d["reference_donor_window_area"] == 0).sum())
    n_occ = int((d["learned_donor_window_area"] > 0).sum())
    ax.set_title(f"Exactly empty in the reference for {n_zero} of these {n} solvents, "
                 f"occupied in the learned profile for {n_occ}",
                 fontsize=8.6, color=INK, pad=7)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        written.append(str(p))
    plt.close(fig)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path,
                    default=Path("results/donor_window_rescore/per_solvent.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("paper/figs"))
    ap.add_argument("--stem", default="fig_donor_window")
    ap.add_argument("--top", type=int, default=16, help="solvents to draw, by scored-row count")
    a = ap.parse_args()

    d = load(a.table, a.top)
    print(f"{len(d)} solvents drawn, of {len(pd.read_csv(a.table))} in the deposit")
    print(f"reference exactly zero in {(d['reference_donor_window_area'] == 0).sum()} of them; "
          f"learned fraction spans "
          f"{100 * d['learned_donor_fraction'].min():.1f}-{100 * d['learned_donor_fraction'].max():.1f}%")
    for p in draw(d, a.out_dir, a.stem):
        print("wrote", p)


if __name__ == "__main__":
    main()
