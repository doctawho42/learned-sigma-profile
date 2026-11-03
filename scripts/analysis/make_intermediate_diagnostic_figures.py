#!/usr/bin/env python3
"""Three figures of the manuscript, drawn from deposited measurements.

fig_pedestal_learned shows the distribution of profile mass in bins where the reference is
exactly zero, for every seed and both roles, against an untrained network of the same
architecture, and the same mass summed per molecule.

fig_cost_map shows the cost of placing mass at each point of the charge-density axis,
against random profiles of equal area, and the transport weight that cost implies.

fig_attribution shows the share of the activity-coefficient gap recovered by substituting
the reference into different parts of the grid. Its two panels are separated and each
names its own control, because the two deposits define different operations: one zeroes a
number of randomly chosen cells, the other substitutes reference values into a randomly
chosen third of the bins.

Everything is drawn from the deposits; no checkpoint is loaded here.

    python scripts/analysis/make_intermediate_diagnostic_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from acs_figure_style import DOUBLE_COL_PT, SINGLE_COL_PT, apply  # noqa: E402

FLOOR = ROOT / "results/t2_softmax_floor/summary.json"
DISTS = ROOT / "results/t2_softmax_floor/bin_mass_distributions.npz"
COSTS = ROOT / "results/t2_softmax_floor/pedestal_costs.json"
WEIGHTS = ROOT / "results/per_bin_sensitivity_weights/summary.json"
ATTRIB = ROOT / "results/per_bin_sensitivity_weights/closure_error_attribution.json"
FIGS = ROOT / "paper/figs"

SEEDS = ["42", "43", "44", "45", "46"]
INK, ACC, NULLC, GREY = "#1a1a1a", "#b2182b", "#9a9a9a", "#d6d6d6"


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    print(f"  written: paper/figs/{name}.pdf and .png")


def ecdf(x: np.ndarray):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, x.size + 1) / x.size


def fig_pedestal(plt) -> None:
    d = json.loads(FLOOR.read_text(encoding="utf8"))
    z = np.load(DISTS)
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_PT / 72.0, 2.25),
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42})

    # (a) where the mass in reference-zero cells actually sits
    for i, s in enumerate(SEEDS):
        for role in ("solute", "solvent"):
            k = f"zero_bins_seed{s}_{role}"
            if k not in z:
                continue
            x, y = ecdf(np.maximum(z[k], 1e-13))
            axa.semilogx(x, y, color=INK, lw=0.6, alpha=0.55,
                         label="trained head, 5 seeds $\\times$ 2 roles"
                               if (i == 0 and role == "solute") else None)
    x, y = ecdf(np.maximum(z["zero_bins_untrained"], 1e-13))
    axa.semilogx(x, y, color=ACC, lw=1.4, label="untrained head, same architecture")
    axa.axvline(1e-5, color=NULLC, lw=0.6, ls=":")
    axa.text(1.3e-5, 0.07, "$10^{-5}$", fontsize=5.5, color="#6a6a6a")
    axa.set_xlabel("profile mass in a bin where the reference is exactly zero")
    axa.set_ylabel("cumulative fraction of such bins")
    axa.set_xlim(1e-13, 1.0)
    axa.set_ylim(0, 1.02)
    axa.legend(frameon=False, fontsize=5.5, loc="upper left")
    axa.set_title("(a) mass placed where the reference has none", loc="left", fontsize=6.5)

    # (b) the pedestal summed over each molecule
    pos, labels, data = [], [], []
    for j, role in enumerate(("solute", "solvent")):
        vals = np.concatenate([z[f"pedestal_seed{s}_{role}"] for s in SEEDS
                               if f"pedestal_seed{s}_{role}" in z])
        data.append(vals)
        pos.append(j)
        labels.append(f"trained\n({role})")
    data.append(z["pedestal_untrained"])
    pos.append(2.4)
    labels.append("untrained")
    bp = axb.boxplot(data, positions=pos, widths=0.55, showfliers=False,
                     patch_artist=True, medianprops={"color": INK, "lw": 1.0})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(ACC if i == 2 else GREY)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.6)
    axb.set_xticks(pos)
    axb.set_xticklabels(labels, fontsize=5.5)
    axb.set_ylabel("pedestal (share of mass)")
    axb.set_ylim(0, 1.0)
    axb.set_title("(b) the pedestal, per molecule", loc="left", fontsize=6.5)
    share = 100 * d["share_reference_zero_cells"]
    axb.text(0.98, 0.97, f"{share:.0f}% of reference cells\nare exactly zero",
             transform=axb.transAxes, ha="right", va="top", fontsize=5.5, color="#6a6a6a")
    save(fig, "fig_pedestal_learned")
    plt.close(fig)


def fig_cost_map(plt) -> None:
    w = json.loads(WEIGHTS.read_text(encoding="utf8"))
    g = np.asarray(w["sigma_grid"], float) * 1e3           # e/nm^2, a convenient scale for the axis label
    b = w["by_rows"]["train"]
    per = [np.asarray(v["R"], float) for v in b["per_seed"].values()]
    R = np.stack(per)
    W = np.stack([np.asarray(v["w"], float) for v in b["per_seed"].values()])
    Rn = np.asarray(b["null"]["R"], float)
    hb = 1e3 * float(np.max(np.abs(np.asarray(w["sigma_grid"], float))[np.asarray(w["core"])]))

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(SINGLE_COL_PT / 72.0, 3.4), sharex=True,
                                   gridspec_kw={"hspace": 0.34})
    axa.fill_between(g, R.min(0), R.max(0), color=GREY, lw=0)
    axa.plot(g, R.mean(0), color=INK, lw=1.2, label="learned profiles, 5 seeds")
    axa.plot(g, Rn, color=NULLC, lw=1.0, ls="--", label="random profiles, same area")
    for ax in (axa, axb):
        for sgn in (-1, 1):
            ax.axvline(sgn * hb, color=ACC, lw=0.6, ls=":")
    axa.set_ylabel("$\\langle|\\Delta \\ln \\gamma|\\rangle$")
    axa.legend(frameon=False, fontsize=5.5, loc="upper center")
    axa.set_title("(a) cost of placing mass at $\\sigma$", loc="left", fontsize=6.5)

    axb.fill_between(g, W.min(0), W.max(0), color=GREY, lw=0)
    axb.plot(g, W.mean(0), color=INK, lw=1.2)
    axb.axhline(1.0, color=NULLC, lw=0.6, ls="-")
    k = int(W.mean(0).argmax())
    axb.annotate("peak at the boundary,\nnot in the corner", xy=(g[k], W.mean(0)[k]),
                 xytext=(-3, 3.15), fontsize=5.5, color=INK, ha="left",
                 arrowprops={"arrowstyle": "->", "lw": 0.5, "color": INK,
                             "shrinkA": 2, "shrinkB": 3})
    axb.text(hb + 1.0, 2.9, "$\\sigma_{\\mathrm{hb}}$", fontsize=6.5, color=ACC)
    axb.set_ylabel("$w \\propto |\\Delta R|$ (mean $1$)")
    axb.set_xlabel("screening-charge density $\\sigma$ ($10^{-3}\\,e\\,\\mathrm{\\AA}^{-2}$)")
    axb.set_title("(b) the transport weight it implies", loc="left", fontsize=6.5)
    save(fig, "fig_cost_map")
    plt.close(fig)


def fig_attribution(plt) -> None:
    c = json.loads(COSTS.read_text(encoding="utf8"))["per_seed"]
    a = json.loads(ATTRIB.read_text(encoding="utf8"))["per_seed"]
    fig, (axa, axb) = plt.subplots(
        # 6.5 in rather than the full 7.0: the shared axis label sits OUTSIDE the axes, and
        # with bbox_inches="tight" a 7.0 canvas prints at 515 pt against the 504 pt limit.
        1, 2, figsize=(6.5, 2.15),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.60})

    def strip(ax, rows, nulls, title, nulltext):
        for i, (_, vals) in enumerate(rows):
            ax.scatter(100 * np.asarray(vals), [i] * len(vals), s=13, facecolors="none",
                       edgecolors=INK, linewidths=0.7, zorder=3)
            ax.scatter([100 * np.mean(vals)], [i], s=26, marker="|", color=ACC, zorder=4)
        n = 100 * np.concatenate(nulls)
        lo, hi = np.percentile(n, [5, 95])
        ax.axvspan(lo, hi, color=GREY, zorder=0)
        ax.axvline(100 * n.mean(), color=NULLC, lw=0.8, ls="--", zorder=1)
        ax.axvline(0.0, color=INK, lw=0.6, zorder=2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([lab for lab, _ in rows], fontsize=5.8)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, loc="left", fontsize=6.5)
        ax.text(0.98, 0.04, nulltext, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.3, color="#6a6a6a")

    strip(axa,
          [("reference shape in the\nbins where it is nonzero",
            [c[s]["recovered_fix_nonzero"] for s in SEEDS]),
           ("exact zeros where the\nreference is zero",
            [c[s]["recovered_zero_pedestal"] for s in SEEDS])],
          [np.asarray(c[s]["null_recovered"]) for s in SEEDS],
          "(a) the error is in the occupied bins",
          "null: zeroing the same number\nof randomly chosen cells")
    strip(axb,
          [("expensive third\nof the grid", [a[s]["recovered_expensive"] for s in SEEDS]),
           ("middle third", [a[s]["recovered_middle"] for s in SEEDS]),
           ("cheap third", [a[s]["recovered_cheap"] for s in SEEDS])],
          [np.asarray(a[s]["null_recovered"]) for s in SEEDS],
          "(b) cost does not localise the error",
          "null: reference shape in a\nrandomly chosen third of bins")
    for ax in (axa, axb):
        ax.set_xlim(-235, 175)
    axb.yaxis.tick_right()
    axb.tick_params(axis="y", length=0)
    fig.supxlabel("recovered share of the closure's error gap (%)", fontsize=7, y=-0.04)
    save(fig, "fig_attribution")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["pedestal", "cost", "attribution"], default=None)
    a = ap.parse_args()
    face = apply()
    print(f"typeface: {face}")
    import matplotlib.pyplot as plt
    if a.only in (None, "pedestal"):
        fig_pedestal(plt)
    if a.only in (None, "cost"):
        fig_cost_map(plt)
    if a.only in (None, "attribution"):
        fig_attribution(plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
