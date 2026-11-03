#!/usr/bin/env python3
"""Schematic of the pipeline and of the two influences acting on the sigma-profile.

The figure shows the shared encoder, the crystal head feeding the ideal term, the profile
head feeding COSMO-SAC, and the equilibrium solve. Two influences act on the profile head:
supervision against the reference tabulation, and the gradient of the solubility loss
through the closure. Their combination is what distinguishes this pipeline from designs in
which the profile predictor is supervised externally and then held fixed.

    python scripts/analysis/make_pipeline_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from acs_figure_style import DOUBLE_COL_PT, apply  # noqa: E402

FIGS = ROOT / "paper/figs"
INK, ENC, HEAD, CLOS, REF = "#1a1a1a", "#cfe3dc", "#cfd9ea", "#f2d6c6", "#efe3f2"


def main() -> int:
    apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_PT / 72.0, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=6.0, lw=0.7, style="round,pad=0.4"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                    edgecolor=INK, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)

    def arrow(x0, y0, x1, y1, ls="-", color=INK, lw=0.8, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=lw, linestyle=ls,
                                     color=color, zorder=1,
                                     connectionstyle=f"arc3,rad={rad}"))

    box(1, 25, 12, 7, "solute\ngraph", "white")
    box(1, 9, 12, 7, "solvent\ngraph", "white")
    box(17, 15, 13, 12, "shared\nencoder", ENC)
    box(34, 25, 13, 8, "crystal\nhead", HEAD)
    box(34, 10, 13, 8, "$\\sigma$-profile\nhead", HEAD)
    box(52, 10, 15, 8, "COSMO-SAC", CLOS)
    box(72, 18, 12, 9, "solve for\n$\\ln x_2$", REF)
    box(89, 19.5, 10, 6, "$\\ln x_2$", "white")
    box(52, 1.0, 15, 5.5, "reference table", "white", fs=5.6)

    for y in (28.5, 12.5):
        arrow(13, y, 17, 21 if y > 20 else 21)
    arrow(30, 23, 34, 29)
    arrow(30, 19, 34, 14)
    arrow(47, 29, 72, 25.5, rad=-0.12)
    ax.text(58, 31.2, "$\\Phi(T)$: crystal term", fontsize=5.6, color=INK, ha="center")
    arrow(47, 14, 52, 14)
    ax.text(49.5, 15.6, "$\\hat{z}$", fontsize=6.2, color=INK, ha="center")
    arrow(67, 14, 72, 20, rad=0.1)
    ax.text(70.5, 15.0, "$\\ln\\gamma_2$", fontsize=5.6, color=INK, ha="left")
    arrow(84, 22.5, 89, 22.5)

    # две тяги на промежуточную величину
    arrow(59.5, 6.5, 59.5, 10.0, ls="--", color="#b2182b", lw=0.9)
    ax.text(61.2, 7.6, "substituted at prediction time", fontsize=5.4, color="#b2182b",
            ha="left", va="center")
    ax.add_patch(FancyArrowPatch((52, 3.7), (41, 9.6), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, linestyle="--", color="#2166ac",
                                 connectionstyle="arc3,rad=0.25", zorder=1))
    ax.text(40.5, 4.6, "supervised against\nthe reference", fontsize=5.4, color="#2166ac",
            ha="center", va="center")
    ax.add_patch(FancyArrowPatch((72, 18.4), (47.6, 16.4), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, linestyle=":", color=INK,
                                 connectionstyle="arc3,rad=0.30", zorder=1))
    ax.text(61.0, 23.0, "gradient of the solubility loss", fontsize=5.4, color=INK,
            ha="center", va="center")

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_pipeline.{ext}", dpi=600, bbox_inches="tight")
    print("записано: paper/figs/fig_pipeline.pdf и .png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
