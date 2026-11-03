#!/usr/bin/env python3
"""fig_pipeline: the forward path, and the two pulls that act on the intermediate.

The figure shows where the sigma-profile comes from, where it goes, and the two pulls acting
on it: supervision against the reference tabulation, and the gradient of the solubility loss
arriving through the closure. That combination is what separates this pipeline from its
nearest predecessor, where the profile predictor is trained externally and then frozen.

The caption in the manuscript describes only what is drawn; the meaning is carried by the text.

LAYOUT. Everything is positioned in one coordinate system whose units are listed below, and
each annotation is placed in a named lane rather than beside the object it labels. The earlier
version placed labels next to their arrows, and three of them collided: the backward gradient
ran into the crystal-term arrow, the substitution label sat on the edge of the closure box, and
the supervision label overlapped the profile head. Lanes make the clearances explicit, and
`_assert_no_overlap` checks the boxes against each other at draw time.

    KMP_DUPLICATE_LIB_OK=TRUE python scripts/analysis/make_pipeline_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from acs_figure_style import DOUBLE_COL_PT, MIN_LABEL_PT, apply  # noqa: E402

FIGS = ROOT / "paper/figs"
INK, ENC, HEAD, CLOS, REF = "#1a1a1a", "#cfe3dc", "#cfd9ea", "#f2d6c6", "#efe3f2"
BLUE, RED = "#2166ac", "#b2182b"

#: The canvas. 50 units over 2.95 inches is about 17 units to the inch, so a 6 pt label is
#: 1.7 units tall and a one-unit gap is just under a point.
XLIM, YLIM, HEIGHT_IN = 108.0, 50.0, 2.95

#: Boxes as (x, y, width, height). The rounded box style pads each by PAD in every direction,
#: so an edge sits PAD outside the rectangle given here; the overlap check accounts for it.
PAD = 0.4
BOXES = {
    "solute":    (1.0, 34.0, 13.0, 8.0),
    "solvent":   (1.0, 19.0, 13.0, 8.0),
    "encoder":   (19.0, 24.5, 14.0, 15.0),
    "crystal":   (38.0, 37.0, 14.0, 8.0),
    "sigma":     (38.0, 19.0, 14.0, 8.0),
    "cosmo":     (57.0, 19.0, 16.0, 8.0),
    "solve":     (80.0, 26.0, 13.0, 9.0),
    "lnx2":      (97.0, 27.5, 8.0, 6.0),
    "reference": (57.0, 2.0, 16.0, 5.5),
}


def _assert_no_overlap() -> None:
    """Two boxes that touch are a layout error, not a style choice."""
    items = list(BOXES.items())
    for i, (na, a) in enumerate(items):
        ax0, ay0, ax1, ay1 = a[0] - PAD, a[1] - PAD, a[0] + a[2] + PAD, a[1] + a[3] + PAD
        for nb, b in items[i + 1:]:
            bx0, by0, bx1, by1 = b[0] - PAD, b[1] - PAD, b[0] + b[2] + PAD, b[1] + b[3] + PAD
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                raise SystemExit(f"boxes {na} and {nb} overlap")
        if ax1 > XLIM or ay1 > YLIM or ax0 < 0 or ay0 < 0:
            raise SystemExit(f"box {na} runs outside the canvas")


def main() -> int:
    apply()
    _assert_no_overlap()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_PT / 72.0, HEIGHT_IN))
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.axis("off")

    def box(name, text, fc, fs=6.2):
        x, y, w, h = BOXES[name]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}", linewidth=0.7,
                                    edgecolor=INK, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3)

    def arrow(p0, p1, ls="-", color=INK, lw=0.8, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                     linewidth=lw, linestyle=ls, color=color, zorder=1,
                                     connectionstyle=f"arc3,rad={rad}"))

    def label(xy, text, fs=5.6, color=INK, ha="center", va="center"):
        assert fs >= MIN_LABEL_PT, f"label below the specification floor: {fs} pt"
        ax.text(xy[0], xy[1], text, fontsize=fs, color=color, ha=ha, va=va, zorder=4)

    box("solute", "solute\ngraph", "white")
    box("solvent", "solvent\ngraph", "white")
    box("encoder", "shared\nencoder", ENC)
    box("crystal", "crystal\nhead", HEAD)
    box("sigma", "$\\sigma$-profile\nhead", HEAD)
    box("cosmo", "COSMO-SAC", CLOS)
    box("solve", "solve for\n$\\ln x_2$", REF)
    box("lnx2", "$\\ln x_2$", "white")
    box("reference", "reference table", "white", fs=5.8)

    # Forward path.
    arrow((14.0, 38.0), (18.6, 35.5))
    arrow((14.0, 23.0), (18.6, 28.5))
    arrow((33.4, 35.5), (37.6, 41.0), rad=0.10)
    arrow((33.4, 27.5), (37.6, 23.0), rad=-0.10)
    arrow((52.4, 41.0), (80.0, 34.5), rad=-0.12)
    arrow((52.4, 23.0), (56.6, 23.0))
    arrow((73.4, 23.0), (80.0, 27.5), rad=0.12)
    arrow((93.4, 30.5), (96.6, 30.5))

    # Lane above the crystal arrow.
    label((66.0, 43.6), "$\\Phi(T)$: crystal term")
    # Beside the short arrows, each clear of every other lane.
    label((54.5, 20.9), "$\\hat{z}$", fs=6.2)
    label((75.2, 19.8), "$\\ln\\gamma_2$", ha="left")

    # The gradient returns from the closure to the head that produced the profile. Its lane is
    # the corridor between the top of the closure box and the crystal arrow above it.
    arrow((79.6, 27.6), (47.0, 27.6), ls=":", lw=0.9, rad=0.20)
    label((66.0, 33.8), "gradient of the solubility loss")

    # The reference tabulation acts twice: it supervises the head during training, and it can
    # replace the profile at prediction time. The two leave the same box on opposite sides.
    arrow((57.0, 5.0), (45.0, 18.6), ls="--", color=BLUE, lw=0.9, rad=0.22)
    label((35.0, 10.5), "supervised against\nthe reference", color=BLUE, fs=5.6)
    arrow((69.0, 7.9), (69.0, 18.6), ls="--", color=RED, lw=0.9)
    label((70.9, 13.2), "substituted at\nprediction time", color=RED, fs=5.6, ha="left")

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_pipeline.{ext}", dpi=600, bbox_inches="tight")
    print("written: paper/figs/fig_pipeline.pdf and .png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
