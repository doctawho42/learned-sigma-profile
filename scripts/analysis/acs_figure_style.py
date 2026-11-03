#!/usr/bin/env python
"""The journal's graphics specification, in one place, applied by every figure script.

ACS is specific about submitted artwork and two of its rules were being broken silently by every
figure in this tree:

  * **Type 3 fonts.** Matplotlib's PDF backend emits Type 3 by default -- procedural glyphs rather
    than an embedded outline font. Production systems commonly reject them, and the failure arrives
    as a request weeks after submission rather than as an error here. ``pdf.fonttype = 42`` embeds
    TrueType instead.
  * **The typeface.** The spec asks for Helvetica or Arial; matplotlib was using DejaVu Sans, which
    is what it falls back to when nothing else is set. Both are installed on the machine that built
    these figures, and :func:`assert_font_available` refuses to let a silent fallback ship.

The other three rules are dimensional and are checked on the produced files rather than promised
here, by ``scripts/analysis/check_figure_spec.py``: a single-column figure at most 240 pt wide, a
double-column one between 300 and 504 pt, no rule thinner than 0.5 pt, and no label under 4.5 pt.

    from acs_figure_style import apply, SINGLE_COL_PT, DOUBLE_COL_PT
    apply()
    fig = plt.figure(figsize=(DOUBLE_COL_PT / 72.0, 2.1))
"""
from __future__ import annotations

#: The two widths the specification allows, in points. A figure between them is out of spec, which
#: is why these are named rather than left to whatever \linewidth happened to be.
SINGLE_COL_PT = 240.0
DOUBLE_COL_MIN_PT, DOUBLE_COL_PT = 300.0, 504.0
#: Floors the specification sets. Enforced on the output, not here; repeated here so a script that
#: sets a linewidth can compare against the number rather than against a memory of it.
MIN_RULE_PT, MIN_LABEL_PT = 0.5, 4.5

# ARIAL FIRST, AND FOR A MEASURED REASON. The specification allows either. On macOS Helvetica ships
# as a .ttc collection, and matplotlib's mathtext could not resolve an italic out of it: asked for
# "Helvetica:italic" it fell through to Apple Chancery, a cursive face, which shipped inside
# fig_pka_hammett before this was caught. Arial resolves to separate Arial.ttf, Arial Bold.ttf and
# Arial Italic.ttf files and every style comes back correct. Check with font_manager.findfont
# before reordering this list.
_PREFERRED = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans", "Liberation Sans"]


def assert_font_available(family: str | None = None) -> str:
    """Return the first preferred face that is installed, or raise.

    A MISSING FONT IS NOT AN ERROR IN MATPLOTLIB -- it warns once and substitutes DejaVu, and the
    figure ships looking fine and violating the spec. This project has lost a session to exactly
    that shape of failure on a GPU that was present but unusable, so the check is an assertion.
    """
    import matplotlib.font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ([family] if family else _PREFERRED):
        if name in installed:
            return name
    raise SystemExit(
        "none of the specification's typefaces is installed: "
        f"{', '.join(_PREFERRED)}. Install one, or the figures will ship in DejaVu Sans, which "
        "matplotlib substitutes without failing.")


def apply(family: str | None = None) -> str:
    """Set the rcParams the specification fixes. Returns the face actually selected."""
    import matplotlib

    face = assert_font_available(family)
    matplotlib.rcParams.update({
        # 42 is TrueType. 3 is the default and is the one that gets rejected.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [face] + [f for f in _PREFERRED if f != face],
        # MATHTEXT HAS ITS OWN FONT SET and ignores font.sans-serif entirely. Left at its default
        # it embeds DejaVu for every $...$ in the figure, which is most of the labels here, so the
        # file ships with two typefaces and one of them is the wrong one. "custom" is the only
        # fontset that takes an arbitrary installed face.
        "mathtext.fontset": "custom",
        "mathtext.rm": face,
        "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",
        "mathtext.sf": face,
        # \mathcal AND \mathtt HAVE NO ARIAL EQUIVALENT, and leaving them unset is not neutral:
        # with fontset "custom" matplotlib resolves \mathcal through the system, and on this
        # machine that returned Apple Chancery, a cursive display face, which shipped inside
        # fig_pka_hammett before the gate caught it. STIX carries a real calligraphic alphabet and
        # is already the fallback for the Greek that Arial lacks, so both go to one place.
        "mathtext.cal": "STIXGeneral",
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.default": "regular",
        # Matplotlib's own defaults sit below the specification's floors on several of these.
        "axes.linewidth": max(0.6, MIN_RULE_PT),
        "xtick.major.width": max(0.6, MIN_RULE_PT),
        "ytick.major.width": max(0.6, MIN_RULE_PT),
        "xtick.minor.width": max(0.5, MIN_RULE_PT),
        "ytick.minor.width": max(0.5, MIN_RULE_PT),
        "grid.linewidth": max(0.5, MIN_RULE_PT),
        "patch.linewidth": max(0.5, MIN_RULE_PT),
        "hatch.linewidth": max(0.5, MIN_RULE_PT),
    })
    return face


if __name__ == "__main__":
    print(f"selected face: {apply()}")
    print(f"single column: <= {SINGLE_COL_PT:.0f} pt")
    print(f"double column: {DOUBLE_COL_MIN_PT:.0f}-{DOUBLE_COL_PT:.0f} pt")
    print(f"floors: rules >= {MIN_RULE_PT} pt, labels >= {MIN_LABEL_PT} pt")
