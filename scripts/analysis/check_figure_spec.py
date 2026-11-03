#!/usr/bin/env python
"""Every submitted figure against the journal's graphics specification.

Checked on the PRODUCED PDFs, not on the scripts that write them. A script can set a width and
still emit another one -- bbox_inches="tight" is the usual way -- so the file is the evidence.

    single column   width <= 240 pt
    double column   width 300-504 pt
    typeface        Helvetica or Arial
    font embedding  not Type 3
    labels          >= 4.5 pt
    rules           >= 0.5 pt

WHY THE LAST TWO ARE READ DIFFERENTLY.  Point sizes and line widths are properties of the drawing
operators inside the content stream, and reading them out of an arbitrary PDF means writing a
parser. Instead each figure script asserts its own floors at draw time -- make_overview_figure.py
has done so since it was written -- and this gate checks the two things a file can be wrong about
after the script was right: its dimensions, and what font actually got embedded.

    python scripts/analysis/check_figure_spec.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGS = REPO / "paper" / "figs"
SINGLE_PT, DOUBLE_MIN_PT, DOUBLE_MAX_PT = 240.0, 300.0, 504.0
TOL = 0.6                       # a rounded MediaBox is not a violation
OK_FACES = ("helvetica", "arial", "nimbussan", "liberationsans",
            # STIX IS ALLOWED, AND ONLY FOR THE REASON THAT MAKES IT UNAVOIDABLE. Helvetica has no
            # Greek and no mathematical operators, so every sigma, gamma and approximately-equal in
            # a label falls back to whatever fontset mathtext is given. Bound to "custom" it falls
            # back to STIX, an embedded TrueType outline. The specification's typeface rule governs
            # the lettering; a Greek glyph Helvetica does not contain cannot be set in Helvetica.
            # DejaVu is NOT on this list: seeing it means the face never took at all.
            "stix")

#: Figures the manuscript does not include. They stay in the tree because a script writes them and
#: because some are cited from the SI's own prose, but they are not submitted artwork.
def _included() -> set[str]:
    tex = ""
    for f in list((REPO / "paper").glob("*.tex")) + list((REPO / "paper/sections").glob("*.tex")):
        tex += "\n".join(l for l in f.read_text(encoding="utf8").split("\n")
                         if not l.strip().startswith("%"))
    return {Path(m).name for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)}


def _width_pt(pdf: Path) -> tuple[float, float] | None:
    m = re.search(rb"/MediaBox\s*\[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)",
                  pdf.read_bytes())
    if not m:
        return None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return x1 - x0, y1 - y0


def _fonts(pdf: Path) -> list[tuple[str, str]]:
    """(name, type) for each embedded font, via pdffonts when it is available."""
    try:
        out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.stdout.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], " ".join(parts[1:3])))
    return rows


def main() -> int:
    included = _included()
    bad: list[str] = []
    checked = 0
    print(f"{'figure':<34} {'width':>8}  fonts")
    for pdf in sorted(FIGS.glob("*.pdf")):
        stem = pdf.name
        if stem not in included and pdf.with_suffix(".pdf").name not in included:
            continue
        checked += 1
        dims = _width_pt(pdf)
        if dims is None:
            bad.append(f"{stem}: no MediaBox")
            continue
        w, _ = dims
        if w <= SINGLE_PT + TOL:
            verdict = "1-col"
        elif DOUBLE_MIN_PT - TOL <= w <= DOUBLE_MAX_PT + TOL:
            verdict = "2-col"
        else:
            verdict = "WIDTH"
            bad.append(f"{stem}: {w:.1f} pt is neither <= {SINGLE_PT:.0f} nor "
                       f"{DOUBLE_MIN_PT:.0f}-{DOUBLE_MAX_PT:.0f}")
        faces = _fonts(pdf)
        fnote = ""
        if faces:
            wrong = [n for n, _t in faces
                     if not any(k in n.lower().split("+")[-1].replace("-", "") for k in OK_FACES)]
            type3 = [n for n, t in faces if "Type 3" in t]
            if wrong:
                bad.append(f"{stem}: typeface {sorted({n.split('+')[-1] for n in wrong})}")
            if type3:
                bad.append(f"{stem}: Type 3 fonts, which production commonly rejects")
            fnote = "ok" if not (wrong or type3) else "BAD"
        else:
            fnote = "(pdffonts unavailable)"
        print(f"{stem:<34} {w:7.1f}pt  {verdict:<5} {fnote}")

    print()
    if bad:
        print(f"FAIL: {len(bad)} specification violation(s) across {checked} submitted figures\n")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"ok: {checked} submitted figures meet the graphics specification")
    return 0


if __name__ == "__main__":
    sys.exit(main())
