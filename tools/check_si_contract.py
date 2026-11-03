#!/usr/bin/env python3
"""Check that the Supporting Information holds what the article says it holds, and no more.

The article's "Supporting Information Available" section is a contract: it names, in order, what
a reader will find in the other document. A Supporting Information that grows past that list stops
being navigable, and one that falls short of it makes a promise the reader cannot collect. Both
failures are silent in LaTeX, which is why they are checked here.

HOW. The contract is the semicolon-separated list in the article's Supporting Information section.
The supply is the sequence of `\\section{...}` headings in the Supporting Information body. CONTRACT
below declares the correspondence, one promise to one section label, and the script requires that
the article still makes exactly those promises in that order, that the Supporting Information still
has exactly those sections in that order, and that each section carries its declared label. A
section with no promise, or a promise with no section, fails.

Usage:
    python tools/check_si_contract.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "paper/learned_sigma_profile.tex"
SI_BODY = ROOT / "paper/sections/si-v2.tex"

#: One promise from the article's list, and the Supporting Information section that keeps it. The
#: promise text is matched as a substring of the article's list item, so a reworded promise that
#: still contains the phrase passes and a promise that drops it fails.
CONTRACT = [
    ("Architecture and training details", "si:arch"),
    ("construction of the splits and the auxiliary supervision", "si:data"),
    ("the control for each measurement", "si:rules"),
    ("what the observations leave undetermined", "si:unresolved"),
    ("second system based on a linear free-energy relation", "si:second"),
]


def article_promises() -> list[str]:
    text = ARTICLE.read_text(encoding="utf8")
    m = re.search(r"\\section\*\{Supporting Information Available\}(.*?)(?=\\section\*|\\bibliography)",
                  text, re.S)
    if not m:
        raise SystemExit("the article has no Supporting Information Available section")
    body = " ".join(m.group(1).split())
    return [p.strip(" .") for p in body.split(";") if p.strip(" .")]


def si_sections() -> list[tuple[str, str]]:
    text = SI_BODY.read_text(encoding="utf8")
    out = []
    for m in re.finditer(r"\\section\{([^}]*)\}\s*\n\\label\{([^}]*)\}", text):
        out.append((m.group(1).strip(), m.group(2).strip()))
    bare = len(re.findall(r"\\section\{", text))
    if bare != len(out):
        raise SystemExit(f"{bare - len(out)} Supporting Information section(s) carry no label on "
                         "the line after the heading; the contract cannot be checked")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="print the matched pairs")
    a = ap.parse_args()

    if not (ARTICLE.exists() and SI_BODY.exists()):
        print("the manuscript sources are not in this repository yet, so the contract cannot "
              "be checked; see paper/README.md")
        return 0

    promises, sections = article_promises(), si_sections()
    problems: list[str] = []

    if len(promises) != len(CONTRACT):
        problems.append(f"the article makes {len(promises)} promises and CONTRACT declares "
                        f"{len(CONTRACT)}; update the table or the article")
    if len(sections) != len(CONTRACT):
        problems.append(f"the Supporting Information has {len(sections)} sections and CONTRACT "
                        f"declares {len(CONTRACT)}")

    for i, (phrase, label) in enumerate(CONTRACT):
        if i < len(promises) and phrase.lower() not in promises[i].lower():
            problems.append(f"promise {i+1} is {promises[i]!r}, which does not contain the "
                            f"declared phrase {phrase!r}")
        if i < len(sections) and sections[i][1] != label:
            problems.append(f"section {i+1} is labelled {sections[i][1]!r}, not {label!r}; "
                            "the order of the sections must follow the order of the promises")
        if a.verbose and i < len(sections) and i < len(promises):
            print(f"  {label:<16} {sections[i][0][:46]:<48} <- {promises[i][:52]}")

    print(f"{len(promises)} promises in the article, {len(sections)} sections in the "
          "Supporting Information")
    if problems:
        print(f"\n{len(problems)} problems:\n")
        for p in problems:
            print(f"  {p}")
        return 1
    print("every promise has its section, every section has its promise, and the order agrees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
