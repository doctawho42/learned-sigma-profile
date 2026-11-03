#!/usr/bin/env python3
"""Verify that this repository's Python differs from its source only where declared.

The reproduction repository was prepared by rewriting comments, docstrings and printed
messages, and by removing material the manuscript does not use. Both claims are only
worth making if they can be checked, so this script compares each file here against the
corresponding file in the working repository and reports any difference in executable
structure that is not declared below.

WHAT THE COMPARISON COVERS. Both files are parsed and their syntax trees compared after
every string literal is replaced with one placeholder and docstrings are discarded. Two
files that agree under this comparison have the same identifiers, control flow, numeric
constants, call signatures and argument order.

WHAT IT DOES NOT COVER, and this matters. String literals are out of scope by
construction, and a dictionary key is a string literal. Renaming a key therefore passes
this check silently. Three keys were renamed when this repository was prepared, from
Russian to English, in `results/per_bin_sensitivity_weights/closure_error_attribution.json`
and in the three files that read it; the numeric content of that deposit was compared
before and after the rename, 266 numeric leaves, and the test suite reads the renamed
keys. The protection for a key rename is that suite, not this script.

Usage:
    python tools/check_code_unchanged.py --source /path/to/working/repository
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Files that exist only here and have no counterpart to compare against.
NEW_HERE = {"tools/check_clean.py", "tools/check_code_unchanged.py",
            "tools/check_inputs_present.py", "tools/fetch_artifacts.py",
            "tools/check_si_contract.py",
            "tests/test_package_is_self_contained.py"}

#: Files whose executable structure is deliberately different here, each with the reason.
#: An entry that no longer differs is reported as stale, so this list cannot rot.
CHANGED_HERE = {
    "tests/test_sigma_bin_weights_hook.py":
        "the retired gate-1 test and its path constant were removed; the test read "
        "results/per_bin_sensitivity_weights/weight_points_at_error.json, a deposit "
        "measured over profile error rather than closure error, which answers a "
        "different question from the one the manuscript asks and is not distributed here",
}


class Normalise(ast.NodeTransformer):
    """Replace string constants with a placeholder and drop docstrings."""

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<text>"), node)
        return node

    def _strip_doc(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip_doc
    visit_FunctionDef = _strip_doc
    visit_AsyncFunctionDef = _strip_doc
    visit_ClassDef = _strip_doc


def shape(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf8"))
    return ast.dump(ast.fix_missing_locations(Normalise().visit(tree)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="root of the working repository to compare against")
    a = ap.parse_args()
    if not a.source.is_dir():
        raise SystemExit(f"not a directory: {a.source}")

    checked = 0
    declared_and_differs: set[str] = set()
    problems: list[str] = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel in NEW_HERE:
            continue
        src = a.source / rel
        if not src.exists():
            problems.append(f"{rel}: no counterpart in the source repository")
            continue
        checked += 1
        try:
            if shape(p) != shape(src):
                if rel in CHANGED_HERE:
                    declared_and_differs.add(rel)
                else:
                    problems.append(f"{rel}: executable structure differs, and no reason "
                                    "is declared in CHANGED_HERE")
        except SyntaxError as exc:
            problems.append(f"{rel}: cannot parse ({exc})")

    for rel in sorted(set(CHANGED_HERE) - declared_and_differs):
        problems.append(f"{rel}: declared in CHANGED_HERE but no longer differs; "
                        "remove the stale entry")

    print(f"compared {checked} files against {a.source}")
    if declared_and_differs:
        print(f"\n{len(declared_and_differs)} declared difference(s):")
        for rel in sorted(declared_and_differs):
            print(f"  {rel}\n      {CHANGED_HERE[rel]}")
    if problems:
        print(f"\n{len(problems)} problems:\n")
        for pr in problems:
            print(f"  {pr}")
        return 1
    print("\nevery other file differs from its source only in comments, docstrings and messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
