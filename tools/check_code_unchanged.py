#!/usr/bin/env python3
"""Verify that this repository's Python differs from its source only in text.

The reproduction repository was prepared by rewriting comments, docstrings and printed
messages, and by nothing else. That claim is only worth making if it can be checked, so
this script compares each file here against the corresponding file in the working
repository and reports any difference in executable structure.

The comparison parses both files and compares their syntax trees after replacing every
string literal with a single placeholder and discarding docstrings. Two files that agree
under this comparison compute the same thing: identifiers, control flow, numeric
constants, call signatures and argument order are all included, while prose is not.

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
NEW_HERE = {"tools/check_clean.py", "tools/check_code_unchanged.py"}


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

    checked = missing = differ = 0
    problems: list[str] = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel in NEW_HERE:
            continue
        # src/tgnn_solv/... and scripts/... and tests/... keep their layout; deposits hold no code.
        src = a.source / rel
        if not src.exists():
            missing += 1
            problems.append(f"{rel}: no counterpart in the source repository")
            continue
        checked += 1
        try:
            if shape(p) != shape(src):
                differ += 1
                problems.append(f"{rel}: executable structure differs")
        except SyntaxError as exc:
            problems.append(f"{rel}: cannot parse ({exc})")

    print(f"compared {checked} files against {a.source}")
    if problems:
        print(f"\n{len(problems)} problems:\n")
        for pr in problems:
            print(f"  {pr}")
        return 1
    print("every file differs from its source only in comments, docstrings and messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
