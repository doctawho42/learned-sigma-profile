"""Assert that the package can be run and read without anything that is not in it.

Two failures are easy to ship and hard to notice. A script can import a helper module that
was left behind, which breaks an entry point for everyone but the author, whose working
directory still holds it. And a deposited file can go unmentioned, which leaves a reader
unable to tell what produced it. Both are checked here rather than asserted in prose.

The import check resolves module-level imports the way the interpreter does when a script
is run directly, with the script's own directory ahead of the path, so a helper that only
resolves inside the development tree is reported.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Packages imported lazily inside a function for a feature this manuscript does not use.
#: They are not dependencies and are not expected to be installed.
OPTIONAL = {
    "opencosmorspy",                    # COSMO-RS reference calculations
    "optuna",                           # hyperparameter search
    "thermo",                           # UNIFAC comparison
    "probe_gsol_descriptor_recovery",   # the descriptor probe, off in every configuration here
    "fastsolv",                         # external baseline
    "descriptastorus",                  # external baseline features
    "tensorboardX",
    "streamlit",
}


def python_files() -> list[Path]:
    return [p for p in sorted(ROOT.rglob("*.py")) if "__pycache__" not in p.parts]


def module_level_imports(path: Path) -> set[str]:
    """Top-level names imported at module level, excluding imports inside functions."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # A guarded import is optional by construction.
            continue
    return names


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_module_level_imports_resolve(path: Path) -> None:
    extra = [str(path.parent), str(ROOT / "src")]
    saved = list(sys.path)
    sys.path[:0] = extra
    try:
        for name in sorted(module_level_imports(path)):
            if name in OPTIONAL:
                continue
            assert importlib.util.find_spec(name) is not None, (
                f"{path.relative_to(ROOT)} imports {name!r} at module level and it cannot "
                "be resolved from this repository; either it is missing or it belongs in OPTIONAL"
            )
    finally:
        sys.path[:] = saved


def test_manifest_names_every_deposited_file() -> None:
    manifest = (ROOT / "MANIFEST.md").read_text(encoding="utf8")
    deposited = sorted(p.relative_to(ROOT).as_posix()
                       for p in (ROOT / "results").rglob("*") if p.is_file())
    assert deposited, "no deposited files found under results/"
    missing = [d for d in deposited if d not in manifest]
    assert not missing, (
        "MANIFEST.md does not name these deposited files, so a reader cannot tell what "
        f"produced them: {missing}"
    )


def test_manifest_names_no_file_that_is_absent() -> None:
    """A path in the manifest that is not here either is stale or marks a missing file."""
    import re
    manifest = (ROOT / "MANIFEST.md").read_text(encoding="utf8")
    cited = set(re.findall(r"`((?:results|scripts|src|configs|tools|tests)/[\w./-]+)`", manifest))
    absent = sorted(c for c in cited if not (ROOT / c).exists())
    assert not absent, f"MANIFEST.md cites paths that are not in this repository: {absent}"
