#!/usr/bin/env python3
"""Verify that this repository holds everything it reads, and nothing it does not read.

A reproduction package has two failure modes that no amount of prose prevents. It can be
incomplete, so that a script reads a file nobody shipped, and it can be padded, so that a
reader cannot tell which of a hundred files any result came from. This script checks both.

HOW. Every Python file here is parsed and every string constant that looks like a
repository-relative artefact path is collected. Each path must fall into exactly one of
four declared classes:

    present   it is in this repository, and the script checks that it is
    external  it arrives from the archived data or checkpoints, named per entry
    written   a script here creates it at run time, and that script is named per entry
    unused    a configuration default for a feature this manuscript does not exercise

An undeclared path fails. A declared path that no longer appears fails as stale, so the
tables below cannot drift away from the code. A `written` entry whose producer is not
shipped fails. Finally every file under `results/` must be read by some script here,
which is the check that the deposit carries nothing the reader has no use for.

Usage:
    python tools/check_inputs_present.py
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: A string constant is treated as an artefact path when it starts with one of these.
PREFIXES = ("results/", "notebooks/", "checkpoints/", "configs/", "figures/", "paper/")

#: Paths supplied by the archived artefacts rather than by this repository, with the
#: archive each one comes from. Unpack both into the repository root.
EXTERNAL = {
    "checkpoints/e5_leakfree": "checkpoints.tar.gz",
    "checkpoints/e5_leakfree/grounded_a_seed42.pt": "checkpoints.tar.gz",
    "notebooks/data/processed": "data.tar.gz",
    "notebooks/data/processed/test.csv": "data.tar.gz",
    "notebooks/data/processed/train.csv": "data.tar.gz",
    "notebooks/data/processed/val.csv": "data.tar.gz",
    "notebooks/data/processed_sigma_aux_stream_rebuilt/sigma_train.csv": "data.tar.gz",
    "notebooks/data/processed_sigma_aux_stream/sigma_train.csv":
        "data.tar.gz; rebuilt by scripts/data/build_sigma_profile_aux_stream.py",
    "notebooks/data/processed_sigma_aux_stream/sigma_val.csv":
        "data.tar.gz; rebuilt by scripts/data/build_sigma_profile_aux_stream.py",
    "notebooks/data/processed_sigma_aux_stream/summary.json":
        "data.tar.gz; rebuilt by scripts/data/build_sigma_profile_aux_stream.py",
}

#: Paths created at run time, each with the file here that creates it.
WRITTEN = {
    "checkpoints/model.pt": "scripts/train.py",
    "paper/si_tables/broad_idac_set_477.csv": "scripts/analysis/run_published_idac_closure_check.py",
    "paper/si_tables/vt2005_matched_set_60.csv": "scripts/analysis/run_published_idac_closure_check.py",
    "results/ideal_sle_baseline.json": "src/tgnn_solv/baselines/ideal_sle.py",
    "results/rf_baseline.json": "src/tgnn_solv/baselines/rf_baseline.py",
    "results/sigma_profile_artifact/summary.json":
        "scripts/data/ingest_vt2005_sigma_profiles.py",
    "results/toc_graphic": "scripts/analysis/make_toc_figure_v2.py",
}

#: Configuration defaults for features this manuscript does not exercise. They are read
#: only when the corresponding option is switched on, which none of the runs of record do.
UNUSED = {
    "results/source_uncertainty_audit_reviewed/supervised_rows_with_source_uncertainty.csv":
        "per-row label uncertainty; off in every configuration distributed here",
}


#: This file necessarily contains the prefixes it scans for, so it does not scan itself.
SELF = "tools/check_inputs_present.py"


def referenced() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in p.parts or p.relative_to(ROOT).as_posix() == SELF:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf8"))
        except SyntaxError as exc:
            raise SystemExit(f"{p.relative_to(ROOT)}: cannot parse ({exc})")
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                s = n.value.strip()
                if s.startswith(PREFIXES) and len(s) < 160 and "\n" not in s:
                    out.setdefault(s, set()).add(p.relative_to(ROOT).as_posix())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="print the full classification")
    a = ap.parse_args()

    refs = referenced()
    problems: list[str] = []
    counts = {"present": 0, "external": 0, "written": 0, "unused": 0}
    rows: list[tuple[str, str, str]] = []

    for path in sorted(refs):
        # A directory a script created as a side effect and left empty is not the artefact,
        # so an empty directory does not count as present.
        here = ROOT / path
        in_pkg = here.is_file() or (here.is_dir() and any(here.iterdir()))
        declared = [c for c, tbl in (("external", EXTERNAL), ("written", WRITTEN),
                                     ("unused", UNUSED)) if path in tbl]
        if len(declared) > 1:
            problems.append(f"{path}: declared in more than one table ({', '.join(declared)})")
            continue
        if in_pkg and not declared:
            counts["present"] += 1
            rows.append(("present", path, "in this repository"))
            continue
        if not declared:
            problems.append(f"{path}: absent and undeclared, read by "
                            f"{sorted(refs[path])[0]}")
            continue
        cls = declared[0]
        counts[cls] += 1
        note = {"external": EXTERNAL, "written": WRITTEN, "unused": UNUSED}[cls][path]
        rows.append((cls, path, note))
        if cls == "written" and not (ROOT / note).exists():
            problems.append(f"{path}: declared as written by {note}, which is not here")
        if cls == "external" and in_pkg:
            problems.append(f"{path}: declared external but present; move it out of EXTERNAL")

    for tbl_name, tbl in (("EXTERNAL", EXTERNAL), ("WRITTEN", WRITTEN), ("UNUSED", UNUSED)):
        for stale in sorted(set(tbl) - set(refs)):
            problems.append(f"{stale}: declared in {tbl_name} but nothing reads it; "
                            "remove the stale entry")

    # Minimality: every deposited file must be read by something here.
    deposited = sorted(p.relative_to(ROOT).as_posix()
                       for p in (ROOT / "results").rglob("*") if p.is_file())
    unread = [d for d in deposited if d not in refs]
    for d in unread:
        problems.append(f"{d}: deposited but no script here reads it")

    print(f"classified {len(refs)} referenced paths: "
          + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"deposited files under results/: {len(deposited)}, all read: {not unread}")
    if a.verbose:
        print()
        for cls, path, note in rows:
            print(f"  {cls:<9} {path:<72} {note}")
    if problems:
        print(f"\n{len(problems)} problems:\n")
        for pr in problems:
            print(f"  {pr}")
        return 1
    print("\nevery path a script reads is here or declared, and every deposited file is read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
