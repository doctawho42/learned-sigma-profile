#!/usr/bin/env python3
"""Check that the repository contains documentation and no development narrative.

Reviewers of the first submission noted that the deposited code carried traces of the
development process rather than documentation of the method: commentary addressed to the
authors, dated notes recording when a mistake was found, and commit messages written as a
work diary. This script defines what that means in checkable terms and fails if any of it
returns. It is intended to run in continuous integration and before any release.

The rules below describe the form of the text, not its content. A comment explaining why a
threshold has the value it has is documentation and passes. A comment recording that the
value was wrong last Tuesday is a development note and fails.

Usage:
    python tools/check_clean.py            # check the whole repository
    python tools/check_clean.py --list     # list the files that would be checked
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".tex", ".cfg", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "figures", "deposits", ".venv"}

#: Text that is not English. The repository is read by an international readership and
#: mixed-language commentary is a development artefact.
NON_ASCII_PROSE = re.compile(r"[Ѐ-ӿ]")

#: A date in a comment records when something happened to the authors. Dates belong in the
#: manuscript and in the version history, not in source commentary.
DATED_NOTE = re.compile(r"(?<!\d)(19|20)\d{2}-[01]\d-[0-3]\d(?!\d)")

#: Phrases that address the reader as a collaborator, recount the authors' own process, or
#: name the tooling used to write the code. Matched case-insensitively on word boundaries.
NARRATIVE = [
    r"\bI (wrote|added|fixed|caught|found|forgot|assumed|thought|realised|realized)\b",
    r"\b(my|our) (mistake|error|bug|oversight|fault)\b",
    r"\bit turned out\b", r"\bturns out\b", r"\bas it happens\b",
    r"\bwas wrong\b", r"\bgot this wrong\b", r"\bthis was broken\b",
    r"\b(caught|found) (this|it) (when|while|after|during)\b",
    r"\bdo not delete\b", r"\bdo not touch\b", r"\bdon't touch\b",
    r"\b(claude|chatgpt|gpt-4|copilot|llm|language model|prompt|agent(ic)?)\b",
    r"\b(todo|fixme|xxx|hack)\b[:\s]",
    r"\bfirst (version|attempt) (of this|here)\b",
    r"\bearlier (version|draft) (of this|here)\b",
    r"\bthe reviewer\b", r"\breferee\b",
]
NARRATIVE_RX = [re.compile(p, re.I) for p in NARRATIVE]

#: Paths tied to one machine. A reader cannot resolve them and they leak the authors' setup.
LOCAL_PATH = re.compile(r"(/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+|/mnt/storage)")

#: Files in which a local path is expected because the file documents where data must be
#: placed. These still may not carry a *personal* home directory.
PATH_EXEMPT = {"README.md", "MANIFEST.md", "environment.md"}


#: The file that defines the rules necessarily contains the patterns it forbids.
SELF = "tools/check_clean.py"


def files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in SUFFIXES:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or rel.as_posix() == SELF:
            continue
        out.append(p)
    return out


def check(p: Path) -> list[str]:
    rel = p.relative_to(ROOT)
    problems: list[str] = []
    try:
        text = p.read_text(encoding="utf8")
    except UnicodeDecodeError:
        return [f"{rel}: not valid UTF-8"]
    for i, line in enumerate(text.splitlines(), 1):
        if NON_ASCII_PROSE.search(line):
            problems.append(f"{rel}:{i}: non-English prose")
        if DATED_NOTE.search(line):
            problems.append(f"{rel}:{i}: dated development note")
        for rx in NARRATIVE_RX:
            m = rx.search(line)
            if m:
                problems.append(f"{rel}:{i}: development narrative ({m.group(0).strip()!r})")
                break
        if p.name not in PATH_EXEMPT and LOCAL_PATH.search(line):
            problems.append(f"{rel}:{i}: path tied to one machine")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the files that are checked")
    a = ap.parse_args()
    fs = files()
    if a.list:
        for f in fs:
            print(f.relative_to(ROOT))
        return 0
    problems: list[str] = []
    for f in fs:
        problems.extend(check(f))
    print(f"checked {len(fs)} files")
    if problems:
        by_file: dict[str, int] = {}
        for pr in problems:
            by_file[pr.split(":")[0]] = by_file.get(pr.split(":")[0], 0) + 1
        print(f"\n{len(problems)} problems in {len(by_file)} files:\n")
        for name, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {name}")
        print("\nfirst twenty:")
        for pr in problems[:20]:
            print(f"  {pr}")
        return 1
    print("clean: documentation only, no development narrative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
