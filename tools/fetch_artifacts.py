#!/usr/bin/env python3
"""Download and unpack the archived checkpoints and data split.

The five trained checkpoints and the data split they were trained on are too large for
version control and are archived with this package instead. This script fetches them from
that archive, checks each download against the checksum the archive publishes, and unpacks
them into the repository root, where the configurations and scripts expect them.

It uses only the standard library, so it runs before the package is installed. Re-running it
is harmless: an archive whose checksum already matches is not downloaded again.

Usage:
    python tools/fetch_artifacts.py                 # checkpoints and data
    python tools/fetch_artifacts.py --only data     # just the split, 7.6 MB
    python tools/fetch_artifacts.py --keep-archives # leave the .tar.gz files in place
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The archived record for this version of the package. The concept identifier
#: 10.5281/zenodo.23907893 always resolves to the most recent version instead.
RECORD = "23907894"
API = f"https://zenodo.org/api/records/{RECORD}"

#: Archive name, what it unpacks to, and roughly how large it is.
ARCHIVES = {
    "checkpoints": ("checkpoints.tar.gz", "checkpoints/e5_leakfree", "190 MB"),
    "data": ("data.tar.gz", "notebooks/data/processed", "7.6 MB"),
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def record_files() -> dict[str, dict]:
    try:
        with urllib.request.urlopen(API, timeout=120) as r:
            meta = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(
                f"the archived record {RECORD} is not publicly available yet.\n"
                "If you are reading this before the record was released, the archives are "
                "not downloadable and the analysis that needs the checkpoints cannot be run; "
                "everything that reads the deposited results under results/ still works."
            )
        raise SystemExit(f"cannot reach the archive: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach the archive: {exc.reason}")
    out = {}
    for f in meta.get("files", []):
        name = f.get("key") or f.get("filename")
        out[name] = f
    return out


def fetch(name: str, info: dict, dest: Path) -> Path:
    want = (info.get("checksum") or "").replace("md5:", "")
    if dest.is_file() and want and md5(dest) == want:
        print(f"  {name}: already downloaded and verified")
        return dest
    url = info["links"]["self"] if "links" in info else info["links"]["download"]
    print(f"  {name}: downloading {info.get('size', 0):,} bytes")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    got = md5(tmp)
    if want and got != want:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"  {name}: checksum mismatch, expected {want}, got {got}")
    tmp.replace(dest)
    print(f"  {name}: checksum verified")
    return dest


def unpack(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (ROOT / member.name).resolve()
            if not str(target).startswith(str(ROOT.resolve())):
                raise SystemExit(f"  refusing to unpack outside the repository: {member.name}")
        tf.extractall(ROOT)
    print(f"  {archive.name}: unpacked")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(ARCHIVES), action="append",
                    help="fetch one archive instead of both; may be given twice")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the downloaded .tar.gz files after unpacking")
    a = ap.parse_args()
    wanted = a.only or sorted(ARCHIVES)

    print(f"archive record {RECORD} (https://doi.org/10.5281/zenodo.{RECORD})")
    for key in wanted:
        name, marker, size = ARCHIVES[key]
        if (ROOT / marker).exists():
            print(f"  {name}: {marker} is already here, skipping ({size})")
            wanted = [w for w in wanted if w != key]
    if not wanted:
        print("nothing to do")
        return 0

    files = record_files()
    for key in wanted:
        name, marker, _ = ARCHIVES[key]
        if name not in files:
            raise SystemExit(f"the archive does not contain {name}; "
                             f"it holds {sorted(files)}")
        archive = fetch(name, files[name], ROOT / name)
        unpack(archive)
        if not a.keep_archives:
            archive.unlink()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
