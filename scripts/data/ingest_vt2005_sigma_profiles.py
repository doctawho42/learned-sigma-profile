#!/usr/bin/env python3
"""Ingest the VT-2005 (or Mullins) sigma-profile database into a flat artifact.

Produces ``results/sigma_profile_artifact/sigma_profiles.csv`` with columns
``smiles, sigma_area, sigma_p_0 .. sigma_p_<n-1>`` (resampled onto the COSMO-SAC
grid used by the model). That artifact feeds
``scripts/data/build_sigma_profile_aux_stream.py``.

The VT-2005 DB is NOT shipped with this repo (it requires QM-derived COSMO files).
Download it separately and point this script at:
  * ``--profiles-dir``: a directory of per-compound profile files, each containing
    numeric rows whose first two columns are (sigma, p(sigma));
  * ``--index-csv``: a CSV mapping each profile file to a SMILES, with columns
    ``file`` and ``smiles`` (CAS/name->SMILES resolution is left to the user).

The grid is taken from the model config (cosmo_sac_sigma_min/max/n_bins); DB
profiles on a different grid are linearly resampled and clipped to >=0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tgnn_solv.config import TGNNSolvConfig


def _vt_index(fname: str) -> int | None:
    """VT-2005 compound index from a profile filename, e.g. VT2005-0001-PROF.txt -> 1
    (skip the '2005' catalogue prefix)."""
    nums = re.findall(r"\d+", str(fname))
    return next((int(t) for t in nums if t != "2005"), None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--profiles-dir", required=True, help="Directory of VT-2005 profile files.")
    p.add_argument("--index-csv", required=True, help="CSV with columns: file, smiles.")
    p.add_argument("--out-csv", default="results/sigma_profile_artifact/sigma_profiles.csv")
    p.add_argument("--summary-json", default="results/sigma_profile_artifact/summary.json")
    p.add_argument("--volume-index", default=None,
                   help="Optional VT-2005 master index (tab-separated) with a 'Vcosmo, A3' "
                        "column; when given, emit a 'v_cosmo' column (true COSMO cavity "
                        "volume) for the Staverman-Guggenheim size factor.")
    return p.parse_args()


def _read_profile_file(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    sig, prof = [], []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            s, p = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        sig.append(s)
        prof.append(p)
    if len(sig) < 2:
        return None
    return np.asarray(sig, dtype=float), np.asarray(prof, dtype=float)


def main() -> None:
    args = parse_args()
    cfg = TGNNSolvConfig()
    grid = np.linspace(cfg.cosmo_sac_sigma_min, cfg.cosmo_sac_sigma_max, cfg.cosmo_sac_n_bins)
    bin_cols = [f"sigma_p_{i}" for i in range(cfg.cosmo_sac_n_bins)]

    index = pd.read_csv(args.index_csv, low_memory=False)
    if not {"file", "smiles"} <= set(index.columns):
        raise ValueError("--index-csv must have columns 'file' and 'smiles'.")

    vol_by_idx: dict[int, float] = {}
    if args.volume_index:
        v2 = pd.read_csv(args.volume_index, sep="\t")
        vcol = next(c for c in v2.columns if "Vcosmo" in c)
        icol = next(c for c in v2.columns if "Index" in c)
        vol_by_idx = {int(r[icol]): float(r[vcol]) for _, r in v2.iterrows()}

    rows, n_fail, n_vol = [], 0, 0
    pdir = Path(args.profiles_dir)
    for rec in index.itertuples(index=False):
        fpath = pdir / str(rec.file)
        if not fpath.exists():
            n_fail += 1
            continue
        parsed = _read_profile_file(fpath)
        if parsed is None:
            n_fail += 1
            continue
        sig, prof = parsed
        prof = np.clip(np.interp(grid, sig, prof, left=0.0, right=0.0), 0.0, None)
        if prof.sum() <= 0:
            n_fail += 1
            continue
        row = {"smiles": str(rec.smiles), "sigma_area": float(prof.sum())}
        row.update({c: float(prof[i]) for i, c in enumerate(bin_cols)})
        if vol_by_idx:
            v = vol_by_idx.get(_vt_index(rec.file))
            row["v_cosmo"] = float(v) if v is not None else float("nan")
            n_vol += int(v is not None)
        rows.append(row)

    out_cols = ["smiles", "sigma_area", *bin_cols] + (["v_cosmo"] if vol_by_idx else [])
    out = pd.DataFrame(rows, columns=out_cols)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    summary = {
        "profiles_dir": args.profiles_dir, "index_csv": args.index_csv,
        "out_csv": args.out_csv, "n_ingested": int(len(out)),
        "n_failed": int(n_fail), "n_bins": int(cfg.cosmo_sac_n_bins),
        "volume_index": args.volume_index, "n_v_cosmo": int(n_vol),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
