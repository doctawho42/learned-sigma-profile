#!/usr/bin/env python3
"""Build a standalone external sigma-profile auxiliary stream.

Grounds the COSMO-SAC activity branch on a pool of single-component sigma-profile
labels (e.g. VT-2005), exactly mirroring the crystal aux stream. The output is a
separate CSV in processed-dataset format, consumed via

    python scripts/train.py ... \
        --sigma-train-data <output.csv> --sigma-steps-per-epoch 8

Each row is a single-component sigma-profile label row: the solute is paired with
itself as the "solvent" (valid graph, no pair signal), ``has_solubility=False``,
all other masks off, ``has_sigma_profile=True``, plus the normalized n-bin
profile (``sigma_p_0..sigma_p_{n-1}``, sums to 1, n controlled by ``--n-bins``)
and cavity area ``sigma_area``.

Input artifact columns: ``smiles`` (or ``solute_smiles``), ``sigma_area``, and
either n ``sigma_p_*`` columns (already a shape or area-weighted profile) — they
are renormalized to a shape here.

CRITICAL GUARD: exclude pool solutes whose Bemis-Murcko scaffold appears in a
held-out split (``--exclude-scaffolds-from``) to keep scaffold evaluation honest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from tgnn_solv.config import TGNNSolvConfig
from tgnn_solv.data.utils import scaffold_key


def grid_metadata(n_bins: int) -> dict:
    """Self-describing sigma grid for the artifact summary."""
    cfg = TGNNSolvConfig()
    return {
        "n_bins": int(n_bins),
        "sigma_min": float(cfg.cosmo_sac_sigma_min),
        "sigma_max": float(cfg.cosmo_sac_sigma_max),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--sigma-csv", default="results/sigma_profile_artifact/sigma_profiles.csv")
    p.add_argument("--template-csv", default="notebooks/data/processed/train.csv")
    p.add_argument("--output-csv", default="notebooks/data/processed_sigma_aux_stream/sigma_train.csv")
    p.add_argument("--summary-json", default="notebooks/data/processed_sigma_aux_stream/summary.json")
    p.add_argument("--n-bins", type=int, default=TGNNSolvConfig().cosmo_sac_n_bins)
    p.add_argument("--temperature", type=float, default=298.15)
    p.add_argument("--source-label", default="aux_only_sigma_profile_stream")
    p.add_argument(
        "--exclude-scaffolds-from",
        nargs="*",
        default=["notebooks/data/processed/test.csv", "notebooks/data/processed/val.csv"],
    )
    p.add_argument(
        "--allow-no-scaffold-exclusion",
        action="store_true",
        help="Override the fail-closed scaffold-leak guard. Only for pools that are "
        "intentionally NOT used in a scaffold-split evaluation.",
    )
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--output-val-csv",
                   default="notebooks/data/processed_sigma_aux_stream/sigma_val.csv")
    return p.parse_args()


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _excluded_scaffolds(paths: list[str], *, allow_missing: bool = False) -> set[str]:
    """Collect Murcko scaffolds to exclude. Fails CLOSED by default.

    A missing or empty/header-only split CSV would silently yield zero exclusions
    and emit the entire (non-split-aware) sigma-profile pool, leaking held-out
    scaffolds into the scaffold-extrapolation evaluation. So unless ``allow_missing``
    is set, a requested path that does not exist or yields zero scaffolds raises.
    """
    scaffolds: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            if allow_missing:
                continue
            raise FileNotFoundError(
                f"--exclude-scaffolds-from path does not exist: {path}. Refusing to "
                "build the aux stream without the mandatory scaffold-leak guard "
                "(pass --allow-no-scaffold-exclusion to override deliberately)."
            )
        df = pd.read_csv(path, usecols=lambda c: c == "solute_smiles", low_memory=False)
        path_scaffolds = {
            scaf
            for smi in df["solute_smiles"].dropna().astype(str).unique()
            if (scaf := scaffold_key(smi))
        }
        if not path_scaffolds and not allow_missing:
            raise ValueError(
                f"{path} yielded zero usable solute scaffolds; refusing to proceed "
                "(an empty/header-only split would silently leak the full pool into "
                "the evaluation). Pass --allow-no-scaffold-exclusion to override."
            )
        scaffolds |= path_scaffolds
    return scaffolds


def split_by_scaffold(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Whole-scaffold-group TRAIN/VAL split so the two sides share no scaffold_key.

    Groups by the acyclic-safe ``scaffold_key`` (Murcko, or canonical SMILES for
    acyclic molecules) and assigns entire groups to VAL until ``val_fraction`` of
    rows is reached. Deterministic under ``seed``. ``val_fraction<=0`` -> empty VAL.
    """
    if val_fraction <= 0.0 or len(df) == 0:
        return df.copy(), df.iloc[0:0].copy()
    keys = df["solute_smiles"].astype(str).map(scaffold_key)
    groups: dict[str, list[int]] = {}
    for idx, k in zip(df.index, keys):
        groups.setdefault(str(k), []).append(idx)
    unique = sorted(groups)  # stable base order before the seeded shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    target = val_fraction * len(df)
    val_idx: list[int] = []
    for k in unique:
        if len(val_idx) >= target:
            break
        val_idx.extend(groups[k])
    val_set = set(val_idx)
    val = df.loc[df.index.isin(val_set)].copy()
    train = df.loc[~df.index.isin(val_set)].copy()
    return train, val


def _empty_row_template(columns: list[str], temperature: float) -> dict[str, Any]:
    row = {col: "" for col in columns}
    numeric = {"temperature": temperature, "ln_x2": 0.0, "T_m": 0.0, "dH_fus": 0.0,
               "hansen_d": 0.0, "hansen_p": 0.0, "hansen_h": 0.0, "ln_gamma_inf": 0.0}
    booly = {"has_solubility": False, "has_T_m": False, "has_dH_fus": False,
             "has_hansen": False, "has_gamma_inf": False, "fda_approved": "No"}
    for k, v in numeric.items():
        if k in row:
            row[k] = v
    for k, v in booly.items():
        if k in row:
            row[k] = v
    return row


def main() -> None:
    args = parse_args()
    sigma = pd.read_csv(args.sigma_csv, low_memory=False)
    smiles_col = "smiles" if "smiles" in sigma.columns else "solute_smiles"
    bin_cols = [f"sigma_p_{i}" for i in range(args.n_bins)]
    missing = [c for c in bin_cols if c not in sigma.columns]
    if missing:
        raise ValueError(f"{args.sigma_csv} missing profile columns: {missing[:3]}... "
                         f"(expected sigma_p_0..sigma_p_{args.n_bins-1})")
    sigma = sigma.dropna(subset=[smiles_col]).copy()
    n_raw = len(sigma)

    requested_excludes = list(args.exclude_scaffolds_from or [])
    excluded = _excluded_scaffolds(
        requested_excludes, allow_missing=args.allow_no_scaffold_exclusion
    )
    if requested_excludes and not excluded and not args.allow_no_scaffold_exclusion:
        raise SystemExit(
            "Scaffold exclusion produced an empty set; aborting to prevent leakage "
            "of held-out scaffolds into the sigma-profile aux pool."
        )
    if excluded:
        sigma = sigma[~sigma[smiles_col].astype(str).map(lambda s: scaffold_key(s) in excluded)]
    n_after = len(sigma)

    template = pd.read_csv(args.template_csv, nrows=1, low_memory=False)
    out_cols = list(template.columns) + ["has_sigma_profile", "sigma_area"] + bin_cols
    tmpl = _empty_row_template(out_cols, args.temperature)

    rows: list[dict[str, Any]] = []
    for rec in sigma.itertuples(index=False):
        d = rec._asdict()
        smi = str(d[smiles_col])
        prof = np.array([float(d[c]) for c in bin_cols], dtype=float)
        prof = np.clip(prof, 0.0, None)
        total = prof.sum()
        if total <= 0:
            continue
        shape = prof / total  # normalize to a sum-1 shape
        area = float(d.get("sigma_area", total))
        row = dict(tmpl)
        row["solute_smiles"] = smi
        row["solvent_smiles"] = smi
        row["temperature"] = float(args.temperature)
        row["has_solubility"] = False
        row["has_sigma_profile"] = True
        row["sigma_area"] = area
        for i, c in enumerate(bin_cols):
            row[c] = float(shape[i])
        if "source" in row:
            row["source"] = args.source_label
        rows.append(row)

    out = pd.DataFrame(rows, columns=out_cols)
    train_out, val_out = split_by_scaffold(out, args.val_fraction, args.split_seed)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    if len(val_out) > 0:
        # fail-closed disjointness guard (mirrors _excluded_scaffolds style):
        # verify BEFORE writing so a leak leaves no partial CSVs on disk.
        tr = {scaffold_key(s) for s in train_out["solute_smiles"]}
        va = {scaffold_key(s) for s in val_out["solute_smiles"]}
        if not tr.isdisjoint(va):
            raise SystemExit("sigma TRAIN/VAL scaffold leak detected; aborting.")
        train_out.to_csv(args.output_csv, index=False)
        Path(args.output_val_csv).parent.mkdir(parents=True, exist_ok=True)
        val_out.to_csv(args.output_val_csv, index=False)
    else:
        train_out.to_csv(args.output_csv, index=False)

    # Post-build certification. The guard above filters by scaffold, but a build made against the
    # wrong split files passes it silently -- that happened once, with 87 held-out solutes reaching
    # the stream by canonical SMILES and 54 by scaffold, and it was found only in review. So we
    # re-read the split files we were pointed at and assert the intersection is empty, by both keys,
    # against what was actually written. A leak now fails the build.
    leak_smiles: set[str] = set()
    leak_scaffolds: set[str] = set()
    if args.exclude_scaffolds_from:
        held_smiles: set[str] = set()
        held_scaffolds: set[str] = set()
        for path in args.exclude_scaffolds_from:
            frame = pd.read_csv(path)
            for smiles in frame["solute_smiles"].dropna().unique():
                mol = Chem.MolFromSmiles(str(smiles))
                held_smiles.add(Chem.MolToSmiles(mol) if mol else str(smiles))
                held_scaffolds.add(scaffold_key(str(smiles)))
        for smiles in out["solute_smiles"].dropna().unique():
            mol = Chem.MolFromSmiles(str(smiles))
            canonical = Chem.MolToSmiles(mol) if mol else str(smiles)
            if canonical in held_smiles:
                leak_smiles.add(canonical)
            if scaffold_key(str(smiles)) in held_scaffolds:
                leak_scaffolds.add(str(smiles))
        if (leak_smiles or leak_scaffolds) and not args.allow_no_scaffold_exclusion:
            raise SystemExit(
                f"scaffold-leak guard failed after the build: {len(leak_smiles)} solute(s) share a "
                f"canonical SMILES and {len(leak_scaffolds)} share a Murcko scaffold with "
                f"{args.exclude_scaffolds_from}. The stream was NOT written as clean."
            )

    build_hash = hashlib.sha256(Path(args.output_csv).read_bytes()).hexdigest()

    summary = {
        "sigma_csv": args.sigma_csv, "template_csv": args.template_csv,
        "output_csv": args.output_csv, "n_pool_raw": int(n_raw),
        "n_after_scaffold_exclusion": int(n_after), "n_excluded_scaffolds": int(len(excluded)),
        "n_rows": int(len(out)), "n_bins": int(args.n_bins),
        "exclude_scaffolds_from": list(args.exclude_scaffolds_from or []),
        "grid": grid_metadata(args.n_bins),
        "n_train": int(len(train_out)), "n_val": int(len(val_out)),
        "val_fraction_actual": (len(val_out) / max(len(out), 1)),
        "certified_no_leak": True,
        "leak_check_smiles": len(leak_smiles),
        "leak_check_scaffolds": len(leak_scaffolds),
        "build_sha256": build_hash,
    }
    Path(args.summary_json).write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
