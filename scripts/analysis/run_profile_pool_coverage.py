#!/usr/bin/env python3
"""What a larger reference-profile database buys, measured rather than argued.

THE QUESTION. The Supporting Information states that enlarging the reference profile database would
not materially grow the set on which a learned and a reference profile can both be scored. The
argument it gave for that was taken on the wrong margin: it counted solutes lacking a profile GIVEN
their solvent has one, which conditions away every pair that fails on both sides -- exactly the
cell a larger database fills. The conclusion may still be right, but a conditional count cannot
establish it. This script measures it instead, on the surface the claim is about.

THE TWO POOLS, same split and same matcher, one thing different:
  * the UD / VT-2005 compound list, the deposited reference (``complist.txt``, 2262 keys);
  * **PubChemQC COSMO**, the largest open sigma-profile database, Zenodo
    ``10.5281/zenodo.20672192`` (CC-BY-4.0): 361,583 profiles over
    334,714 distinct InChIKeys, PySCF BP86/def2-SVP, already split into the three
    Hsieh--Sandler--Lin sub-profiles on the same 51-bin grid this pipeline uses.

Neither pool is redistributed here, so both are passed as paths and the script reports what is
missing rather than failing. The index file this reads is ``sigma_profile_index.parquet`` from that
record; it carries ``InChIKey``, ``SMILES``, ``Heavy_Atoms`` and ``Molecular_Weight`` per CID.

WHAT IS MEASURED. Coverage is counted on the **labelled test rows**, both components at once,
matched on InChIKey -- the quantity ``check_ud_bigsoldb_intersection.py`` reports and the one the
claim is about. Counted in **distinct solutes** as well as rows, because rows cluster inside solutes
and the solute count is the effective sample size. The 2x2 over the two components is printed in
full, so the cell the old argument could not see is visible.

THE SIZE CONTROL, and it is the point. A pool can miss a molecule by accident of coverage or by
construction. The script therefore reports the pool's own heavy-atom and molecular-weight range
beside the covered and uncovered test solutes': a pool whose ceiling sits below the corpus's
median cannot be the route to the corpus however large it is, and that is a different statement
from "this pool happens not to have them".

Usage:
    python scripts/analysis/run_profile_pool_coverage.py \\
        --ud ~/COSMOSAC/profiles/UD --pubchemqc /path/to/sigma_profile_index.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger                     # noqa: E402
from rdkit.Chem import Descriptors, inchi            # noqa: E402

RDLogger.DisableLog("rdApp.*")

SPLIT = ROOT / "notebooks/data/processed/test.csv"
UD = Path.home() / "COSMOSAC/profiles/UD"
OUT = ROOT / "results/profile_pool_coverage/summary.json"
KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")

#: Provenance of the second pool, printed into the deposit so the number carries its source.
PUBCHEMQC_DOI = "10.5281/zenodo.20672192"


def inchikeys(smiles: pd.Series) -> pd.Series:
    cache: dict[str, str | None] = {}

    def one(s: object) -> str | None:
        if not isinstance(s, str):
            return None
        if s not in cache:
            mol = Chem.MolFromSmiles(s)
            try:
                cache[s] = inchi.MolToInchiKey(mol) if mol is not None else None
            except Exception:
                cache[s] = None
        return cache[s]

    return smiles.map(one)


def size_of(smiles: str) -> tuple[int | None, float | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol.GetNumHeavyAtoms(), round(Descriptors.MolWt(mol), 1)


def coverage(lab: pd.DataFrame, keys: set[str], k2: pd.Series, k1: pd.Series) -> dict:
    c2, c1 = k2.isin(keys), k1.isin(keys)
    cells = {}
    for sol in (True, False):
        for svt in (True, False):
            m = (c2 == sol) & (c1 == svt)
            name = f"solute_{'yes' if sol else 'no'}_solvent_{'yes' if svt else 'no'}"
            cells[name] = {"rows": int(m.sum()),
                           "solutes": int(lab.loc[m, "solute_smiles"].nunique())}
    both = c2 & c1
    sub = lab[both]
    return {
        "pool_size": len(keys),
        "n_matched_rows": int(both.sum()),
        "n_solutes": int(sub["solute_smiles"].nunique()),
        "n_solvents": int(sub["solvent_smiles"].nunique()),
        "n_pairs": int(sub.groupby(["solute_smiles", "solvent_smiles"]).ngroups),
        "cells": cells,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ud", type=Path, default=UD, help="directory holding complist.txt")
    ap.add_argument("--pubchemqc", type=Path, default=None,
                    help=f"sigma_profile_index.parquet from {PUBCHEMQC_DOI}")
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    pools: dict[str, set[str]] = {}
    sizes: dict[str, dict] = {}

    ud_list = a.ud / "complist.txt"
    if ud_list.exists():
        pools["ud_vt2005"] = set(KEY.findall(ud_list.read_text(errors="ignore")))
    else:
        print(f"UD list absent at {ud_list}; that arm is not reported")

    if a.pubchemqc and a.pubchemqc.exists():
        idx = pd.read_parquet(a.pubchemqc,
                              columns=["InChIKey", "Heavy_Atoms", "Molecular_Weight"])
        k = idx["InChIKey"].dropna().astype(str)
        pools["pubchemqc_cosmo"] = set(k[k.str.fullmatch(KEY.pattern)])
        sizes["pubchemqc_cosmo"] = {
            "n_rows": int(len(idx)),
            "heavy_atoms_median": float(idx.Heavy_Atoms.median()),
            "heavy_atoms_max": int(idx.Heavy_Atoms.max()),
            "mw_median": float(idx.Molecular_Weight.median()),
            "mw_max": float(idx.Molecular_Weight.max()),
            "share_over_25_heavy_atoms": float((idx.Heavy_Atoms > 25).mean()),
        }
    else:
        print(f"PubChemQC index not given or absent; that arm is not reported ({PUBCHEMQC_DOI})")

    if not pools:
        raise SystemExit("no pool available; nothing is reported and nothing is written")

    d = pd.read_csv(a.split, low_memory=False)
    lab = d[d["has_solubility"].astype(bool)] if "has_solubility" in d else d
    k2, k1 = inchikeys(lab["solute_smiles"]), inchikeys(lab["solvent_smiles"])

    result = {
        "what": "both-sides reference-profile coverage of the labelled test rows, by pool",
        "match_rule": "InChIKey on solute and solvent at once",
        "pubchemqc_source": PUBCHEMQC_DOI,
        "n_labelled_test_rows": int(len(lab)),
        "n_test_solutes": int(lab["solute_smiles"].nunique()),
        "n_test_solvents": int(lab["solvent_smiles"].nunique()),
        "pools": {name: coverage(lab, keys, k2, k1) for name, keys in pools.items()},
        "pool_sizes": sizes,
    }

    # The size control: are the solutes a pool misses systematically bigger than the ones it has?
    if "pubchemqc_cosmo" in pools:
        keys = pools["pubchemqc_cosmo"]
        sol = pd.DataFrame({"smiles": sorted(lab["solute_smiles"].dropna().unique())})
        sol["key"] = inchikeys(sol["smiles"])
        sol[["heavy", "mw"]] = sol["smiles"].apply(lambda s: pd.Series(size_of(s)))
        sol["covered"] = sol["key"].isin(keys)
        result["test_solute_size_by_coverage"] = {
            ("covered" if f else "missing"): {
                "n": int(len(g)),
                "heavy_atoms_median": float(g.heavy.median()),
                "heavy_atoms_min": int(g.heavy.min()),
                "heavy_atoms_max": int(g.heavy.max()),
                "mw_median": float(g.mw.median()),
            }
            for f, g in sol.groupby("covered")
        }
        svt = pd.DataFrame({"smiles": sorted(lab["solvent_smiles"].dropna().unique())})
        svt["key"] = inchikeys(svt["smiles"])
        result["test_solvents_covered"] = {
            "n_total": int(len(svt)),
            "n_covered": int(svt["key"].isin(keys).sum()),
        }

    for name, c in result["pools"].items():
        print(f"{name:<18} pool {c['pool_size']:>7}  rows {c['n_matched_rows']:>5}  "
              f"solutes {c['n_solutes']:>3}/{result['n_test_solutes']}  pairs {c['n_pairs']:>4}")
    if "test_solute_size_by_coverage" in result:
        for k, v in result["test_solute_size_by_coverage"].items():
            print(f"  solutes {k:<8} n={v['n']:>3}  heavy atoms median "
                  f"{v['heavy_atoms_median']:.0f} (range {v['heavy_atoms_min']}-{v['heavy_atoms_max']})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
