#!/usr/bin/env python3
r"""Score the DEPLOYED COSMO-SAC-2002 closure on a PUBLISHED infinite-dilution
activity-coefficient evaluation set, through the same code path, convention and
profile source the paper uses.

WHY THIS EXISTS
---------------
The paper decomposes a fixed closure's error into a map part (B_closure) and an
input part (B_insuff) on two internally assembled IDAC sets (n=477 broad, n=60
VT-2005-matched).  Both were compiled here from primary literature, and both are
scored by this repository's own ``tgnn_solv.layers.CosmoSacLayer``.  A reader
therefore cannot tell whether the closure error being decomposed is the error a
published COSMO-SAC has, or an artifact of this data assembly or this
implementation.  ``run_closure_reference_validation.py`` already settles the
*implementation* half against the NIST reference code on identical profiles
(residual RMSE 0.0035 ln-gamma).  This script settles the *data* half: it takes
an evaluation set that a COSMO-SAC publication used, and scores our closure on it.

THE PUBLISHED SET
-----------------
de Souza Jr., Alcantara, Staudt, Coutinho and Soares, "Development of a COSMO-SAC
Parametrization with Advanced QM Method TZVPD-FINE", *Ind. Eng. Chem. Res.* 2025,
64, 14700-14711 (doi 10.1021/acs.iecr.5c01146) evaluate COSMO-SAC -- explicitly
"the original COSMO-SAC model presented by Lin and Sandler", i.e. the 2002 closure
this paper deploys -- on the IDAC database distributed with *The Properties of
Gases and Liquids*, 6th ed. (PGL6ed), a public GitHub repository.  They report
AAD = (1/N) sum |ln gamma_calc^inf - ln gamma_exp^inf| = 1.7457 (R^2 0.8780) for
COSMO-SAC with HF-TZVP sigma-profiles, and 0.6775 (R^2 0.9599) with their new
BP-TZVPD-FINE profiles.

The set itself is three tab/space-separated record files in
https://github.com/PGLadmin/PGLWrapper/tree/HEAD/Input :

  IdacRecLazzaroniDb2023.txt      4347 records, mostly non-aqueous, GC-derived
  IdacRecJaubert+TdeOrginW.txt    3272 records, organics at infinite dilution IN WATER
  IdacRecJaubert+TdeWinOrg.txt     519 records, WATER at infinite dilution in organics

Components are keyed by an integer CAS ("IdCas"); ``Input/IdCasTrcInchiName.txt``
maps that key to CASRN / name / InChIKey.

WHAT IS AND IS NOT LIKE-FOR-LIKE
--------------------------------
Same: the closure (COSMO-SAC-2002 / Lin-Sandler), the target (ln gamma^inf), the
metric (AAD as defined above), the evaluation set, and every record's own
temperature.  NOT the same: the sigma-profile source.  The published numbers use
HF-TZVP (Ferrarini et al.) or BP-TZVPD-FINE profiles; this paper deploys the
VT-2005 database, so that is what we feed.  The profile database is the single
uncontrolled axis, and it is exactly the axis the published paper shows matters
most (0.68 vs 1.75 for the same closure).  Our number must therefore be read as
"COSMO-SAC-2002 on the published set with VT-2005 profiles", bracketed by the two
published profile choices rather than equal to either.

CONVENTIONS
-----------
``res`` is the deployed convention -- residual (restoring) term only, no
Staverman-Guggenheim combinatorial contribution -- and is the headline here
because it is what the paper's B_closure is measured in.  ``full`` adds SG from
the VT-2005 COSMO cavity volumes and is reported alongside.

REPRODUCE
---------
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python \
        scripts/analysis/run_published_idac_closure_check.py --fetch

``--fetch`` downloads the four PGL6ed input files (about 2.2 MB, of which only the
three record files and a used-subset of the component map are deposited).  Without
it the script reads the deposited copies under
``results/published_idac_check/pgl6ed_source/``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from tgnn_solv.layers import CosmoSacLayer  # noqa: E402

OUT_DIR = REPO / "results/published_idac_check"
SRC_DIR = OUT_DIR / "pgl6ed_source"
SIGMA = REPO / "results/sigma_profile_artifact/sigma_profiles.csv"

RAW_BASE = "https://raw.githubusercontent.com/PGLadmin/PGLWrapper/HEAD/Input/"
RECORD_FILES = {
    "IdacRecLazzaroniDb2023.txt": "LazzaroniDb2023",
    "IdacRecJaubert+TdeOrginW.txt": "Jaubert_TDE_organic_in_water",
    "IdacRecJaubert+TdeWinOrg.txt": "Jaubert_TDE_water_in_organic",
}
MAP_FILE = "IdCasTrcInchiName.txt"

# Published anchors (de Souza Jr. et al., IECR 2025, 64, 14700; Table 3, full IDAC
# database, N = 6977).  AAD = mean |ln gamma_calc - ln gamma_exp|.
PUBLISHED = {
    "citation": (
        "de Souza Jr., E. T.; Alcantara, M. L.; Staudt, P. B.; Coutinho, J. A. P.; "
        "Soares, R. de P. Development of a COSMO-SAC Parametrization with Advanced "
        "QM Method TZVPD-FINE. Ind. Eng. Chem. Res. 2025, 64, 14700-14711."
    ),
    "doi": "10.1021/acs.iecr.5c01146",
    "evaluation_set": "PGL6ed IDAC database (github.com/PGLadmin/PGLWrapper)",
    "n_reported": 6977,
    "metric": "AAD = mean |ln gamma_calc^inf - ln gamma_exp^inf|",
    "rows": [
        {"model": "COSMO-SAC (Lin-Sandler 2002)", "profiles": "HF-TZVP", "aad": 1.7457, "r2": 0.8780},
        {"model": "COSMO-SAC (Lin-Sandler 2002)", "profiles": "BP-TZVPD-FINE", "aad": 0.6775, "r2": 0.9599},
        {"model": "COSMO-SAC-HB2", "profiles": "HF-TZVP", "aad": 0.4785, "r2": 0.9617},
        {"model": "COSMO-SAC-HB2", "profiles": "BP-TZVPD-FINE", "aad": 0.4640, "r2": 0.9728},
        {"model": "COSMO-SAC-HB2", "profiles": "BP-TZVP", "aad": 0.4773, "r2": 0.9698},
    ],
}

WATER_INCHIKEY = "XLYOFNOQVPJJNP-UHFFFAOYSA-N"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in list(RECORD_FILES) + [MAP_FILE]:
        url = RAW_BASE + urllib.parse.quote(name)
        target = dest / name
        with urllib.request.urlopen(url, timeout=120) as fh:  # noqa: S310 (fixed public host)
            target.write_bytes(fh.read())
        print(f"fetched {name} ({target.stat().st_size} B)")


def _read_records(path: Path, tag: str) -> pd.DataFrame:
    """All three files are whitespace/tab separated with one header line:
    solute_id, solvent_id, T(K), gamma_2^inf [, name columns / source flag]."""
    rows = []
    with path.open(errors="ignore") as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            parts = [p.strip() for p in parts if p.strip() != ""]
            if len(parts) < 4:
                continue
            rows.append(parts[:4])
    df = pd.DataFrame(rows, columns=["solute_id", "solvent_id", "T_K", "gamma_inf_exp"])
    df["record_file"] = path.name
    df["record_set"] = tag
    return df


def load_records(src: Path) -> pd.DataFrame:
    frames = [_read_records(src / name, tag) for name, tag in RECORD_FILES.items()]
    df = pd.concat(frames, ignore_index=True)
    df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
    df["gamma_inf_exp"] = pd.to_numeric(df["gamma_inf_exp"], errors="coerce")
    df = df.dropna(subset=["T_K", "gamma_inf_exp"])
    df = df[df["gamma_inf_exp"] > 0].reset_index(drop=True)
    df["m"] = np.log(df["gamma_inf_exp"].to_numpy(float))
    return df


def load_component_map(src: Path, out: Path) -> pd.DataFrame:
    """Full PGL6ed component map if present, else the deposited used-subset.

    The upstream map is 2.8 MB and 33,569 rows, of which this set touches a few
    hundred; only the subset is deposited, so a re-run without ``--fetch`` reads
    ``component_map_used.csv`` and reproduces the same rows.
    """
    full = src / MAP_FILE
    path = full if full.exists() else (out / "component_map_used.csv")
    m = pd.read_csv(path, sep="\t" if path == full else ",", dtype=str)
    m["IdCas"] = m["IdCas"].astype(str).str.strip()
    return m


def load_profiles(csv_path: Path):
    """VT-2005 profile artifact keyed by InChIKey.

    The deployed sigma-oracle matches on canonical SMILES; the published set is
    keyed by CAS, so InChIKey is the bridge.  Exact InChIKey only -- a first-block
    fallback would add 10 records out of ~7900 and buys nothing.
    """
    df = pd.read_csv(csv_path)
    cols = [f"sigma_p_{i}" for i in range(51)]
    table: dict[str, tuple[np.ndarray, float, float, str]] = {}
    for rec in df.itertuples(index=False):
        d = rec._asdict()
        smi = str(d.get("smiles", ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ik = Chem.MolToInchiKey(mol)
        if ik in table:
            continue
        p = np.array([float(d[c]) for c in cols], dtype=float)
        try:
            v = float(d.get("v_cosmo", float("nan")))
        except (TypeError, ValueError):
            v = float("nan")
        table[ik] = (p, float(d.get("sigma_area", p.sum())), v, smi)
    return table


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(table, solute_ik, solvent_ik, T, convention: str, batch: int = 2048) -> np.ndarray:
    layer = CosmoSacLayer()
    layer.eval()
    out = []
    for i in range(0, len(solute_ik), batch):
        s2 = solute_ik[i:i + batch]
        s1 = solvent_ik[i:i + batch]
        t = T[i:i + batch]
        p2 = torch.tensor(np.stack([table[k][0] for k in s2]), dtype=torch.float)
        A2 = torch.tensor([table[k][1] for k in s2], dtype=torch.float)
        p1 = torch.tensor(np.stack([table[k][0] for k in s1]), dtype=torch.float)
        A1 = torch.tensor([table[k][1] for k in s1], dtype=torch.float)
        Tt = torch.tensor(np.asarray(t, dtype=float), dtype=torch.float)
        if convention == "full":
            V2 = torch.tensor([table[k][2] for k in s2], dtype=torch.float)
            V1 = torch.tensor([table[k][2] for k in s1], dtype=torch.float)
        else:
            V2 = V1 = None
        with torch.no_grad():
            out.append(layer.ln_gamma_inf(p2, p1, A2, A1, V2, V1, Tt).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def metrics(m: np.ndarray, g: np.ndarray) -> dict:
    m = np.asarray(m, float)
    g = np.asarray(g, float)
    ok = np.isfinite(m) & np.isfinite(g)
    m, g = m[ok], g[ok]
    if m.size == 0:
        return {"n": 0}
    d = g - m
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    return {
        "n": int(m.size),
        "aad": round(float(np.mean(np.abs(d))), 4),
        "rmse": round(float(np.sqrt(np.mean(d ** 2))), 4),
        "mse": round(float(np.mean(d ** 2)), 4),
        "bias": round(float(np.mean(d)), 4),
        "r2": round(float(1.0 - np.sum(d ** 2) / ss_tot), 4) if ss_tot > 0 else float("nan"),
        "mean_m": round(float(m.mean()), 4),
    }


def _donor_class_fn():
    """Reuse the exact H-bond-donor classifier the paper's closure validation uses."""
    spec = importlib.util.spec_from_file_location(
        "closure_validation_mod", HERE / "run_closure_validation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["closure_validation_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.donor_class


def anchor_table(summary: dict) -> str:
    """The four-row anchor the article needs: the published COSMO-SAC-2002 on this
    set, the same closure as deployed here, and the paper's own two sets."""
    o = summary["ours"]
    p = summary["published_reference"]["rows"]
    own = summary["papers_own_sets_same_closure_res_convention"]

    def row(label, profiles, n, aad, r2):
        return (f"{label} & {profiles} & {n} & "
                f"{aad:.2f} & {'--' if r2 != r2 else f'{r2:.2f}'} \\\\")

    lines = [
        "% Auto-generated by scripts/analysis/run_published_idac_closure_check.py.",
        "% Do not hand-edit: re-run the script.",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"COSMO-SAC-2002 on & $\sigma$-profiles & $n$ & AAD & $R^2$ \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{The published evaluation set, as published"
        r"~\cite{desouza2025cosmosac}}}\\",
    ]
    for r in p:
        if r["model"].startswith("COSMO-SAC (Lin"):
            lines.append(row("PGL6ed IDAC database", r["profiles"],
                             summary["published_reference"]["n_reported"], r["aad"], r["r2"]))
    lines.append(r"\multicolumn{5}{@{}l}{\emph{The same set, through the closure this paper "
                 r"deploys (residual convention)}}\\")
    a = o["all_scored_records"]["res"]
    lines.append(row("PGL6ed IDAC database", "VT-2005", a["n"], a["aad"], a["r2"]))
    na = o["non_aqueous"]["res"]
    lines.append(row("\\quad non-aqueous rows", "VT-2005", na["n"], na["aad"], na["r2"]))
    aq = o["aqueous"]["res"]
    lines.append(row("\\quad aqueous rows", "VT-2005", aq["n"], aq["aad"], aq["r2"]))
    lines.append(r"\multicolumn{5}{@{}l}{\emph{This paper's own two sets, same closure and "
                 r"convention}}\\")
    b = own.get("broad_idac_set_477")
    if b:
        lines.append(row("broad IDAC set", "UD", b["n"], b["aad"], b["r2"]))
    v = own.get("vt2005_matched_set_60")
    if v:
        lines.append(row("VT-2005-matched set", "VT-2005", v["n"], v["aad"], v["r2"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download the PGL6ed input files before scoring")
    ap.add_argument("--source-dir", default=str(SRC_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    src = Path(args.source_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch(src)

    rec = load_records(src)
    cmap = load_component_map(src, out)
    id2ik = dict(zip(cmap["IdCas"], cmap["INCHIKEY"]))
    id2cas = dict(zip(cmap["IdCas"], cmap["CASRN"]))
    id2name = dict(zip(cmap["IdCas"], cmap["NAME"]))
    table = load_profiles(SIGMA)

    rec["solute_inchikey"] = rec["solute_id"].map(id2ik)
    rec["solvent_inchikey"] = rec["solvent_id"].map(id2ik)
    n_all = len(rec)
    rec = rec.dropna(subset=["solute_inchikey", "solvent_inchikey"])
    hit = rec[rec["solute_inchikey"].isin(table) & rec["solvent_inchikey"].isin(table)].copy()

    hit["solute_cas"] = hit["solute_id"].map(id2cas)
    hit["solvent_cas"] = hit["solvent_id"].map(id2cas)
    hit["solute_name"] = hit["solute_id"].map(id2name)
    hit["solvent_name"] = hit["solvent_id"].map(id2name)
    hit["solute_smiles"] = hit["solute_inchikey"].map(lambda k: table[k][3])
    hit["solvent_smiles"] = hit["solvent_inchikey"].map(lambda k: table[k][3])

    s2 = hit["solute_inchikey"].tolist()
    s1 = hit["solvent_inchikey"].tolist()
    T = hit["T_K"].to_numpy(float)
    hit["g_res"] = score(table, s2, s1, T, "res")
    hit["g_full"] = score(table, s2, s1, T, "full")

    donor_class = _donor_class_fn()
    solvent_class = {smi: donor_class(smi) for smi in hit["solvent_smiles"].unique()}
    hit["solvent_donor_class"] = hit["solvent_smiles"].map(solvent_class)
    hit["aqueous"] = (hit["solute_inchikey"] == WATER_INCHIKEY) | (
        hit["solvent_inchikey"] == WATER_INCHIKEY)

    m = hit["m"].to_numpy(float)
    summary = {
        "what": "deployed COSMO-SAC-2002 closure scored on a published IDAC evaluation set",
        "closure": "tgnn_solv.layers.CosmoSacLayer (COSMO-SAC-2002, Lin & Sandler)",
        "profile_source": "VT-2005 (results/sigma_profile_artifact/sigma_profiles.csv)",
        "matching": "PGL6ed IdCas -> InChIKey (Input/IdCasTrcInchiName.txt) -> VT-2005, exact InChIKey",
        "coverage": {
            "records_in_published_files": int(n_all),
            "records_scored": int(len(hit)),
            "fraction_scored": round(len(hit) / n_all, 4),
            "unique_pairs": int(hit.groupby(["solute_inchikey", "solvent_inchikey"]).ngroups),
            "by_record_set": hit["record_set"].value_counts().to_dict(),
            "n_reported_by_publication": PUBLISHED["n_reported"],
            "note": (
                "the publication reports 6977 points from the same repository; the three "
                "record files now in the repository hold 8138 usable records, of which "
                f"{len(hit)} have a VT-2005 profile on both sides. The record selection "
                "behind 6977 is not stated file-by-file, so the row sets are close but "
                "not proven identical."
            ),
        },
        "published_reference": PUBLISHED,
        "ours": {},
    }

    def add(label: str, mask) -> None:
        sub_m = m[mask]
        summary["ours"][label] = {
            "res": metrics(sub_m, hit["g_res"].to_numpy(float)[mask]),
            "full": metrics(sub_m, hit["g_full"].to_numpy(float)[mask]),
        }

    all_mask = np.ones(len(hit), dtype=bool)
    add("all_scored_records", all_mask)
    add("non_aqueous", ~hit["aqueous"].to_numpy())
    add("aqueous", hit["aqueous"].to_numpy())
    for tag in RECORD_FILES.values():
        add(f"set:{tag}", (hit["record_set"] == tag).to_numpy())
    for cls in ["strong_donor", "acceptor_only", "inert"]:
        add(f"solvent_class:{cls}", (hit["solvent_donor_class"] == cls).to_numpy())
    # 298 K, non-aqueous: the closest match in this published set to the regime the
    # paper's own VT-2005-matched set occupies (298 K, low-to-moderate gamma)
    near298 = (hit["T_K"].sub(298.15).abs() <= 1.0).to_numpy()
    add("298K_nonaqueous", near298 & ~hit["aqueous"].to_numpy())

    # pair-averaged (one row per solute/solvent/T), so replicate-heavy systems do
    # not dominate the aggregate
    agg = hit.groupby(["solute_inchikey", "solvent_inchikey", "T_K"], as_index=False).agg(
        m=("m", "mean"), g_res=("g_res", "mean"), g_full=("g_full", "mean"))
    summary["ours"]["all_pair_temperature_averaged"] = {
        "res": metrics(agg["m"].to_numpy(float), agg["g_res"].to_numpy(float)),
        "full": metrics(agg["m"].to_numpy(float), agg["g_full"].to_numpy(float)),
    }

    # --- the paper's own two sets, same closure, same convention, for continuity --
    own = {}
    for name, path, mcol, gcol, profiles in [
        ("broad_idac_set_477", REPO / "paper/si_tables/broad_idac_set_477.csv",
         "m_ln_gamma_inf", "g_2002_res", "UD (University of Delaware)"),
        ("vt2005_matched_set_60", REPO / "paper/si_tables/vt2005_matched_set_60.csv",
         "m", "g_res", "VT-2005"),
    ]:
        if path.exists():
            d = pd.read_csv(path)
            own[name] = {"profiles": profiles,
                         **metrics(d[mcol].to_numpy(float), d[gcol].to_numpy(float))}
    summary["papers_own_sets_same_closure_res_convention"] = own

    hit.to_csv(out / "scored_records.csv", index=False)
    used_ids = sorted(set(hit["solute_id"]) | set(hit["solvent_id"]))
    cmap[cmap["IdCas"].isin(used_ids)].to_csv(out / "component_map_used.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    tidy = []
    for label, block in summary["ours"].items():
        for conv, mtr in block.items():
            tidy.append({"subset": label, "convention": conv, **mtr})
    pd.DataFrame(tidy).to_csv(out / "summary.csv", index=False)
    (out / "table_closure_anchor.tex").write_text(anchor_table(summary))

    print(json.dumps({k: summary[k] for k in ("coverage",)}, indent=2))
    print(json.dumps(summary["ours"], indent=2))
    print(json.dumps(summary["papers_own_sets_same_closure_res_convention"], indent=2))


if __name__ == "__main__":
    main()
