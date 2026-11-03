"""
Data source loaders.

Each function downloads (if needed) and parses one data source,
returning a clean pandas DataFrame with canonical SMILES.

Sources:
  BigSolDBv2.1  — primary solubility (~121k records)
  Bradley ONS   — melting points (~28k)
  Curated       — T_m, ΔH_fus, Hansen, IDAC from NIST/literature
  CombiSolv-QM  — solvation free energies (~10k)
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, rdMolDescriptors

from .utils import (
    canonicalize,
    download_file,
    verify_csv,
    RAW_DIR,
)


# Common solvent densities at ~20-25 C (g/mL).
_DENSITY_G_ML_RAW = {
    "O": 0.997,
    "CO": 0.792,
    "CCO": 0.789,
    "CCCO": 0.804,
    "CC(O)C": 0.786,
    "CCCCO": 0.810,
    "OCCO": 1.113,
    "CCOCC": 0.713,
    "CC(=O)C": 0.784,
    "CC#N": 0.786,
    "CS(=O)C": 1.095,
    "CN(C)C=O": 0.944,
    "ClCCl": 1.326,
    "ClC(Cl)Cl": 1.489,
    "c1ccccc1": 0.879,
    "Cc1ccccc1": 0.867,
    "CCCCCC": 0.655,
    "CCCCCCC": 0.684,
    "CCCCCCCC": 0.703,
    "CCOC(=O)C": 0.902,
    "C1CCOC1": 0.889,
    "C1COCCO1": 1.033,
    "CC(=O)O": 1.049,
    "O=Cc1ccccc1": 1.045,
}
_DENSITY_G_ML_CANON = None
_IDAC_CANDIDATE_FILENAMES = (
    "idac.csv",
    "idac.tsv",
    "idac_gamma_inf.csv",
    "gamma_inf.csv",
    "idac_ln_gamma_inf.csv",
)
BIGSOLDB_DENSITIES_URL = (
    "https://raw.githubusercontent.com/levakrasnovs/BigSolDBv2.0/"
    "refs/heads/main/BigSolDBv2.1_densities.csv"
)
BIGSOLDB_DENSITIES_PATH = RAW_DIR / "BigSolDBv2.1_densities.csv"


def _normalize_column_name(name: str) -> str:
    """Normalize a raw column name for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Optional[str]:
    """Resolve the first matching column using normalized aliases."""
    normalized = {
        _normalize_column_name(column): column for column in df.columns
    }
    for alias in aliases:
        key = _normalize_column_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def _local_idac_path() -> Optional[tuple[str, object]]:
    """Return a configured local IDAC file if available."""
    env_path = os.environ.get("TGNN_SOLV_IDAC_PATH")
    if env_path:
        path = RAW_DIR.__class__(env_path).expanduser()
        if path.exists():
            return "env", path

    for filename in _IDAC_CANDIDATE_FILENAMES:
        path = RAW_DIR / filename
        if path.exists():
            return "raw", path
    return None


def _load_local_idac(path) -> pd.DataFrame:
    """Parse a local IDAC file into the canonical dataframe shape."""
    if not verify_csv(path):
        raise ValueError(f"IDAC file is not a valid text table: {path}")

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    solute_col = _resolve_column(
        df,
        (
            "solute_smiles",
            "smiles_solute",
            "solute",
            "solute_canonical_smiles",
            "solute_can_smiles",
        ),
    )
    solvent_col = _resolve_column(
        df,
        (
            "solvent_smiles",
            "smiles_solvent",
            "solvent",
            "solvent_canonical_smiles",
            "solvent_can_smiles",
        ),
    )
    gamma_col = _resolve_column(
        df,
        (
            "ln_gamma_inf",
            "gamma_inf",
            "log_gamma_inf",
            "lngamma_inf",
            "lngammainf",
        ),
    )
    temp_col = _resolve_column(
        df,
        (
            "temperature",
            "temperature_k",
            "temp",
            "temp_k",
            "t",
        ),
    )

    missing = []
    if solute_col is None:
        missing.append("solute_smiles")
    if solvent_col is None:
        missing.append("solvent_smiles")
    if gamma_col is None:
        missing.append("ln_gamma_inf")
    if missing:
        raise ValueError(
            "IDAC file is missing required columns. "
            f"Need aliases for: {', '.join(missing)}"
        )
    if _normalize_column_name(gamma_col) == "gammainf":
        print("  Interpreting `gamma_inf` column as precomputed ln_gamma_inf values")

    result = pd.DataFrame(
        {
            "solute_smiles": df[solute_col],
            "solvent_smiles": df[solvent_col],
            "ln_gamma_inf": pd.to_numeric(df[gamma_col], errors="coerce"),
        }
    )
    if temp_col is not None:
        result["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")
    else:
        result["temperature"] = 298.15

    result["solute_smiles"] = result["solute_smiles"].apply(canonicalize)
    result["solvent_smiles"] = result["solvent_smiles"].apply(canonicalize)
    result["temperature"] = result["temperature"].fillna(298.15)

    before = len(result)
    result = result.dropna(
        subset=["solute_smiles", "solvent_smiles", "ln_gamma_inf"]
    )
    dropped = before - len(result)
    result = result.drop_duplicates(
        subset=["solute_smiles", "solvent_smiles", "temperature"],
        keep="first",
    ).reset_index(drop=True)

    print(f"  Local file: {path}")
    print(f"  Parsed rows: {len(result)}")
    if dropped > 0:
        print(f"  Dropped invalid/empty rows: {dropped}")
    return result


def _density_map() -> dict:
    global _DENSITY_G_ML_CANON
    if _DENSITY_G_ML_CANON is None:
        _DENSITY_G_ML_CANON = {}
        for smi, rho in _DENSITY_G_ML_RAW.items():
            can = canonicalize(smi)
            if can:
                _DENSITY_G_ML_CANON[can] = rho
        _DENSITY_G_ML_CANON.update(_load_bigsoldb_roomtemp_density_map())
    return _DENSITY_G_ML_CANON


# ================================================================== #
#  BigSolDBv2.1                                                       #
# ================================================================== #

BIGSOLDB_URL = "https://zenodo.org/records/18552681/files/BigSolDBv2.1.csv"
BIGSOLDB_PATH = RAW_DIR / "BigSolDBv2.1.csv"


def _load_bigsoldb_roomtemp_density_map() -> dict[str, float]:
    """Build a solvent density map from the BigSolDB companion density table.

    The companion CSV stores densities by solvent name and temperature. For the
    current room-temperature conversions we select the measurements nearest to
    298.15 K for each solvent, average ties across sources, then join them to
    canonical `SMILES_Solvent` values from the main BigSolDB table.
    """
    if not download_file(
        BIGSOLDB_DENSITIES_URL,
        BIGSOLDB_DENSITIES_PATH,
        "BigSolDBv2.1 solvent densities",
    ) or not verify_csv(BIGSOLDB_DENSITIES_PATH):
        return {}

    if not download_file(BIGSOLDB_URL, BIGSOLDB_PATH, "BigSolDBv2.1") or not verify_csv(
        BIGSOLDB_PATH
    ):
        return {}

    try:
        density_df = pd.read_csv(BIGSOLDB_DENSITIES_PATH, low_memory=False)
        raw_df = pd.read_csv(BIGSOLDB_PATH, low_memory=False)
    except Exception:
        return {}

    solvent_name_col = _resolve_column(density_df, ("Solvent",))
    temp_col = _resolve_column(density_df, ("Temperature_K",))
    rho_col = _resolve_column(density_df, ("Density_g/cm^3", "Density_g_cm_3"))
    raw_solvent_name_col = _resolve_column(raw_df, ("Solvent",))
    raw_solvent_smiles_col = _resolve_column(raw_df, ("SMILES_Solvent",))
    if (
        solvent_name_col is None
        or temp_col is None
        or rho_col is None
        or raw_solvent_name_col is None
        or raw_solvent_smiles_col is None
    ):
        return {}

    name_to_smiles = raw_df[[raw_solvent_name_col, raw_solvent_smiles_col]].copy()
    name_to_smiles.columns = ["solvent_name", "solvent_smiles"]
    name_to_smiles["solvent_name"] = (
        name_to_smiles["solvent_name"].astype(str).str.strip().str.lower()
    )
    raw_smiles = name_to_smiles["solvent_smiles"].astype(str).str.strip()
    name_to_smiles = name_to_smiles.loc[
        raw_smiles.ne("-") & raw_smiles.ne("") & raw_smiles.str.lower().ne("nan")
    ].copy()
    unique_smiles = name_to_smiles["solvent_smiles"].astype(str).unique().tolist()
    RDLogger.DisableLog("rdApp.error")
    try:
        canonical_map = {smi: canonicalize(smi) for smi in unique_smiles}
    finally:
        RDLogger.EnableLog("rdApp.error")
    name_to_smiles["solvent_smiles"] = name_to_smiles["solvent_smiles"].map(canonical_map)
    name_to_smiles = (
        name_to_smiles.dropna(subset=["solvent_smiles"])
        .drop_duplicates(subset=["solvent_name", "solvent_smiles"])
        .drop_duplicates(subset=["solvent_name"], keep="first")
    )

    densities = density_df[[solvent_name_col, temp_col, rho_col]].copy()
    densities.columns = ["solvent_name", "temperature_k", "density_g_ml"]
    densities["solvent_name"] = densities["solvent_name"].astype(str).str.strip().str.lower()
    densities["temperature_k"] = pd.to_numeric(densities["temperature_k"], errors="coerce")
    densities["density_g_ml"] = pd.to_numeric(densities["density_g_ml"], errors="coerce")
    densities = densities.dropna(subset=["solvent_name", "temperature_k", "density_g_ml"])
    if densities.empty or name_to_smiles.empty:
        return {}

    densities["delta_room_temp"] = (densities["temperature_k"] - 298.15).abs()
    nearest = densities.groupby("solvent_name")["delta_room_temp"].transform("min")
    room_temp = densities.loc[nearest.eq(densities["delta_room_temp"])]
    room_temp = (
        room_temp.groupby("solvent_name", as_index=False)["density_g_ml"].mean()
    )
    merged = room_temp.merge(name_to_smiles, on="solvent_name", how="inner")
    return {
        str(row.solvent_smiles): float(row.density_g_ml)
        for row in merged.itertuples(index=False)
        if pd.notna(row.solvent_smiles) and pd.notna(row.density_g_ml)
    }


# Publications whose solvent columns are transposed. Both were identified from the corpus alone, by
# the homologous-series screen below, and then confirmed against an independent source: 10.1021/
# je4000718 stores naringenin at water 8.6e-3 to 1.0e-2 mole fraction and ethanol 4.0e-7 to 4.0e-6,
# monotonically across all nine temperatures, where naringenin's aqueous solubility is ~1e-7 and its
# ethanol solubility sits between methanol's and propanol's. 10.1021/je5001654 carries the same
# signature for chrysin. Quarantined by name so the screen's threshold does not have to carry them.
QUARANTINED_SOURCES: tuple[str, ...] = (
    "10.1021/je4000718",
    "10.1021/je5001654",
)

# The n-alcohol homologous series, in order. Solubility varies smoothly along it -- adding one CH2
# changes the solvent's polarity by a small, monotone step -- so an interior member cannot sit far
# below both of its neighbours. Water is deliberately NOT in this list: it is not a member of the
# series and for a lipophilic solute it is legitimately orders below all of them.
_ALCOHOL_SERIES: tuple[str, ...] = (
    "CO",          # methanol
    "CCO",         # ethanol
    "CCCO",        # 1-propanol
    "CCCCO",       # 1-butanol
    "CCCCCO",      # 1-pentanol
    "CCCCCCCCO",   # 1-octanol
)

# An interior member this far below BOTH neighbours, in ln x2, is flagged. The defect it is built to
# catch shows a 9.2 ln gap; a real homologous-series step is well under 1. Three ln units is roughly
# 20x in mole fraction and leaves the screen a wide margin against ordinary scatter.
_SERIES_INVERSION_LN: float = 3.0


def _quarantine_bad_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows from publications known to store transposed solvent columns."""
    col = "source_raw" if "source_raw" in df.columns else None
    if col is None or not QUARANTINED_SOURCES:
        return df
    src = df[col].astype(str)
    bad = pd.Series(False, index=df.index)
    for doi in QUARANTINED_SOURCES:
        bad |= src.str.contains(doi, regex=False, na=False)
    return df[~bad]


def _aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicate (solute, solvent, T) rows to the median ln x2 over sources.

    The median rather than the first row, and in log space rather than in mole fraction: solubility
    here spans thirty orders of magnitude, so an arithmetic mean is the largest replicate under
    another name, while the median of ln x2 is the geometric median of the solubility itself.
    """
    key = ["solute_smiles", "solvent_smiles", "temperature"]
    if df.empty or not set(key).issubset(df.columns):
        return df
    med = df.groupby(key, sort=False)["ln_x2"].transform("median")
    # Keep the row closest to the group's median so every other column stays a real, self-consistent
    # record rather than a blend of provenances; then write the median itself into ln_x2.
    order = (df["ln_x2"] - med).abs()
    keep = order.groupby([df[c] for c in key], sort=False).idxmin()
    out = df.loc[keep].copy()
    out["ln_x2"] = med.loc[keep].to_numpy()
    return out.sort_index()


def _screen_alcohol_series(
    df: pd.DataFrame, *, threshold_ln: float = _SERIES_INVERSION_LN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop (solute, solvent) cells whose value inverts the n-alcohol homologous series.

    For each solute, the series members it carries are compared in order. An INTERIOR member sitting
    more than ``threshold_ln`` below both of its immediate present neighbours is chemically
    impossible and is removed, together with every row of that solute in that solvent. Returns the
    screened frame and a report of what was removed.
    """
    empty = pd.DataFrame(columns=["solute_smiles", "solvent_smiles", "gap_ln", "n_rows"])
    if df.empty or "ln_x2" not in df.columns:
        return df, empty

    rank = {smi: i for i, smi in enumerate(_ALCOHOL_SERIES)}
    sub = df[df["solvent_smiles"].isin(rank)]
    if sub.empty:
        return df, empty

    per_cell = sub.groupby(["solute_smiles", "solvent_smiles"])["ln_x2"].median()
    flagged: list[dict[str, object]] = []
    for solute, cells in per_cell.groupby(level=0):
        present = sorted(
            ((rank[s], s, v) for (_, s), v in cells.items()), key=lambda t: t[0]
        )
        if len(present) < 3:
            continue
        for i in range(1, len(present) - 1):
            _, smi, val = present[i]
            gap = min(present[i - 1][2], present[i + 1][2]) - val
            if gap > threshold_ln:
                flagged.append(
                    {"solute_smiles": solute, "solvent_smiles": smi, "gap_ln": float(gap)}
                )
    if not flagged:
        return df, empty

    report = pd.DataFrame(flagged)
    bad_pairs = set(zip(report["solute_smiles"], report["solvent_smiles"]))
    mask = pd.Series(
        list(zip(df["solute_smiles"], df["solvent_smiles"])), index=df.index
    ).isin(bad_pairs)
    report["n_rows"] = report.apply(
        lambda r: int(
            (
                (df["solute_smiles"] == r["solute_smiles"])
                & (df["solvent_smiles"] == r["solvent_smiles"])
            ).sum()
        ),
        axis=1,
    )
    return df[~mask], report


def load_bigsoldb(*, preserve_source_detail: bool = False) -> pd.DataFrame:
    """
    Load BigSolDBv2.1 solubility database.

    Returns DataFrame with columns:
      solute_smiles, solvent_smiles, temperature, ln_x2, source

    When ``preserve_source_detail=True``, the returned frame also includes:
      source_family, source_raw

    and ``source`` is populated with the raw BigSolDB source identifier
    (typically a DOI) instead of the collapsed family label.
    """
    print("\n" + "=" * 60)
    print("Loading BigSolDBv2.1")
    print("=" * 60)

    got = download_file(BIGSOLDB_URL, BIGSOLDB_PATH, "BigSolDBv2.1")
    if not got or not verify_csv(BIGSOLDB_PATH):
        print("  BigSolDB not available — using synthetic fallback")
        return _synthetic_bigsoldb()

    df = pd.read_csv(BIGSOLDB_PATH, low_memory=False)
    n_raw = len(df)
    print(f"  Raw records: {n_raw:,}")
    print(f"  Columns: {list(df.columns)}")

    result = _process_bigsoldb_raw(
        df,
        preserve_source_detail=preserve_source_detail,
    )
    return result


def _process_bigsoldb_raw(
    df: pd.DataFrame,
    *,
    preserve_source_detail: bool = False,
) -> pd.DataFrame:
    """Parse and clean raw BigSolDB CSV."""

    # --- Map columns (case-insensitive) ---
    col_map = {
        "smiles_solute": "solute_smiles",
        "smiles_solvent": "solvent_smiles",
        "temperature_k": "temperature",
        "solubility(mole_fraction)": "x2",
        "solvent": "solvent_name",
        "compound_name": "solute_name",
        "logs(mol/l)": "logS",
        "fda_approved": "fda_approved",
        "source": "source_raw",
    }

    # Build lowercase → original column name mapping
    lower_map = {c.lower().replace(" ", ""): c for c in df.columns}

    result = pd.DataFrame()
    for key_lower, dst in col_map.items():
        src = lower_map.get(key_lower.replace(" ", ""))
        if src:
            result[dst] = df[src]

    # --- Canonicalize SMILES ---
    print("  Canonicalizing SMILES...")
    result["solute_smiles"] = result["solute_smiles"].apply(canonicalize)
    result["solvent_smiles"] = result["solvent_smiles"].apply(canonicalize)

    n_before = len(result)
    result = result.dropna(subset=["solute_smiles", "solvent_smiles"])
    print(f"  Invalid SMILES dropped: {n_before - len(result):,}")

    # --- Mole fraction → ln(x₂) ---
    if "x2" in result.columns:
        result["x2"] = pd.to_numeric(result["x2"], errors="coerce")
        valid_x2 = (result["x2"] > 0) & (result["x2"] <= 1.0)
        result.loc[~valid_x2, "x2"] = np.nan
        result["ln_x2"] = np.log(result["x2"])
    else:
        result["ln_x2"] = np.nan

    # Fallback: logS → x₂ where mole fraction is missing
    missing = result["ln_x2"].isna()
    if missing.any() and "logS" in result.columns:
        print(f"  Recovering ln(x₂) from logS for {missing.sum():,} records")
        logS = pd.to_numeric(result.loc[missing, "logS"], errors="coerce")
        S_mol_L = 10.0 ** logS

        water_smi = canonicalize("O")
        is_water = result.loc[missing, "solvent_smiles"] == water_smi

        C_solvent = pd.Series(np.nan, index=result.loc[missing].index)
        C_solvent[is_water] = 55.35
        density_map = _density_map()

        def _estimate_c_solvent_molarity(mol: Chem.Mol) -> Optional[float]:
            """Estimate solvent molarity from 3D molar volume."""
            if mol is None:
                return None
            try:
                mol_h = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 0xF00D
                if AllChem.EmbedMolecule(mol_h, params) != 0:
                    return None
                try:
                    AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
                except Exception:
                    pass
                vol_a3 = rdMolDescriptors.CalcMolVolume(mol_h)
            except Exception:
                return None
            if not np.isfinite(vol_a3) or vol_a3 <= 0:
                return None
            # 1 Å^3 = 1e-24 cm^3; V_m(cm^3/mol) = vol_a3 * NA * 1e-24
            v_m_cm3 = vol_a3 * 0.602214076
            if v_m_cm3 <= 0 or not np.isfinite(v_m_cm3):
                return None
            return 1000.0 / v_m_cm3  # mol/L

        solvent_cache = {}
        for idx in C_solvent[C_solvent.isna()].index:
            slv = result.loc[idx, "solvent_smiles"]
            if slv in solvent_cache:
                C_solvent[idx] = solvent_cache[slv]
                continue
            mol = Chem.MolFromSmiles(slv) if slv else None
            c_est = None
            rho = density_map.get(slv)
            if rho is not None and mol is not None:
                mw = Descriptors.MolWt(mol)
                if mw > 0:
                    c_est = 1000.0 * rho / mw
            if c_est is None:
                c_est = _estimate_c_solvent_molarity(mol)
            if c_est is None and mol is not None:
                c_est = 850.0 / Descriptors.MolWt(mol)
            if c_est is not None:
                solvent_cache[slv] = c_est
                C_solvent[idx] = c_est

        x2_calc = S_mol_L / (S_mol_L + C_solvent)
        ok = x2_calc.notna() & (x2_calc > 0) & (x2_calc <= 1)
        result.loc[ok[ok].index, "ln_x2"] = np.log(x2_calc[ok])

    result = result.dropna(subset=["ln_x2"])

    # --- Temperature ---
    if "temperature" in result.columns:
        result["temperature"] = pd.to_numeric(
            result["temperature"], errors="coerce"
        )
        result = result[
            (result["temperature"] >= 200) & (result["temperature"] <= 500)
        ]
        result["temperature"] = result["temperature"].fillna(298.15)
    else:
        result["temperature"] = 298.15

    # --- QC ---
    result = result[(result["ln_x2"] > -30) & (result["ln_x2"] <= 0)]

    # --- Quarantine of sources with transposed solvent columns ---
    result = _quarantine_bad_sources(result)

    # --- Replicate aggregation ---
    # Was `drop_duplicates(keep="first")`, which made the surviving label an arbitrary member of its
    # replicate group: over the 3,556 systems BigSolDB measures more than once it kept the largest
    # value 54.2% of the time and the smallest 43.8%. The median over sources is the consensus and
    # is robust to exactly the failure the quarantine above catches by name -- one transposed source
    # among three cannot move it, whereas "first" hands it the label outright.
    #
    # The median is taken in ln x2, i.e. of the log-solubility, so it is the geometric median in mole
    # fraction. That is the right space: solubility spans thirty orders of magnitude here and an
    # arithmetic mean would be the largest replicate in all but name.
    result = _aggregate_replicates(result)

    # --- Physical-consistency screen ---
    # Catches the same defect structurally rather than by DOI, so a source nobody has audited cannot
    # reintroduce it. Flags interior members of the n-alcohol homologous series that sit far below
    # both neighbours, which no solute does.
    result, _screen_report = _screen_alcohol_series(result)

    result["source_family"] = "BigSolDBv2.1"
    if preserve_source_detail and "source_raw" in result.columns:
        source_raw = (
            result["source_raw"]
            .fillna("BigSolDBv2.1")
            .astype(str)
            .str.strip()
            .replace("", "BigSolDBv2.1")
        )
        result["source_raw"] = source_raw
        result["source"] = source_raw
    else:
        result["source"] = result["source_family"]
    keep = ["solute_smiles", "solvent_smiles", "temperature", "ln_x2", "source"]
    if preserve_source_detail:
        keep.extend(["source_family", "source_raw"])
    for c in ["solute_name", "solvent_name", "fda_approved"]:
        if c in result.columns:
            keep.append(c)
    result = result[keep].reset_index(drop=True)

    _print_solubility_stats(result)
    return result


def _print_solubility_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for a solubility DataFrame."""
    n = len(df)
    n_sol = df["solute_smiles"].nunique()
    n_slv = df["solvent_smiles"].nunique()
    print("\n  === Summary ===")
    print(f"  Records:    {n:,}")
    print(f"  Solutes:    {n_sol:,}")
    print(f"  Solvents:   {n_slv:,}")
    print(f"  T range:    {df['temperature'].min():.0f}–"
          f"{df['temperature'].max():.0f} K")
    print(f"  ln(x₂):    [{df['ln_x2'].min():.2f}, "
          f"{df['ln_x2'].max():.2f}]")


def _synthetic_bigsoldb() -> pd.DataFrame:
    """Minimal synthetic data for demo/testing (30 records)."""
    print("  Using synthetic BigSolDB (30 records)")
    data = [
        ("CC(=O)Nc1ccc(O)cc1", "CCO", 298.15, 0.043),
        ("CC(=O)Nc1ccc(O)cc1", "O", 298.15, 0.0017),
        ("CC(=O)Nc1ccc(O)cc1", "CCCCCC", 298.15, 1e-5),
        ("CC(=O)Nc1ccc(O)cc1", "CC(=O)C", 298.15, 0.025),
        ("CC(=O)Nc1ccc(O)cc1", "CCO", 313.15, 0.065),
        ("c1ccc2ccccc2c1", "c1ccccc1", 298.15, 0.295),
        ("c1ccc2ccccc2c1", "CCO", 298.15, 0.082),
        ("c1ccc2ccccc2c1", "CCCCCC", 298.15, 0.11),
        ("c1ccc2ccccc2c1", "O", 298.15, 2.5e-5),
        ("OC(=O)c1ccccc1", "CCO", 298.15, 0.18),
        ("OC(=O)c1ccccc1", "O", 298.15, 0.0043),
        ("OC(=O)c1ccccc1", "CC(=O)C", 298.15, 0.22),
        ("OC(=O)c1ccccc1", "CCCCCC", 298.15, 0.003),
        ("c1ccccc1", "O", 298.15, 0.00041),
        ("CCCCCC", "O", 298.15, 2.3e-5),
        ("Oc1ccccc1", "O", 298.15, 0.015),
        ("Oc1ccccc1", "CCO", 298.15, 0.35),
        ("CC(=O)Oc1ccccc1C(=O)O", "O", 298.15, 6e-4),
        ("CC(=O)Oc1ccccc1C(=O)O", "CCO", 298.15, 0.042),
        ("Nc1ccccc1", "O", 298.15, 0.0068),
        ("ClC(Cl)Cl", "O", 298.15, 0.0012),
        ("CCCCCCCC", "O", 298.15, 5.7e-7),
        ("O=Cc1ccccc1", "O", 298.15, 0.005),
        ("O=Cc1ccccc1", "CCO", 298.15, 0.45),
        ("CCOC(=O)c1ccccc1", "O", 298.15, 1.3e-4),
        ("CCOC(=O)c1ccccc1", "CCO", 298.15, 0.30),
        ("c1ccc(-c2ccccc2)cc1", "c1ccccc1", 298.15, 0.15),
        ("c1ccc(-c2ccccc2)cc1", "CCCCCC", 298.15, 0.08),
        ("OC(=O)c1ccccc1", "Cc1ccccc1", 298.15, 0.12),
        ("Oc1ccccc1", "CC(=O)C", 298.15, 0.28),
    ]

    df = pd.DataFrame(
        data, columns=["solute_smiles", "solvent_smiles", "temperature", "x2"]
    )
    df["solute_smiles"] = df["solute_smiles"].apply(canonicalize)
    df["solvent_smiles"] = df["solvent_smiles"].apply(canonicalize)
    df["ln_x2"] = np.log(df["x2"].clip(lower=1e-20))
    df["source"] = "synthetic"
    df = df.drop(columns=["x2"])
    return df.dropna(subset=["solute_smiles", "solvent_smiles"]).reset_index(
        drop=True
    )


# ================================================================== #
#  Melting Points                                                     #
# ================================================================== #

BRADLEY_MP_URL = (
    "https://zenodo.org/records/19012778/files/Bradley_Melting_Point_Dataset.csv"
)

CURATED_MELTING_POINTS_K_RAW = {
    "CC(=O)Nc1ccc(O)cc1": 442.0,
    "c1ccc2ccccc2c1": 353.4,
    "OC(=O)c1ccccc1": 395.5,
    "c1ccccc1": 278.6,
    "Oc1ccccc1": 314.1,
    "CC(=O)Oc1ccccc1C(=O)O": 408.2,
    "O": 273.15,
    "CCO": 159.0,
    "CO": 175.5,
    "CC(=O)C": 178.2,
    "CCCCCC": 177.8,
    "CCCCCCC": 182.6,
    "CCCCCCCC": 216.4,
    "C1CCCCC1": 279.7,
    "c1ccncc1": 231.5,
    "Cc1ccccc1": 178.2,
    "CC#N": 227.2,
    "ClC(Cl)Cl": 209.6,
    "ClCCl": 176.0,
    "CS(=O)C": 291.7,
    "Nc1ccccc1": 267.1,
    "CC(=O)O": 289.8,
    "O=Cc1ccccc1": 247.2,
    "c1ccc(-c2ccccc2)cc1": 342.4,
    "c1ccc2cc3ccccc3cc2c1": 489.7,
    "c1ccc2c(c1)ccc1ccccc12": 372.4,
    "O=C(O)/C=C/c1ccccc1": 406.2,
    "OC(=O)c1cc(O)c(O)c(O)c1": 523.2,
    "O=C(O)C(O)(CC(=O)O)CC(=O)O": 426.2,
    "Clc1ccccc1": 228.0,
    "CC(=O)c1ccccc1": 293.2,
    "CCOC(=O)c1ccccc1": 238.5,
}

CURATED_FUSION_ENTHALPIES_J_MOL_RAW = {
    "CC(=O)Nc1ccc(O)cc1": 26400,
    "c1ccc2ccccc2c1": 19060,
    "OC(=O)c1ccccc1": 18020,
    "c1ccccc1": 9866,
    "Oc1ccccc1": 11290,
    "CC(=O)Oc1ccccc1C(=O)O": 29800,
    "O": 6010,
    "CCO": 4810,
    "CO": 3180,
    "CC(=O)C": 5770,
    "CCCCCC": 13080,
    "CCCCCCC": 14160,
    "CCCCCCCC": 20730,
    "C1CCCCC1": 2630,
    "c1ccncc1": 8280,
    "Cc1ccccc1": 6640,
    "CC#N": 8167,
    "O=Cc1ccccc1": 11300,
    "Nc1ccccc1": 10590,
    "CC(=O)O": 11540,
    "Clc1ccccc1": 9560,
    "c1ccc(-c2ccccc2)cc1": 18580,
    "c1ccc2cc3ccccc3cc2c1": 19200,
    "c1ccc2c(c1)ccc1ccccc12": 19300,
    "O=C(O)/C=C/c1ccccc1": 22400,
    "O=C(O)C(O)(CC(=O)O)CC(=O)O": 28700,
    "CCCO": 5200,
    "CC(O)C": 5370,
    "CCCCO": 9280,
    "OCCO": 9960,
    "ClC(Cl)Cl": 9500,
}


def _canonicalized_mapping_frame(
    mapping: dict[str, float],
    *,
    value_column: str,
) -> pd.DataFrame:
    rows = []
    for smi, value in mapping.items():
        can = canonicalize(smi)
        if can is None:
            continue
        rows.append({"solute_smiles": can, value_column: float(value)})
    if not rows:
        return pd.DataFrame(columns=["solute_smiles", value_column])
    return pd.DataFrame(rows).drop_duplicates(subset=["solute_smiles"]).reset_index(
        drop=True
    )


def _melting_points_to_kelvin(values: pd.Series) -> pd.Series:
    """Coerce a melting-point column to Kelvin, auto-detecting Celsius vs Kelvin.

    The raw column may be Celsius (legacy ``mpC``) or ALREADY Kelvin (the current
    ``bradley_mp.csv`` ships a ``Tm`` column in Kelvin, e.g. saccharin = 502 K).
    Decide by the column median: a melting-point dataset whose median already
    exceeds ~250 is in Kelvin (a 250 C median is unphysical for a broad organic
    set); otherwise it is Celsius and needs +273.15. This prevents the double
    C->K conversion that previously put every Bradley T_m ~273 K too high
    (saccharin -> 775 K) and corrupted the crystal-label supervision.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(numeric.median(skipna=True))
    if np.isfinite(median) and median > 250.0:
        return numeric  # already Kelvin
    return numeric + 273.15


def load_bradley_melting_points() -> pd.DataFrame:
    """Load the Bradley melting-point table only, converted to Kelvin."""
    bradley_mp = {}
    mp_path = RAW_DIR / "bradley_mp.csv"
    download_file(BRADLEY_MP_URL, mp_path, "Bradley Melting Points")

    if mp_path.exists() and verify_csv(mp_path):
        df = pd.read_csv(mp_path)
        smi_col = next(
            (c for c in df.columns if "smiles" in c.lower()),
            df.columns[0],
        )
        mp_col = next(
            (
                c
                for c in df.columns
                if any(x in c.lower() for x in ["mpc", "mp", "melting", "tm"])
            ),
            None,
        )
        if mp_col is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            mp_col = num_cols[0] if len(num_cols) > 0 else df.columns[-1]

        kelvin = _melting_points_to_kelvin(df[mp_col])
        for idx, smi_raw in df[smi_col].items():
            smi = canonicalize(str(smi_raw))
            if smi is None:
                continue
            val_k = kelvin.loc[idx]
            if pd.notna(val_k) and 80 < float(val_k) < 800:
                bradley_mp[smi] = float(val_k)

    return _canonicalized_mapping_frame(bradley_mp, value_column="T_m")


def load_curated_melting_points() -> pd.DataFrame:
    """Load the small curated melting-point table in Kelvin."""
    return _canonicalized_mapping_frame(
        CURATED_MELTING_POINTS_K_RAW,
        value_column="T_m",
    )


def load_curated_fusion_enthalpies() -> pd.DataFrame:
    """Load the curated fusion-enthalpy table in J/mol."""
    return _canonicalized_mapping_frame(
        CURATED_FUSION_ENTHALPIES_J_MOL_RAW,
        value_column="dH_fus",
    )


def load_melting_points() -> pd.DataFrame:
    """
    Load melting points: Bradley ONS (°C) + curated NIST (K).

    Curated values override Bradley when both exist for the same
    compound (NIST is more reliable).

    Returns DataFrame: solute_smiles, T_m (K).
    """
    print("\n" + "=" * 60)
    print("Loading Melting Points")
    print("=" * 60)

    bradley_df = load_bradley_melting_points()
    curated_df = load_curated_melting_points()
    n_override = len(
        set(bradley_df["solute_smiles"].tolist())
        & set(curated_df["solute_smiles"].tolist())
    )

    result = pd.concat([bradley_df, curated_df], ignore_index=True)
    result = result.drop_duplicates(subset=["solute_smiles"], keep="last").reset_index(
        drop=True
    )

    print(f"  Bradley valid: {len(bradley_df):,}")
    print(f"  Curated: {len(curated_df):,}, overrides: {n_override}")
    print(f"  Total: {len(result):,}")
    if not result.empty:
        print(f"  T_m range: [{result['T_m'].min():.0f}, "
              f"{result['T_m'].max():.0f}] K")
    return result


# ================================================================== #
#  Fusion Enthalpies                                                  #
# ================================================================== #

def load_fusion_enthalpies() -> pd.DataFrame:
    """Curated ΔH_fus from NIST WebBook (J/mol)."""
    print("\n" + "=" * 60)
    print("Loading ΔH_fus data")
    print("=" * 60)
    df = load_curated_fusion_enthalpies()
    print(f"  Records: {len(df)}")
    return df


# ================================================================== #
#  Hansen Solubility Parameters                                       #
# ================================================================== #

def load_hansen() -> pd.DataFrame:
    """Hansen parameters (δd, δp, δh) in MPa^0.5."""
    print("\n" + "=" * 60)
    print("Loading Hansen parameters")
    print("=" * 60)

    data = [
        ("O", 15.5, 16.0, 42.3),
        ("CO", 15.1, 12.3, 22.3),
        ("CCO", 15.8, 8.8, 19.4),
        ("CCCO", 16.0, 6.8, 17.4),
        ("CC(O)C", 15.8, 6.1, 16.4),
        ("CCCCO", 16.0, 5.7, 15.8),
        ("CCCCCO", 15.9, 4.5, 13.9),
        ("CCCCCCCCO", 16.0, 3.3, 11.9),
        ("OCCO", 17.0, 11.0, 26.0),
        ("OCC(O)CO", 17.4, 12.1, 29.3),
        ("CC(=O)C", 15.5, 10.4, 7.0),
        ("CCC(=O)C", 16.0, 9.0, 5.1),
        ("CC#N", 15.3, 18.0, 6.1),
        ("CS(=O)C", 18.4, 16.4, 10.2),
        ("CN(C)C=O", 17.4, 13.7, 11.3),
        ("C1CCOC1", 16.8, 5.7, 8.0),
        ("CCOCC", 14.5, 2.9, 5.1),
        ("ClCCl", 17.0, 7.3, 7.1),
        ("ClC(Cl)Cl", 17.8, 3.1, 5.7),
        ("ClC(Cl)(Cl)Cl", 17.8, 0.0, 0.6),
        ("c1ccccc1", 18.4, 0.0, 2.0),
        ("Cc1ccccc1", 18.0, 1.4, 2.0),
        ("CCCCCC", 14.9, 0.0, 0.0),
        ("CCCCCCC", 15.3, 0.0, 0.0),
        ("C1CCCCC1", 16.8, 0.0, 0.2),
        ("CCOC(=O)C", 15.8, 5.3, 7.2),
        ("CC(=O)O", 14.5, 8.0, 13.5),
        ("c1ccncc1", 19.0, 8.8, 5.9),
        ("CN1CCCC1=O", 18.0, 12.3, 7.2),
        ("C1COCCO1", 19.0, 1.8, 7.4),
        ("O=[N+]([O-])c1ccccc1", 20.0, 8.6, 4.1),
        ("CC(=O)Nc1ccc(O)cc1", 18.5, 10.2, 14.1),
        ("c1ccc2ccccc2c1", 19.2, 2.0, 5.9),
        ("OC(=O)c1ccccc1", 18.2, 7.0, 9.8),
        ("CC(=O)Oc1ccccc1C(=O)O", 17.6, 6.4, 10.6),
        ("Oc1ccccc1", 18.0, 5.9, 14.9),
        ("Nc1ccccc1", 19.4, 5.1, 10.2),
        ("c1ccc(-c2ccccc2)cc1", 19.7, 1.0, 2.0),
        ("c1ccc2cc3ccccc3cc2c1", 20.3, 0.7, 2.4),
        ("Clc1ccccc1", 19.0, 4.3, 2.0),
        ("CC(=O)c1ccccc1", 19.6, 8.6, 3.7),
        ("CCCCCCCC", 15.5, 0.0, 0.0),
        ("CC(C)CC(C)C", 14.5, 0.0, 0.0),
    ]

    rows = []
    for smi, dd, dp, dh in data:
        can = canonicalize(smi)
        if can:
            rows.append({
                "solute_smiles": can,
                "hansen_d": dd, "hansen_p": dp, "hansen_h": dh,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["solute_smiles"])
    print(f"  Records: {len(df)}")
    return df


# ================================================================== #
#  IDAC (γ∞)                                                          #
# ================================================================== #

def load_idac() -> pd.DataFrame:
    """Infinite dilution activity coefficients ln(γ∞)."""
    print("\n" + "=" * 60)
    print("Loading IDAC data")
    print("=" * 60)

    local = _local_idac_path()
    if local is not None:
        origin, path = local
        if origin == "env":
            print("  Using TGNN_SOLV_IDAC_PATH override")
        else:
            print("  Using local IDAC file from raw data directory")
        return _load_local_idac(path)

    print("  No local IDAC file found; using built-in fallback")

    data = [
        ("CCCCCC", "O", 7.60, 298.15),
        ("c1ccccc1", "O", 7.78, 298.15),
        ("Cc1ccccc1", "O", 8.15, 298.15),
        ("C1CCCCC1", "O", 7.80, 298.15),
        ("CCCCCCC", "O", 8.60, 298.15),
        ("CCCCCCCC", "O", 9.65, 298.15),
        ("c1ccc2ccccc2c1", "O", 8.60, 298.15),
        ("CCO", "O", 1.39, 298.15),
        ("CO", "O", 0.69, 298.15),
        ("CC(=O)C", "O", 1.95, 298.15),
        ("CC#N", "O", 0.92, 298.15),
        ("CCCCO", "O", 3.33, 298.15),
        ("Oc1ccccc1", "O", 3.90, 298.15),
        ("O", "CCO", 0.47, 298.15),
        ("O", "CCCCCC", 5.70, 298.15),
        ("O", "c1ccccc1", 5.30, 298.15),
        ("O", "CC(=O)C", 0.30, 298.15),
        ("O", "ClC(Cl)Cl", 5.50, 298.15),
        ("CCCCCC", "CCO", 3.80, 298.15),
        ("c1ccccc1", "CCO", 2.50, 298.15),
        ("CCO", "CCCCCC", 4.80, 298.15),
        ("CC(=O)C", "CCCCCC", 3.20, 298.15),
        ("CCCCCC", "c1ccccc1", 1.05, 298.15),
        ("c1ccc2ccccc2c1", "c1ccccc1", 0.15, 298.15),
        ("Cc1ccccc1", "c1ccccc1", 0.05, 298.15),
        ("CC(=O)C", "CCO", 0.75, 298.15),
        ("CCO", "CC(=O)C", 0.50, 298.15),
        ("CCO", "O", 1.30, 313.15),
        ("CCO", "O", 1.47, 283.15),
        ("CCCCCC", "O", 7.45, 313.15),
        ("c1ccccc1", "O", 7.50, 313.15),
    ]

    rows = []
    for sol, slv, lng, T in data:
        sol_c, slv_c = canonicalize(sol), canonicalize(slv)
        if sol_c and slv_c:
            rows.append({
                "solute_smiles": sol_c,
                "solvent_smiles": slv_c,
                "ln_gamma_inf": lng,
                "temperature": T,
            })

    df = pd.DataFrame(rows)
    print(f"  Records: {len(df)}")
    return df
