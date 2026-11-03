"""
DataBuilder: merge primary solubility data with all auxiliary sources.

Uses pd.merge (vectorized) instead of row-wise apply for speed.
Includes filter_for_sle() to remove records incompatible with the
solid-liquid equilibrium framework.
"""

from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from .utils import canonicalize


# ================================================================== #
#  SLE compatibility filter                                           #
# ================================================================== #

def filter_for_sle(
    df: pd.DataFrame,
    x2_max: float = 0.98,
    min_atoms: int = 2,
    max_atoms: int = 200,
    include_water_solubility: bool = True,
) -> pd.DataFrame:
    """
    Remove records incompatible with the SLE framework.

    Filters:
      1. Miscible systems (x₂ > x2_max): SLE requires a solid phase
      2. Molecules too small or too large for GNN
      3. Fragmented molecules (ions, salts with no bonds)
      4. Self-solvation (solute == solvent)

    Parameters
    ----------
    df : DataFrame with columns solute_smiles, solvent_smiles, ln_x2
    x2_max : maximum allowed mole fraction (removes miscible pairs)
    min_atoms : minimum heavy atoms per molecule
    max_atoms : maximum heavy atoms per molecule
    include_water_solubility : keep supervised water-solvent rows even though
        water (`O`) has only one atom and would otherwise fail `min_atoms`

    Returns
    -------
    Filtered DataFrame (reset index).
    """
    n0 = len(df)
    water_smi = canonicalize("O") or "O"

    # 1. Miscible systems
    ln_x2_max = np.log(x2_max)
    miscible = df["ln_x2"] > ln_x2_max
    n_misc = miscible.sum()
    df = df[~miscible].copy()

    # 2. Structural validity
    bad_idx = []
    for idx, row in df.iterrows():
        for col in ["solute_smiles", "solvent_smiles"]:
            mol = Chem.MolFromSmiles(row[col])
            if mol is None:
                bad_idx.append(idx)
                break
            n_at = mol.GetNumAtoms()
            # Preserve supervised water rows by exempting the solvent-side
            # water graph from the minimum-atom threshold only.
            is_water_solvent = (
                include_water_solubility
                and col == "solvent_smiles"
                and row[col] == water_smi
            )
            if n_at > max_atoms or (n_at < min_atoms and not is_water_solvent):
                bad_idx.append(idx)
                break
            # Fragmented molecules (e.g. [Na+].[Cl-])
            if mol.GetNumBonds() == 0 and n_at > 1:
                bad_idx.append(idx)
                break

    n_struct = len(bad_idx)
    df = df.drop(index=bad_idx)

    # 3. Self-solvation
    self_solv = df["solute_smiles"] == df["solvent_smiles"]
    n_self = self_solv.sum()
    df = df[~self_solv]

    print(
        f"  SLE filter: {n0} → {len(df)} "
        f"(misc: -{n_misc}, struct: -{n_struct}, self: -{n_self})"
    )
    return df.reset_index(drop=True)


# ================================================================== #
#  DataBuilder                                                        #
# ================================================================== #

class DataBuilder:
    """
    Merge primary solubility DataFrame with auxiliary data sources.

    All merges use pd.merge (vectorized) for speed.

    Usage::

        builder = DataBuilder()
        builder.add_mp(mp_df)
        builder.add_dh(dh_df)
        builder.add_hansen(hansen_df)
        builder.add_gamma(idac_df)
        builder.add_dg(combisolv_df)
        unified = builder.build(bigsoldb_df)
    """

    def __init__(self) -> None:
        self.mp_df: Optional[pd.DataFrame] = None
        self.dh_df: Optional[pd.DataFrame] = None
        self.hansen_df: Optional[pd.DataFrame] = None
        self.gamma_df: Optional[pd.DataFrame] = None

    def add_mp(self, df: pd.DataFrame) -> None:
        """Add melting points: solute_smiles, T_m."""
        self.mp_df = (
            df[["solute_smiles", "T_m"]]
            .drop_duplicates("solute_smiles")
            .copy()
        )

    def add_dh(self, df: pd.DataFrame) -> None:
        """Add fusion enthalpies: solute_smiles, dH_fus."""
        self.dh_df = (
            df[["solute_smiles", "dH_fus"]]
            .drop_duplicates("solute_smiles")
            .copy()
        )

    def add_hansen(self, df: pd.DataFrame) -> None:
        """Add Hansen parameters: solute_smiles, hansen_d/p/h."""
        self.hansen_df = (
            df[["solute_smiles", "hansen_d", "hansen_p", "hansen_h"]]
            .drop_duplicates("solute_smiles")
            .copy()
        )

    def add_gamma(self, df: pd.DataFrame) -> None:
        """Add IDAC: solute_smiles, solvent_smiles, ln_gamma_inf, temperature."""
        self.gamma_df = (
            df[["solute_smiles", "solvent_smiles",
                "ln_gamma_inf", "temperature"]]
            .rename(columns={"temperature": "gamma_T"})
            .copy()
        )

    def add_dg(self, df: pd.DataFrame) -> None:
        """Add solvation free energies: solute/solvent_smiles, dG_solv, temperature."""
        self.dg_df = (
            df[["solute_smiles", "solvent_smiles",
                "dG_solv", "temperature"]]
            .rename(columns={"temperature": "dg_T"})
            .copy()
        )

    def build(self, solubility_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build unified dataset.

        Every solubility record gets auxiliary columns attached via
        left merge.  Additionally, auxiliary-only records (compounds
        with T_m but not in the solubility set) are appended for
        Phase 1 pretraining.
        """
        print("\n" + "=" * 60)
        print("Building unified dataset")
        print("=" * 60)

        df = solubility_df.copy()
        df["has_solubility"] = True

        # --- Merge T_m ---
        if self.mp_df is not None:
            df = df.merge(self.mp_df, on="solute_smiles", how="left")
        else:
            df["T_m"] = np.nan
        df["has_T_m"] = df["T_m"].notna()
        df["T_m"] = df["T_m"].fillna(0.0)

        # --- Merge ΔH_fus ---
        if self.dh_df is not None:
            df = df.merge(self.dh_df, on="solute_smiles", how="left")
        else:
            df["dH_fus"] = np.nan
        df["has_dH_fus"] = df["dH_fus"].notna()
        df["dH_fus"] = df["dH_fus"].fillna(0.0)

        # --- Merge Hansen ---
        if self.hansen_df is not None:
            df = df.merge(self.hansen_df, on="solute_smiles", how="left")
        else:
            for c in ["hansen_d", "hansen_p", "hansen_h"]:
                df[c] = np.nan
        df["has_hansen"] = df["hansen_d"].notna()
        for c in ["hansen_d", "hansen_p", "hansen_h"]:
            df[c] = df[c].fillna(0.0)

        # --- Merge γ∞ (pair match, nearest temperature within 5 K) ---
        if self.gamma_df is not None:
            df = df.copy()
            df["_row_id"] = np.arange(len(df))
            df["temperature"] = pd.to_numeric(
                df["temperature"], errors="coerce"
            )

            gamma = self.gamma_df.copy()
            gamma["gamma_T"] = pd.to_numeric(
                gamma["gamma_T"], errors="coerce"
            )
            gamma = gamma.dropna(subset=["gamma_T"])
            if not gamma.empty:
                gamma = gamma.sort_values(
                    ["gamma_T", "solute_smiles", "solvent_smiles"]
                )
                df_merge = df[df["temperature"].notna()].copy()
                df_no_temp = df[df["temperature"].isna()].copy()
                if not df_merge.empty:
                    df_merge = df_merge.sort_values(
                        [
                            "temperature",
                            "solute_smiles",
                            "solvent_smiles",
                            "_row_id",
                        ]
                    )
                    df_merge = pd.merge_asof(
                        df_merge,
                        gamma,
                        by=["solute_smiles", "solvent_smiles"],
                        left_on="temperature",
                        right_on="gamma_T",
                        direction="nearest",
                        tolerance=5.0,
                    )
                df = pd.concat(
                    [df_merge, df_no_temp], ignore_index=False
                )
                df = df.sort_values("_row_id").drop(columns=["_row_id"])
                df = df.drop(columns=["gamma_T"], errors="ignore")
            else:
                df = df.drop(columns=["_row_id"])
        else:
            df["ln_gamma_inf"] = np.nan
        if "ln_gamma_inf" not in df.columns:
            df["ln_gamma_inf"] = np.nan
        df["has_gamma_inf"] = df["ln_gamma_inf"].notna()
        df["ln_gamma_inf"] = df["ln_gamma_inf"].fillna(0.0)

        # --- Dedup after merges (merges can multiply rows) ---
        df = df.drop_duplicates(
            subset=["solute_smiles", "solvent_smiles", "temperature"],
            keep="first",
        )

        # --- Auxiliary-only records for Phase 1 pretraining ---
        existing_solutes = set(df["solute_smiles"].unique())
        existing_pairs = {
            (str(row["solute_smiles"]), str(row["solvent_smiles"]))
            for _, row in df.iterrows()
        }
        existing_keys = {
            (
                str(row["solute_smiles"]),
                str(row["solvent_smiles"]),
                float(row["temperature"]),
            )
            for _, row in df[df["temperature"].notna()].iterrows()
        }
        aux_rows = []
        water_smi = canonicalize("O")

        if self.mp_df is not None:
            for _, r in self.mp_df.iterrows():
                smi = r["solute_smiles"]
                if smi not in existing_solutes:
                    key = (str(smi), str(water_smi), 298.15)
                    row = {
                        "solute_smiles": smi,
                        "solvent_smiles": water_smi,
                        "temperature": 298.15,
                        "ln_x2": 0.0,
                        "has_solubility": False,
                        "T_m": r["T_m"],
                        "has_T_m": True,
                        "dH_fus": 0.0,
                        "has_dH_fus": False,
                        "hansen_d": 0.0,
                        "hansen_p": 0.0,
                        "hansen_h": 0.0,
                        "has_hansen": False,
                        "ln_gamma_inf": 0.0,
                        "has_gamma_inf": False,
                        "source": "aux_only",
                    }
                    aux_rows.append(row)
                    existing_keys.add(key)

        if self.gamma_df is not None:
            gamma_aux = self.gamma_df.dropna(
                subset=["solute_smiles", "solvent_smiles", "gamma_T", "ln_gamma_inf"]
            ).drop_duplicates(
                subset=["solute_smiles", "solvent_smiles", "gamma_T"],
                keep="first",
            )
            for _, r in gamma_aux.iterrows():
                pair_key = (str(r["solute_smiles"]), str(r["solvent_smiles"]))
                if pair_key in existing_pairs:
                    continue
                key = (
                    pair_key[0],
                    pair_key[1],
                    float(r["gamma_T"]),
                )
                if key in existing_keys:
                    continue
                aux_rows.append(
                    {
                        "solute_smiles": r["solute_smiles"],
                        "solvent_smiles": r["solvent_smiles"],
                        "temperature": float(r["gamma_T"]),
                        "ln_x2": 0.0,
                        "has_solubility": False,
                        "T_m": 0.0,
                        "has_T_m": False,
                        "dH_fus": 0.0,
                        "has_dH_fus": False,
                        "hansen_d": 0.0,
                        "hansen_p": 0.0,
                        "hansen_h": 0.0,
                        "has_hansen": False,
                        "ln_gamma_inf": float(r["ln_gamma_inf"]),
                        "has_gamma_inf": True,
                        "source": "aux_only_gamma",
                    }
                )
                existing_keys.add(key)

        if aux_rows:
            aux_df = pd.DataFrame(aux_rows)

            # Attach T_m to aux-only records where available
            if self.mp_df is not None:
                mp_map = self.mp_df.set_index("solute_smiles")["T_m"]
                matched = aux_df["solute_smiles"].map(mp_map)
                aux_df.loc[matched.notna(), "T_m"] = matched.dropna()
                aux_df.loc[matched.notna(), "has_T_m"] = True

            # Attach dH_fus to aux-only records where available
            if self.dh_df is not None:
                dh_map = self.dh_df.set_index("solute_smiles")["dH_fus"]
                matched = aux_df["solute_smiles"].map(dh_map)
                aux_df.loc[matched.notna(), "dH_fus"] = matched.dropna()
                aux_df.loc[matched.notna(), "has_dH_fus"] = True

            # Attach Hansen to aux-only records where available
            if self.hansen_df is not None:
                h_map = self.hansen_df.set_index("solute_smiles")
                for c in ["hansen_d", "hansen_p", "hansen_h"]:
                    matched = aux_df["solute_smiles"].map(h_map[c])
                    aux_df.loc[matched.notna(), c] = matched.dropna()
                aux_df["has_hansen"] = aux_df["solute_smiles"].isin(
                    self.hansen_df["solute_smiles"]
                )

            df = pd.concat([df, aux_df], ignore_index=True)
            print(f"  Aux-only records added: {len(aux_rows):,}")

        # --- Statistics ---
        n = len(df)
        print("\n  === Unified Dataset ===")
        print(f"  Total:          {n:>7,d}")
        print(f"  Solubility:     {df['has_solubility'].sum():>7,d}")
        print(f"  T_m:            {df['has_T_m'].sum():>7,d} "
              f"({df['has_T_m'].sum() / n * 100:.1f}%)")
        print(f"  ΔH_fus:         {df['has_dH_fus'].sum():>7,d}")
        print(f"  Hansen:         {df['has_hansen'].sum():>7,d}")
        print(f"  γ∞:             {df['has_gamma_inf'].sum():>7,d}")
        print(f"  Solutes:        {df['solute_smiles'].nunique():>7,d}")
        print(f"  Solvents:       {df['solvent_smiles'].nunique():>7,d}")

        return df
