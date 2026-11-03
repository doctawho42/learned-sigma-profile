"""Process-oriented optimization workflows built on top of solvent screening."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional runtime guard
    Chem = None

from ..config import TGNNSolvConfig
from ..model import TGNNSolv
from ..baselines.direct_gnn import DirectGNN
from .core import clamp
from .solvent_screening import SolventScreener, _cost_rank, _toxicity_severity


class ProcessOptimizer:
    """Optimize process-facing operating windows on top of model predictions."""

    def __init__(
        self,
        model: TGNNSolv | DirectGNN,
        cfg: TGNNSolvConfig,
        device: torch.device,
        solvent_library: Sequence[str] | Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.screener = SolventScreener(model, cfg, self.device, solvent_library=solvent_library)
        self.model_family = self.screener.model_family

    @property
    def solvent_library(self) -> list[dict[str, Any]]:
        return self.screener.solvent_library

    def optimize_crystallization(
        self,
        solute_smiles: str,
        target_yield: float = 0.80,
        T_range: tuple[float, float] = (273.0, 373.0),
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find solvents and hot/cold endpoints that maximize crystallization yield."""

        constraints = constraints or {}
        min_green = int(constraints.get("min_green_score", 1))
        max_tox = int(constraints.get("max_toxicity_class", 3))
        min_dissolve = float(constraints.get("min_dissolving_concentration_mg_mL", constraints.get("min_dissolving_concentration", 1.0)))
        max_boiling_raw = constraints.get("max_boiling_point_K")
        max_boiling = float(max_boiling_raw) if max_boiling_raw is not None else None
        min_delta_t = float(constraints.get("min_delta_T", 12.0))
        grid_points = max(8, int(constraints.get("grid_points", 16)))
        top_n = max(1, int(constraints.get("top_n", 5)))

        candidate_rows: list[dict[str, Any]] = []
        t_low = max(float(T_range[0]), 273.0)
        t_high = float(T_range[1])
        canonical_solute = self.screener._require_smiles(solute_smiles, "solute")

        for entry in self._metadata_filtered_library(
            min_green=min_green,
            max_tox=max_tox,
            max_boiling=max_boiling,
        ):
            boiling_point = entry.get("boiling_point_K")
            hot_upper = min(
                t_high,
                373.0,
                (float(boiling_point) - 10.0) if boiling_point is not None else t_high,
            )
            if hot_upper - t_low < min_delta_t:
                continue
            scan_df = self.screener._augment_scan(
                self.screener._temperature_scan(
                    canonical_solute,
                    str(entry["smiles"]),
                    T_min=t_low,
                    T_max=hot_upper,
                    n_points=grid_points,
                ),
                canonical_solute,
                str(entry["smiles"]),
            )
            scan_df = scan_df.sort_values("T").reset_index(drop=True)
            for cold_idx in range(0, len(scan_df) - 1):
                for hot_idx in range(cold_idx + 1, len(scan_df)):
                    cold_row = scan_df.iloc[cold_idx]
                    hot_row = scan_df.iloc[hot_idx]
                    if float(hot_row["T"]) - float(cold_row["T"]) < min_delta_t:
                        continue
                    hot_mg_ml = float(hot_row.get("solubility_mg_mL", float("nan")))
                    if math.isfinite(hot_mg_ml) and hot_mg_ml < min_dissolve:
                        continue
                    x2_hot = float(hot_row["x2"])
                    x2_cold = float(cold_row["x2"])
                    if x2_hot <= 0:
                        continue
                    yield_fraction = clamp((x2_hot - x2_cold) / max(x2_hot, 1e-12), 0.0, 1.0)
                    supersaturation_ratio = x2_hot / max(x2_cold, 1e-12)
                    hot_pred = self.screener._predict_one(canonical_solute, str(entry["smiles"]), float(hot_row["T"]))
                    cold_pred = self.screener._predict_one(canonical_solute, str(entry["smiles"]), float(cold_row["T"]))
                    score = (
                        0.62 * yield_fraction
                        + 0.18 * clamp(math.log10(max(supersaturation_ratio, 1.0) + 1e-9), 0.0, 1.0)
                        + 0.10 * clamp((float(entry.get("green_score") or 0) - 1.0) / 9.0, 0.0, 1.0)
                        + 0.10 * clamp((hot_mg_ml - min_dissolve) / max(min_dissolve, 1.0), 0.0, 1.0)
                    )
                    candidate_rows.append(
                        {
                            "solvent_name": entry["name"],
                            "solvent_smiles": entry["smiles"],
                            "solvent_class": entry["solvent_class"],
                            "T_hot": float(hot_row["T"]),
                            "T_cold": float(cold_row["T"]),
                            "delta_T": float(hot_row["T"]) - float(cold_row["T"]),
                            "yield": yield_fraction,
                            "target_yield_gap": yield_fraction - float(target_yield),
                            "x2_hot": x2_hot,
                            "x2_cold": x2_cold,
                            "ln_x2_hot": float(hot_row["ln_x2"]),
                            "ln_x2_cold": float(cold_row["ln_x2"]),
                            "hot_solubility_mg_mL": hot_mg_ml,
                            "cold_solubility_mg_mL": float(cold_row.get("solubility_mg_mL", float("nan"))),
                            "supersaturation_ratio": supersaturation_ratio,
                            "green_score": entry.get("green_score"),
                            "toxicity_class": entry.get("ild_class"),
                            "boiling_point_K": entry.get("boiling_point_K"),
                            "cost_relative": entry.get("cost_relative"),
                            "recommended": yield_fraction >= float(target_yield),
                            "objective_score": score,
                            "Phi_hot": hot_pred.get("Phi"),
                            "Phi_cold": cold_pred.get("Phi"),
                            "gamma_2_hot": hot_pred.get("gamma_2"),
                            "gamma_2_cold": cold_pred.get("gamma_2"),
                            "tau_12_hot": hot_pred.get("tau_12"),
                            "tau_21_hot": hot_pred.get("tau_21"),
                            "T_m_pred": hot_pred.get("T_m"),
                            "dH_fus_pred": hot_pred.get("dH_fus"),
                            "temperature_scan": scan_df.where(scan_df.notna(), None).to_dict(orient="records"),
                        }
                    )

        candidate_rows.sort(
            key=lambda row: (
                bool(row["recommended"]),
                float(row["yield"]),
                float(row["objective_score"]),
            ),
            reverse=True,
        )
        return candidate_rows[:top_n]

    def optimize_extraction(
        self,
        solute_smiles: str,
        source_solvent: str,
        T: float = 298.15,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Rank extraction solvents by partition leverage and operability."""

        constraints = constraints or {}
        min_green = int(constraints.get("min_green_score", 1))
        max_tox = int(constraints.get("max_toxicity_class", 3))
        max_boiling = float(constraints.get("max_boiling_point_K", 500.0))
        min_partition = float(constraints.get("min_partition_coefficient", 1.0))

        canonical_solute = self.screener._require_smiles(solute_smiles, "solute")
        canonical_source = self.screener._require_smiles(source_solvent, "source solvent")
        source_pred = self.screener._predict_one(canonical_solute, canonical_source, float(T))
        source_meta = self._solvent_metadata(canonical_source, source_pred)

        rows: list[dict[str, Any]] = []
        for entry in self._metadata_filtered_library(min_green=min_green, max_tox=max_tox, max_boiling=max_boiling):
            if str(entry["smiles"]) == canonical_source:
                continue
            pred = self.screener._predict_one(canonical_solute, str(entry["smiles"]), float(T))
            partition = float(pred["x2"]) / max(float(source_pred["x2"]), 1e-12)
            if partition < min_partition:
                continue
            miscible = self.screener._solvent_miscibility(source_meta, entry)
            bp = entry.get("boiling_point_K")
            evaporation_score = clamp((430.0 - float(bp)) / 140.0, 0.0, 1.0) if bp is not None else 0.3
            score = (
                0.62 * clamp(math.log10(partition + 1e-9) / 2.0, 0.0, 1.0)
                + 0.18 * (0.0 if miscible else 1.0)
                + 0.10 * evaporation_score
                + 0.10 * clamp((float(entry.get("green_score") or 0) - 1.0) / 9.0, 0.0, 1.0)
            )
            rows.append(
                {
                    "solvent_name": entry["name"],
                    "solvent_smiles": entry["smiles"],
                    "solvent_class": entry["solvent_class"],
                    "source_solvent": canonical_source,
                    "x2_source": float(source_pred["x2"]),
                    "x2_extract_solvent": float(pred["x2"]),
                    "ln_x2_source": float(source_pred["ln_x2"]),
                    "ln_x2_extract_solvent": float(pred["ln_x2"]),
                    "partition_coefficient": partition,
                    "miscible_with_source": miscible,
                    "boiling_point_K": bp,
                    "green_score": entry.get("green_score"),
                    "toxicity_class": entry.get("ild_class"),
                    "recommended": (partition >= 3.0 and not miscible and _toxicity_severity(str(entry.get("ild_class", ""))) <= 2),
                    "overall_score": score,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(
            ["recommended", "partition_coefficient", "overall_score", "boiling_point_K"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def optimize_reaction_medium(
        self,
        reactant_smiles_list: list[str],
        product_smiles: str,
        T_reaction: float = 298.15,
        constraints: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Rank solvents by reactant solubility and product crash-out leverage."""

        constraints = constraints or {}
        min_green = int(constraints.get("min_green_score", 1))
        max_tox = int(constraints.get("max_toxicity_class", 3))
        max_boiling = float(constraints.get("max_boiling_point_K", 550.0))
        min_reactant_solubility = float(constraints.get("min_reactant_solubility_mg_mL", 1.0))
        min_selectivity = float(constraints.get("min_selectivity_ratio", 1.0))

        canonical_reactants = [self.screener._require_smiles(smiles, f"reactant {idx + 1}") for idx, smiles in enumerate(reactant_smiles_list)]
        canonical_product = self.screener._require_smiles(product_smiles, "product")

        rows: list[dict[str, Any]] = []
        for entry in self._metadata_filtered_library(min_green=min_green, max_tox=max_tox, max_boiling=max_boiling):
            product_pred = self.screener._predict_one(canonical_product, str(entry["smiles"]), float(T_reaction))
            product_mg = self.screener._x2_to_mg_ml(
                float(product_pred["x2"]),
                solute_mw=self.screener._molecular_weight(canonical_product),
                solvent_mw=self.screener._molecular_weight(str(entry["smiles"])),
                solvent_density_g_ml=entry.get("density_g_mL"),
            )
            reactant_mg_values: list[float] = []
            reactant_ln_values: list[float] = []
            row: dict[str, Any] = {
                "solvent_name": entry["name"],
                "solvent_smiles": entry["smiles"],
                "solvent_class": entry["solvent_class"],
                "product_smiles": canonical_product,
                "product_ln_x2": float(product_pred["ln_x2"]),
                "product_x2": float(product_pred["x2"]),
                "product_solubility_mg_mL": product_mg,
                "green_score": entry.get("green_score"),
                "toxicity_class": entry.get("ild_class"),
                "boiling_point_K": entry.get("boiling_point_K"),
            }
            for idx, reactant in enumerate(canonical_reactants, start=1):
                pred = self.screener._predict_one(reactant, str(entry["smiles"]), float(T_reaction))
                mg_ml = self.screener._x2_to_mg_ml(
                    float(pred["x2"]),
                    solute_mw=self.screener._molecular_weight(reactant),
                    solvent_mw=self.screener._molecular_weight(str(entry["smiles"])),
                    solvent_density_g_ml=entry.get("density_g_mL"),
                )
                reactant_mg_values.append(mg_ml)
                reactant_ln_values.append(float(pred["ln_x2"]))
                row[f"reactant_{idx}_smiles"] = reactant
                row[f"reactant_{idx}_ln_x2"] = float(pred["ln_x2"])
                row[f"reactant_{idx}_solubility_mg_mL"] = mg_ml

            row["reactant_min_solubility_mg_mL"] = min(reactant_mg_values) if reactant_mg_values else float("nan")
            row["reactant_mean_solubility_mg_mL"] = float(np.mean(reactant_mg_values)) if reactant_mg_values else float("nan")
            row["reactant_product_selectivity"] = row["reactant_min_solubility_mg_mL"] / max(product_mg, 1e-12)
            row["reaction_driving_force"] = (float(np.mean(reactant_ln_values)) - float(product_pred["ln_x2"])) if reactant_ln_values else float("nan")
            row["recommended"] = (
                row["reactant_min_solubility_mg_mL"] >= min_reactant_solubility
                and row["reactant_product_selectivity"] >= min_selectivity
            )
            row["overall_score"] = (
                0.55 * clamp(math.log10(max(row["reactant_product_selectivity"], 1e-12) + 1e-9) / 2.0, 0.0, 1.0)
                + 0.30 * clamp((row["reactant_min_solubility_mg_mL"] - min_reactant_solubility) / max(min_reactant_solubility, 1.0), 0.0, 1.0)
                + 0.15 * clamp((float(entry.get("green_score") or 0) - 1.0) / 9.0, 0.0, 1.0)
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(
            ["recommended", "overall_score", "reactant_product_selectivity", "reactant_min_solubility_mg_mL"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def design_solvent_system(
        self,
        solute_smiles: str,
        target_solubility_range: tuple[float, float],
        T: float = 298.15,
    ) -> pd.DataFrame:
        """Recommend binary solvent mixtures that target a solubility window."""

        low_target, high_target = sorted(float(value) for value in target_solubility_range)
        canonical_solute = self.screener._require_smiles(solute_smiles, "solute")
        pure_df = self.screener.screen(canonical_solute, T=T, top_k=0, filters=None, return_details=True)
        if pure_df.empty:
            return pure_df

        candidate_df = pure_df.head(min(24, len(pure_df))).copy()
        rows: list[dict[str, Any]] = []
        solute_hansen = None
        if {"hansen_sol_d", "hansen_sol_p", "hansen_sol_h"} <= set(candidate_df.columns):
            first = candidate_df.iloc[0]
            if pd.notna(first.get("hansen_sol_d")):
                solute_hansen = (
                    float(first["hansen_sol_d"]),
                    float(first["hansen_sol_p"]),
                    float(first["hansen_sol_h"]),
                )

        for (_, left), (_, right) in combinations(candidate_df.iterrows(), 2):
            left_meta = self._solvent_metadata(str(left["solvent_smiles"]))
            right_meta = self._solvent_metadata(str(right["solvent_smiles"]))
            if not self.screener._solvent_miscibility(left_meta, right_meta):
                continue
            for phi_left in np.linspace(0.1, 0.9, 9):
                phi_right = 1.0 - float(phi_left)
                ln_x2_mix = float(phi_left) * float(left["ln_x2"]) + phi_right * float(right["ln_x2"])
                mix_hansen = self._mix_hansen(left, right, float(phi_left), phi_right)
                ra_mix = None
                if mix_hansen is not None and solute_hansen is not None:
                    ra_mix = self._hansen_distance(solute_hansen, mix_hansen)
                    endpoint_ra = min(
                        value
                        for value in [left.get("hansen_Ra"), right.get("hansen_Ra")]
                        if value is not None and pd.notna(value)
                    ) if any(pd.notna(val) for val in [left.get("hansen_Ra"), right.get("hansen_Ra")]) else None
                    if endpoint_ra is not None:
                        ln_x2_mix += 0.08 * (float(endpoint_ra) - ra_mix)
                x2_mix = clamp(math.exp(min(0.0, ln_x2_mix)), 0.0, 0.999999)
                mg_ml_mix = self._binary_x2_to_mg_ml(
                    x2_mix,
                    solute_smiles=canonical_solute,
                    left_smiles=str(left["solvent_smiles"]),
                    right_smiles=str(right["solvent_smiles"]),
                    phi_left=float(phi_left),
                    phi_right=phi_right,
                    density_left=left_meta.get("density_g_mL"),
                    density_right=right_meta.get("density_g_mL"),
                )
                target_error = 0.0 if low_target <= mg_ml_mix <= high_target else min(abs(mg_ml_mix - low_target), abs(mg_ml_mix - high_target))
                rows.append(
                    {
                        "solvent_a": left["solvent_name"],
                        "solvent_a_smiles": left["solvent_smiles"],
                        "solvent_b": right["solvent_name"],
                        "solvent_b_smiles": right["solvent_smiles"],
                        "phi_a": float(phi_left),
                        "phi_b": phi_right,
                        "estimated_ln_x2": ln_x2_mix,
                        "estimated_x2": x2_mix,
                        "estimated_solubility_mg_mL": mg_ml_mix,
                        "target_error_mg_mL": target_error,
                        "green_score_mix": float(phi_left) * float(left.get("green_score") or 0) + phi_right * float(right.get("green_score") or 0),
                        "boiling_point_mix_K": float(phi_left) * float(left_meta.get("boiling_point_K") or 0) + phi_right * float(right_meta.get("boiling_point_K") or 0),
                        "hansen_Ra_mix": ra_mix,
                        "recommended": low_target <= mg_ml_mix <= high_target,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(
            ["recommended", "target_error_mg_mL", "green_score_mix"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df.attrs["assumptions"] = {
            "mixture_model": "Binary mixtures are approximated by linear interpolation in solvent Hansen space and pure-solvent ln x2 values. This is a heuristic, not a mixture-native thermodynamic model.",
            "solubility_conversion": self.screener.MG_ML_ASSUMPTION,
        }
        return df

    def _metadata_filtered_library(
        self,
        *,
        min_green: int = 1,
        max_tox: int = 3,
        max_boiling: float | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        for entry in self.screener.solvent_library:
            green = entry.get("green_score")
            tox = _toxicity_severity(str(entry.get("ild_class", "")))
            bp = entry.get("boiling_point_K")
            if green is not None and int(green) < min_green:
                continue
            if tox > max_tox:
                continue
            if max_boiling is not None and bp is not None and float(bp) > float(max_boiling):
                continue
            rows.append(entry)
        return rows

    def _solvent_metadata(self, solvent_smiles: str, prediction: dict[str, Any] | None = None) -> dict[str, Any]:
        canonical = self.screener._require_smiles(solvent_smiles, "solvent")
        library_row = next((row for row in self.screener.solvent_library if str(row["smiles"]) == canonical), None)
        if library_row is not None:
            return dict(library_row)
        metadata = {
            "name": canonical,
            "smiles": canonical,
            "solvent_class": "custom",
            "boiling_point_K": None,
            "density_g_mL": 1.0,
            "ild_class": "not classified",
            "green_score": None,
            "cost_relative": "medium",
            "h_bond_donor": False,
            "h_bond_acceptor": False,
            "protic": False,
            "miscible_with_water": False,
            "hansen_d": None,
            "hansen_p": None,
            "hansen_h": None,
        }
        if prediction and prediction.get("hansen_slv") is not None:
            hansen = prediction["hansen_slv"]
            if len(hansen) >= 3:
                metadata["hansen_d"] = hansen[0]
                metadata["hansen_p"] = hansen[1]
                metadata["hansen_h"] = hansen[2]
        return metadata

    def _mix_hansen(
        self,
        left: pd.Series,
        right: pd.Series,
        phi_left: float,
        phi_right: float,
    ) -> tuple[float, float, float] | None:
        coords = []
        for prefix in ("hansen_slv_d", "hansen_slv_p", "hansen_slv_h"):
            left_val = left.get(prefix)
            right_val = right.get(prefix)
            if pd.isna(left_val) or pd.isna(right_val):
                return None
            coords.append(float(phi_left) * float(left_val) + float(phi_right) * float(right_val))
        return coords[0], coords[1], coords[2]

    def _hansen_distance(
        self,
        solute_coords: tuple[float, float, float],
        solvent_coords: tuple[float, float, float],
    ) -> float:
        delta_d = float(solute_coords[0]) - float(solvent_coords[0])
        delta_p = float(solute_coords[1]) - float(solvent_coords[1])
        delta_h = float(solute_coords[2]) - float(solvent_coords[2])
        return math.sqrt(4.0 * delta_d * delta_d + delta_p * delta_p + delta_h * delta_h)

    def _binary_x2_to_mg_ml(
        self,
        x2: float,
        *,
        solute_smiles: str,
        left_smiles: str,
        right_smiles: str,
        phi_left: float,
        phi_right: float,
        density_left: float | None,
        density_right: float | None,
    ) -> float:
        solute_mw = self.screener._molecular_weight(solute_smiles)
        left_mw = self.screener._molecular_weight(left_smiles)
        right_mw = self.screener._molecular_weight(right_smiles)
        if (
            solute_mw is None
            or left_mw is None
            or right_mw is None
            or density_left is None
            or density_right is None
        ):
            return float("nan")
        x2 = clamp(float(x2), 0.0, 0.999999)
        solvent_moles_per_ml = (
            float(phi_left) * float(density_left) / max(float(left_mw), 1e-9)
            + float(phi_right) * float(density_right) / max(float(right_mw), 1e-9)
        )
        solute_moles_per_ml = (x2 / max(1.0 - x2, 1e-9)) * solvent_moles_per_ml
        return solute_moles_per_ml * float(solute_mw) * 1000.0


__all__ = ["ProcessOptimizer"]
