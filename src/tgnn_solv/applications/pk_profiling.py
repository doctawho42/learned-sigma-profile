"""PK-relevant solubility profiling built on top of TGNN-Solv predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..baselines.direct_gnn import DirectGNN
from ..config import TGNNSolvConfig
from ..model import TGNNSolv
from .core import clamp
from .drug_properties import DrugPropertyPredictor


@dataclass(frozen=True)
class CompartmentSpec:
    name: str
    label: str
    pH: float
    volume_mL: float
    residence_time_min: tuple[int, int]
    absorption_weight: float
    medium_mode: str


GI_COMPARTMENTS: tuple[CompartmentSpec, ...] = (
    CompartmentSpec("stomach_fasted", "Stomach (fasted)", 1.5, 250.0, (15, 60), 0.0, "water"),
    CompartmentSpec("stomach_fed", "Stomach (fed)", 4.5, 500.0, (30, 180), 0.0, "fed_gastric"),
    CompartmentSpec("duodenum", "Duodenum", 5.8, 120.0, (15, 45), 0.25, "fassif"),
    CompartmentSpec("jejunum_ileum", "Jejunum / Ileum", 6.5, 150.0, (120, 240), 0.55, "water"),
    CompartmentSpec("colon", "Colon", 6.8, 60.0, (240, 720), 0.20, "water"),
)


BIORELEVANT_MEDIA_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "FaSSGF", "label": "FaSSGF", "pH": 1.6, "mode": "water", "note": "Fasted-state simulated gastric fluid"},
    {"name": "FeSSGF", "label": "FeSSGF", "pH": 5.0, "mode": "fed_gastric", "note": "Fed-state simulated gastric fluid"},
    {"name": "FaSSIF", "label": "FaSSIF", "pH": 6.5, "mode": "fassif", "note": "Fasted-state simulated intestinal fluid"},
    {"name": "FeSSIF", "label": "FeSSIF", "pH": 5.8, "mode": "fessif", "note": "Fed-state simulated intestinal fluid"},
)


IV_VEHICLES: tuple[dict[str, Any], ...] = (
    {
        "name": "Water for injection",
        "vehicle_type": "aqueous",
        "mode": "solvent",
        "smiles": "O",
        "density_g_mL": 0.997,
        "max_fraction_vv": 1.00,
        "osmolality_base": "low",
        "note": "Primary IV vehicle.",
        "research_only": False,
    },
    {
        "name": "Ethanol",
        "vehicle_type": "cosolvent",
        "mode": "solvent",
        "smiles": "CCO",
        "density_g_mL": 0.789,
        "max_fraction_vv": 0.10,
        "osmolality_base": "medium",
        "note": "Often limited to low v/v fractions in parenteral products.",
        "research_only": False,
    },
    {
        "name": "PEG 400 surrogate",
        "vehicle_type": "cosolvent",
        "mode": "solvent",
        "smiles": "OCCOCCOCCOCCO",
        "density_g_mL": 1.125,
        "max_fraction_vv": 0.60,
        "osmolality_base": "high",
        "note": "Polyether surrogate for PEG-rich IV formulations.",
        "research_only": False,
    },
    {
        "name": "Propylene glycol",
        "vehicle_type": "cosolvent",
        "mode": "solvent",
        "smiles": "CC(O)CO",
        "density_g_mL": 1.036,
        "max_fraction_vv": 0.60,
        "osmolality_base": "high",
        "note": "Useful co-solvent, but osmolality and hemolysis limits matter.",
        "research_only": False,
    },
    {
        "name": "DMSO",
        "vehicle_type": "cosolvent",
        "mode": "solvent",
        "smiles": "CS(C)=O",
        "density_g_mL": 1.095,
        "max_fraction_vv": 0.10,
        "osmolality_base": "medium",
        "note": "Research-only co-solvent in most settings.",
        "research_only": True,
    },
    {
        "name": "Cremophor EL surrogate",
        "vehicle_type": "surfactant",
        "mode": "water_factor",
        "factor_kind": "surfactant",
        "max_fraction_vv": 0.20,
        "osmolality_base": "medium",
        "note": "Approximate surfactant-assisted solubilization.",
        "research_only": False,
    },
    {
        "name": "Polysorbate 80 surrogate",
        "vehicle_type": "surfactant",
        "mode": "water_factor",
        "factor_kind": "surfactant",
        "max_fraction_vv": 0.20,
        "osmolality_base": "medium",
        "note": "Approximate nonionic surfactant effect.",
        "research_only": False,
    },
    {
        "name": "Cyclodextrin solution",
        "vehicle_type": "complexing_agent",
        "mode": "water_factor",
        "factor_kind": "cyclodextrin",
        "max_fraction_vv": 0.15,
        "osmolality_base": "medium",
        "note": "Approximate inclusion-complex effect in water.",
        "research_only": False,
    },
)


TOPICAL_VEHICLES: tuple[dict[str, Any], ...] = (
    {
        "name": "Water-based gel",
        "vehicle_type": "aqueous_gel",
        "mode": "solvent",
        "smiles": "O",
        "density_g_mL": 0.997,
        "note": "Hydrophilic gel base.",
    },
    {
        "name": "Ethanol solution",
        "vehicle_type": "alcohol",
        "mode": "solvent",
        "smiles": "CCO",
        "density_g_mL": 0.789,
        "note": "Volatile alcohol solution.",
    },
    {
        "name": "IPA solution",
        "vehicle_type": "alcohol",
        "mode": "solvent",
        "smiles": "CC(C)O",
        "density_g_mL": 0.786,
        "note": "Isopropanol vehicle.",
    },
    {
        "name": "PEG ointment surrogate",
        "vehicle_type": "polyether",
        "mode": "solvent",
        "smiles": "OCCOCCOCCOCCO",
        "density_g_mL": 1.125,
        "note": "PEG-rich ointment surrogate.",
    },
    {
        "name": "Mineral oil surrogate",
        "vehicle_type": "hydrocarbon",
        "mode": "solvent",
        "smiles": "CCCCCCCCCCCCCCCC",
        "density_g_mL": 0.770,
        "note": "Long-alkane mineral oil surrogate.",
    },
    {
        "name": "Petrolatum surrogate",
        "vehicle_type": "hydrocarbon",
        "mode": "solvent",
        "smiles": "CCCCCCCCCCCCCCCCCC",
        "density_g_mL": 0.800,
        "note": "Heavy hydrocarbon ointment surrogate.",
    },
    {
        "name": "Silicone-like isododecane surrogate",
        "vehicle_type": "silicone_like",
        "mode": "solvent",
        "smiles": "CC(C)CCCC(C)(C)C",
        "density_g_mL": 0.750,
        "note": "Hydrophobic volatile fluid surrogate.",
    },
    {
        "name": "Propylene glycol",
        "vehicle_type": "polyol",
        "mode": "solvent",
        "smiles": "CC(O)CO",
        "density_g_mL": 1.036,
        "note": "Humectant and co-solvent.",
    },
    {
        "name": "DMSO",
        "vehicle_type": "penetration_enhancer",
        "mode": "solvent",
        "smiles": "CS(C)=O",
        "density_g_mL": 1.095,
        "note": "Strong penetration enhancer; often irritation-limited.",
    },
    {
        "name": "Oleic acid",
        "vehicle_type": "penetration_enhancer",
        "mode": "solvent",
        "smiles": "CCCCCCCCC=CCCCCCCCC(=O)O",
        "density_g_mL": 0.895,
        "note": "Fatty-acid enhancer surrogate.",
    },
)


class PKSolubilityProfiler:
    """Profile oral/PK-relevant solubility behavior from model predictions."""

    def __init__(
        self,
        model: TGNNSolv | DirectGNN,
        cfg: TGNNSolvConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.drug = DrugPropertyPredictor(model, cfg, self.device)
        self.screener = self.drug.screener
        self.model_family = self.drug.model_family

    def gi_tract_profile(self, solute_smiles: str, dose_mg: float = 100) -> dict[str, Any]:
        """Estimate dissolution pressure along the GI tract."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        descriptor_profile = self.drug._descriptor_profile(canonical)
        ionization = self.drug._estimate_ionization(canonical)
        bcs = self.drug.bcs_classify(canonical, dose_mg=float(dose_mg), volume_mL=250.0, T=310.15)

        compartments: list[dict[str, Any]] = []
        max_absorbable_dose = 0.0
        weighted_absorption = 0.0
        rate_limiting_step = "none"

        for index, spec in enumerate(GI_COMPARTMENTS):
            medium_effect = self._gi_medium_factor(spec.medium_mode, descriptor_profile, bcs)
            solubility = self._corrected_aqueous_solubility(
                canonical,
                pH=spec.pH,
                T=310.15,
                ionization=ionization,
                enhancement_factor=medium_effect,
            )
            dose_number = self.drug._dose_number(float(dose_mg), float(spec.volume_mL), solubility)
            dissolved_fraction = 0.0
            if solubility is not None and math.isfinite(float(solubility)):
                dissolved_fraction = clamp(float(solubility) * float(spec.volume_mL) / max(float(dose_mg), 1e-9), 0.0, 1.0)
                max_absorbable_dose = max(max_absorbable_dose, float(solubility) * float(spec.volume_mL))
            dissolution_limited = bool(dose_number is None or dose_number > 1.0 or dissolved_fraction < 0.9)
            compartments.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "index": index,
                    "pH": spec.pH,
                    "temperature_K": 310.15,
                    "volume_mL": spec.volume_mL,
                    "residence_time_min": list(spec.residence_time_min),
                    "solubility_mg_mL": solubility,
                    "dissolved_fraction": dissolved_fraction,
                    "dose_number": dose_number,
                    "dissolution_limited": dissolution_limited,
                    "medium_mode": spec.medium_mode,
                    "medium_effect_factor": medium_effect,
                }
            )
            if spec.absorption_weight > 0:
                weighted_absorption += dissolved_fraction * float(spec.absorption_weight)
                if rate_limiting_step == "none" and dissolved_fraction < 0.5:
                    rate_limiting_step = spec.name

        permeability_factor = self._permeability_factor(descriptor_profile)
        f_abs_estimate = clamp(weighted_absorption * permeability_factor, 0.0, 1.0)
        if rate_limiting_step == "none" and not bcs.get("high_permeability"):
            rate_limiting_step = "permeability"
        elif rate_limiting_step == "none":
            rate_limiting_step = "no severe dissolution bottleneck detected"

        return {
            "compartments": compartments,
            "max_absorbable_dose": max_absorbable_dose,
            "f_abs_estimate": f_abs_estimate,
            "rate_limiting_step": rate_limiting_step,
            "permeability_factor": permeability_factor,
            "descriptor_profile": descriptor_profile,
            "bcs_context": {
                "bcs_class": bcs.get("bcs_class"),
                "high_permeability": bcs.get("high_permeability"),
                "high_solubility": bcs.get("high_solubility"),
            },
        }

    def biorelevant_media_screen(self, solute_smiles: str) -> dict[str, Any]:
        """Predict solubility in biorelevant media approximations."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        descriptor_profile = self.drug._descriptor_profile(canonical)
        ionization = self.drug._estimate_ionization(canonical)
        bcs = self.drug.bcs_classify(canonical, dose_mg=100.0, volume_mL=250.0, T=310.15)

        media_rows: list[dict[str, Any]] = []
        for spec in BIORELEVANT_MEDIA_SPECS:
            factor = self._gi_medium_factor(str(spec["mode"]), descriptor_profile, bcs)
            solubility = self._corrected_aqueous_solubility(
                canonical,
                pH=float(spec["pH"]),
                T=310.15,
                ionization=ionization,
                enhancement_factor=factor,
            )
            media_rows.append(
                {
                    "medium": spec["name"],
                    "label": spec["label"],
                    "pH": float(spec["pH"]),
                    "solubility_mg_mL": solubility,
                    "enhancement_factor": factor,
                    "note": spec["note"],
                }
            )

        media = {row["medium"]: row for row in media_rows}
        fassif = media.get("FaSSIF", {}).get("solubility_mg_mL")
        fessif = media.get("FeSSIF", {}).get("solubility_mg_mL")
        food_effect_ratio = None
        if fassif is not None and fessif is not None and float(fassif) > 0:
            food_effect_ratio = float(fessif) / float(fassif)

        if food_effect_ratio is None:
            food_effect = "undetermined"
            recommendation = "Need both FaSSIF and FeSSIF estimates to interpret food effect."
        elif food_effect_ratio >= 2.0:
            food_effect = "positive food effect"
            recommendation = "Take with food or evaluate fed-state exposure carefully."
        elif food_effect_ratio <= 0.67:
            food_effect = "negative food effect"
            recommendation = "Take on an empty stomach may reduce variability."
        else:
            food_effect = "minimal food effect"
            recommendation = "No strong food-effect flag from the current solubility proxy."

        return {
            "media": media_rows,
            "food_effect_ratio": food_effect_ratio,
            "food_effect_prediction": food_effect,
            "administration_recommendation": recommendation,
            "bcs_context": {
                "bcs_class": bcs.get("bcs_class"),
                "high_solubility": bcs.get("high_solubility"),
            },
        }

    def iv_formulation_screening(self, solute_smiles: str) -> pd.DataFrame:
        """Rank IV-relevant co-solvents and excipient systems."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        descriptor_profile = self.drug._descriptor_profile(canonical)
        water_25 = self.drug._predict_in_water(canonical, T=298.15)
        water_37 = self.drug._predict_in_water(canonical, T=310.15)

        rows: list[dict[str, Any]] = []
        for entry in IV_VEHICLES:
            pure_25 = self._vehicle_solubility(canonical, entry, T=298.15, descriptor_profile=descriptor_profile, water_reference=water_25)
            pure_37 = self._vehicle_solubility(canonical, entry, T=310.15, descriptor_profile=descriptor_profile, water_reference=water_37)
            max_fraction = float(entry["max_fraction_vv"])
            est_conc_25 = self._blend_capacity(float(water_25.get("solubility_mg_mL") or 0.0), pure_25["solubility_mg_mL"], max_fraction)
            est_conc_37 = self._blend_capacity(float(water_37.get("solubility_mg_mL") or 0.0), pure_37["solubility_mg_mL"], max_fraction)
            osmolality_concern = self._osmolality_flag(max_fraction, str(entry.get("osmolality_base", "medium")))
            recommended = bool(
                est_conc_37 >= 1.0
                and osmolality_concern != "high"
                and (not bool(entry.get("research_only")))
            )
            rows.append(
                {
                    "vehicle_name": entry["name"],
                    "vehicle_type": entry["vehicle_type"],
                    "max_fraction_vv": max_fraction,
                    "solubility_25C_mg_mL": pure_25["solubility_mg_mL"],
                    "solubility_37C_mg_mL": pure_37["solubility_mg_mL"],
                    "iv_estimated_concentration_25C_mg_mL": est_conc_25,
                    "iv_estimated_concentration_37C_mg_mL": est_conc_37,
                    "osmolality_concern": osmolality_concern,
                    "research_only": bool(entry.get("research_only", False)),
                    "recommended": recommended,
                    "note": entry.get("note"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(
            ["recommended", "iv_estimated_concentration_37C_mg_mL", "max_fraction_vv"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    def topical_vehicle_screening(self, solute_smiles: str) -> pd.DataFrame:
        """Rank topical vehicles by approximate thermodynamic activity."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        descriptor_profile = self.drug._descriptor_profile(canonical)
        water_32 = self.drug._predict_in_water(canonical, T=305.15)
        rows: list[dict[str, Any]] = []
        for entry in TOPICAL_VEHICLES:
            pred = self._vehicle_solubility(canonical, entry, T=305.15, descriptor_profile=descriptor_profile, water_reference=water_32)
            x2 = pred["x2"]
            gamma2 = pred["gamma_2"]
            thermodynamic_activity = x2 * gamma2
            near_saturation_score = 1.0 - clamp(abs(math.log10(max(x2, 1e-9)) + 1.0) / 3.0, 0.0, 1.0)
            permeation_potential = 0.70 * clamp(thermodynamic_activity / max(1e-9, 0.25), 0.0, 1.0) + 0.30 * near_saturation_score
            recommended = bool(pred["solubility_mg_mL"] >= 0.1 and permeation_potential >= 0.55)
            rows.append(
                {
                    "vehicle_name": entry["name"],
                    "vehicle_type": entry["vehicle_type"],
                    "solubility_mg_mL": pred["solubility_mg_mL"],
                    "x2": x2,
                    "gamma_2": gamma2,
                    "thermodynamic_activity": thermodynamic_activity,
                    "near_saturation_score": near_saturation_score,
                    "permeation_potential": permeation_potential,
                    "recommended": recommended,
                    "note": entry.get("note"),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(
            ["recommended", "permeation_potential", "thermodynamic_activity"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _corrected_aqueous_solubility(
        self,
        solute_smiles: str,
        *,
        pH: float,
        T: float,
        ionization: Any | None = None,
        enhancement_factor: float = 1.0,
    ) -> float | None:
        ionization = ionization or self.drug._estimate_ionization(solute_smiles)
        water_pred = self.drug._predict_in_water(solute_smiles, T=T)
        intrinsic = water_pred.get("solubility_mg_mL")
        corrected = self.drug._apply_ph_correction(float(intrinsic), pH, ionization) if intrinsic is not None else None
        if corrected is None:
            return None
        return float(corrected) * float(enhancement_factor)

    def _gi_medium_factor(
        self,
        mode: str,
        descriptor_profile: dict[str, float],
        bcs_context: dict[str, Any],
    ) -> float:
        logp = float(descriptor_profile.get("MolLogP", 0.0) or 0.0)
        tpsa = float(descriptor_profile.get("TPSA", 120.0) or 120.0)
        low_intrinsic = 1.0 if not bcs_context.get("high_solubility", False) else 0.0
        lipophilic = clamp((logp - 1.0) / 4.0, 0.0, 1.0)
        polar_penalty = clamp((tpsa - 90.0) / 90.0, 0.0, 1.0)
        if mode == "water":
            return 1.0
        if mode == "fed_gastric":
            return clamp(1.0 + 0.25 * lipophilic + 0.20 * low_intrinsic, 1.0, 1.8)
        if mode == "fassif":
            return clamp(1.5 + 3.0 * lipophilic + 3.5 * low_intrinsic - 0.4 * polar_penalty, 1.0, 10.0)
        if mode == "fessif":
            return clamp(3.0 + 8.0 * lipophilic + 10.0 * low_intrinsic - 0.8 * polar_penalty, 1.0, 50.0)
        return 1.0

    def _vehicle_solubility(
        self,
        solute_smiles: str,
        vehicle: dict[str, Any],
        *,
        T: float,
        descriptor_profile: dict[str, float],
        water_reference: dict[str, Any],
    ) -> dict[str, float]:
        mode = str(vehicle.get("mode", "solvent"))
        if mode == "water_factor":
            kind = str(vehicle.get("factor_kind", "surfactant"))
            factor = self._water_factor_for_vehicle(kind, descriptor_profile)
            solubility = float(water_reference.get("solubility_mg_mL") or 0.0) * factor
            x2 = float(water_reference.get("x2") or 0.0) * factor
            gamma2 = float(water_reference.get("gamma_2") or 1.0)
            return {"solubility_mg_mL": solubility, "x2": x2, "gamma_2": gamma2}

        solvent_smiles = str(vehicle.get("smiles", "O"))
        pred = self.screener._predict_one(solute_smiles, solvent_smiles, float(T))
        solute_mw = self.screener._molecular_weight(solute_smiles)
        solvent_mw = self.screener._molecular_weight(solvent_smiles)
        density = vehicle.get("density_g_mL", 1.0)
        mg_ml = self.screener._x2_to_mg_ml(
            float(pred["x2"]),
            solute_mw=solute_mw,
            solvent_mw=solvent_mw,
            solvent_density_g_ml=density,
        )
        return {
            "solubility_mg_mL": mg_ml,
            "x2": float(pred["x2"]),
            "gamma_2": float(pred.get("gamma_2") or 1.0),
        }

    def _water_factor_for_vehicle(self, kind: str, descriptor_profile: dict[str, float]) -> float:
        logp = float(descriptor_profile.get("MolLogP", 0.0) or 0.0)
        lipophilic = clamp((logp - 1.0) / 4.5, 0.0, 1.0)
        tpsa = float(descriptor_profile.get("TPSA", 120.0) or 120.0)
        polar_bonus = clamp((120.0 - tpsa) / 120.0, 0.0, 1.0)
        if kind == "surfactant":
            return clamp(2.0 + 4.0 * lipophilic + 1.0 * polar_bonus, 2.0, 12.0)
        if kind == "cyclodextrin":
            return clamp(1.5 + 2.0 * lipophilic + 1.5 * clamp(float(descriptor_profile.get("HBA", 0.0)) / 6.0, 0.0, 1.0), 1.5, 8.0)
        return 1.0

    def _blend_capacity(self, water_mg_ml: float, cosolvent_mg_ml: float, max_fraction: float) -> float:
        return (1.0 - float(max_fraction)) * float(water_mg_ml) + float(max_fraction) * float(cosolvent_mg_ml)

    def _osmolality_flag(self, max_fraction: float, base_flag: str) -> str:
        if base_flag == "high":
            return "high"
        if max_fraction >= 0.40:
            return "high"
        if base_flag == "medium" or max_fraction >= 0.15:
            return "medium"
        return "low"

    def _permeability_factor(self, descriptor_profile: dict[str, float]) -> float:
        if self.drug._high_permeability(descriptor_profile):
            return 0.90
        tpsa = float(descriptor_profile.get("TPSA", 140.0) or 140.0)
        if tpsa <= 170.0:
            return 0.50
        return 0.25


__all__ = [
    "BIORELEVANT_MEDIA_SPECS",
    "GI_COMPARTMENTS",
    "IV_VEHICLES",
    "PKSolubilityProfiler",
    "TOPICAL_VEHICLES",
]
