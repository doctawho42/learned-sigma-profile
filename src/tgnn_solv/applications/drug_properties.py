"""Drug-developability workflows built on top of TGNN-Solv predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
except Exception:  # pragma: no cover - optional runtime guard
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None

from ..baselines.direct_gnn import DirectGNN
from ..config import TGNNSolvConfig
from ..model import TGNNSolv
from .core import PHARMA_MEDIA_LIBRARY, clamp
from .solvent_screening import SolventScreener, _canonicalize_smiles


REFERENCE_DRUG_LIBRARY: list[dict[str, Any]] = [
    {"name": "Paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1", "reference_note": "Common oral analgesic"},
    {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "reference_note": "Lipophilic NSAID"},
    {"name": "Carbamazepine", "smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "reference_note": "Low-solubility CNS drug"},
    {"name": "Caffeine", "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "reference_note": "Polar, well-known xanthine"},
    {"name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "reference_note": "Very polar biguanide"},
    {"name": "Naproxen", "smiles": "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1", "reference_note": "Aromatic carboxylic acid"},
]


@dataclass(frozen=True)
class IonizationEstimate:
    acidic_pka: float | None
    basic_pka: float | None
    classification: str
    source: str
    note: str


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _score_window(value: float, low: float, high: float, *, soft_span: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return clamp(1.0 - (low - value) / max(soft_span, 1e-9), 0.0, 1.0)
    return clamp(1.0 - (value - high) / max(soft_span, 1e-9), 0.0, 1.0)


def _score_threshold(value: float, *, good: float, moderate: float) -> float:
    if value >= good:
        return 1.0
    if value <= moderate:
        return 0.0
    return clamp((value - moderate) / max(good - moderate, 1e-9), 0.0, 1.0)


class DrugPropertyPredictor:
    """BCS-style developability workflow built on top of solvent predictions."""

    def __init__(
        self,
        model: TGNNSolv | DirectGNN,
        cfg: TGNNSolvConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.screener = SolventScreener(model, cfg, self.device)
        self.model_family = self.screener.model_family
        self._water_meta = self.screener._library_by_smiles.get(_canonicalize_smiles("O")) or {
            "name": "Water",
            "smiles": "O",
            "density_g_mL": 0.997,
            "hansen_d": 15.5,
            "hansen_p": 16.0,
            "hansen_h": 42.3,
        }

    def bcs_classify(
        self,
        solute_smiles: str,
        dose_mg: float = 100,
        volume_mL: float = 250,
        T: float = 310.15,
    ) -> dict[str, Any]:
        """Classify a candidate by solubility/permeability proxy."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        intrinsic = self._predict_in_water(canonical, T=T)
        descriptors = self._descriptor_profile(canonical)
        ionization = self._estimate_ionization(canonical)
        solubility_ph = {
            "1.0": self._apply_ph_correction(float(intrinsic.get("solubility_mg_mL", float("nan"))), 1.0, ionization),
            "4.5": self._apply_ph_correction(float(intrinsic.get("solubility_mg_mL", float("nan"))), 4.5, ionization),
            "6.8": self._apply_ph_correction(float(intrinsic.get("solubility_mg_mL", float("nan"))), 6.8, ionization),
        }
        dose_numbers = {
            key: self._dose_number(float(dose_mg), float(volume_mL), value)
            for key, value in solubility_ph.items()
        }
        dose_number = max((value for value in dose_numbers.values() if value is not None), default=None)
        high_solubility = bool(dose_number is not None and dose_number <= 1.0)
        high_permeability = self._high_permeability(descriptors)
        bcs_class = self._assign_bcs_class(high_solubility, high_permeability)
        recommendations = self._bcs_recommendations(bcs_class)
        formulation_challenge = self._formulation_challenge(bcs_class)
        caveats = [
            "Solubility is predicted for intrinsic neutral solubility in water and then adjusted with a heuristic ionization correction.",
            "BCS high-solubility classification is approximated from 250 mL dose-number logic using predicted equilibrium solubility rather than measured biorelevant media.",
            "Permeability is inferred from RDKit proxy descriptors unless external transport or Caco-2 data is supplied.",
        ]
        if ionization.classification == "neutral_or_unknown":
            caveats.append("No confident ionizable motif was detected, so pH-dependent correction was skipped.")
        elif ionization.source != "none":
            caveats.append(f"pH correction used {ionization.source}: {ionization.note}")

        return {
            "bcs_class": bcs_class,
            "solubility_intrinsic_mg_mL": intrinsic.get("solubility_mg_mL"),
            "solubility_intrinsic_x2": intrinsic.get("x2"),
            "solubility_pH1": solubility_ph["1.0"],
            "solubility_pH4_5": solubility_ph["4.5"],
            "solubility_pH6_8": solubility_ph["6.8"],
            "dose_number": dose_number,
            "dose_number_by_ph": dose_numbers,
            "high_solubility": high_solubility,
            "permeability_proxy": {
                "LogP": descriptors.get("MolLogP"),
                "TPSA": descriptors.get("TPSA"),
                "HBA": descriptors.get("HBA"),
                "HBD": descriptors.get("HBD"),
            },
            "high_permeability": high_permeability,
            "formulation_challenge": formulation_challenge,
            "recommendations": recommendations,
            "ionization_estimate": {
                "acidic_pKa": ionization.acidic_pka,
                "basic_pKa": ionization.basic_pka,
                "classification": ionization.classification,
                "source": ionization.source,
                "note": ionization.note,
            },
            "water_prediction": intrinsic,
            "caveats": caveats,
        }

    def developability_score(self, solute_smiles: str, T: float = 310.15) -> dict[str, Any]:
        """Build a composite developability score from solubility-facing signals."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        water_pred = self._predict_in_water(canonical, T=T)
        descriptors = self._descriptor_profile(canonical)
        full_screen = self.screener.screen(canonical, T=T, top_k=len(self.screener.solvent_library), filters=None, return_details=True)
        water_mg = float(water_pred.get("solubility_mg_mL", float("nan")))

        if math.isnan(water_mg):
            solubility_score = 0.25
        elif water_mg >= 10.0:
            solubility_score = 1.0
        elif water_mg >= 1.0:
            solubility_score = 0.65 + 0.35 * clamp(math.log10(water_mg) / 1.0, 0.0, 1.0)
        elif water_mg >= 0.1:
            solubility_score = 0.20 + 0.45 * clamp(math.log10(water_mg / 0.1 + 1e-9), 0.0, 1.0)
        else:
            solubility_score = 0.05

        crystal_score, crystal_details = self._crystal_stability_score(water_pred)
        lipophilicity_score = self._lipophilicity_score(descriptors, water_pred)
        solvent_diversity_count = 0
        solvent_diversity_fraction = 0.0
        if not full_screen.empty and "solubility_mg_mL" in full_screen.columns:
            solvent_diversity_count = int((full_screen["solubility_mg_mL"].fillna(0.0) > 1.0).sum())
            solvent_diversity_fraction = solvent_diversity_count / max(len(full_screen), 1)
        solvent_diversity_score = clamp(solvent_diversity_fraction / 0.35, 0.0, 1.0)

        temp_scan = self.screener._temperature_scan(canonical, "O", T_min=max(273.15, T - 15.0), T_max=min(373.15, T + 15.0), n_points=7)
        if temp_scan.empty:
            sensitivity = 0.0
        else:
            slopes = np.gradient(temp_scan["ln_x2"].to_numpy(), temp_scan["T"].to_numpy())
            sensitivity = float(np.median(np.abs(slopes)))
        temperature_sensitivity_score = clamp(0.55 + 0.45 * clamp(sensitivity / 0.05, 0.0, 1.0), 0.0, 1.0)

        weights = {
            "solubility": 0.32,
            "crystal_stability": 0.20,
            "lipophilicity": 0.16,
            "solvent_diversity": 0.16,
            "temperature_sensitivity": 0.16,
        }
        component_scores = {
            "solubility": solubility_score,
            "crystal_stability": crystal_score,
            "lipophilicity": lipophilicity_score,
            "solvent_diversity": solvent_diversity_score,
            "temperature_sensitivity": temperature_sensitivity_score,
        }
        developability = float(sum(weights[key] * component_scores[key] for key in weights))
        if developability >= 0.72:
            traffic_light = "green"
        elif developability >= 0.45:
            traffic_light = "yellow"
        else:
            traffic_light = "red"

        key_risks: list[str] = []
        recommendations: list[str] = []
        if solubility_score < 0.35:
            key_risks.append("Aqueous intrinsic solubility at 37 C is likely dose-limiting.")
            recommendations.append("Prioritize salt screening, amorphous dispersions, nanocrystals, or lipid-enabled formulations.")
        if crystal_score < 0.40:
            key_risks.append("Predicted crystal stability is high, which raises the dissolution barrier.")
            recommendations.append("Investigate crystal-form engineering, co-crystals, or metastable solid forms.")
        if lipophilicity_score < 0.45:
            key_risks.append("Descriptor profile indicates problematic lipophilicity or hydrogen-bond balance.")
            recommendations.append("Evaluate polar prodrugs, ionizable analogues, or formulation-driven exposure rescue.")
        if solvent_diversity_score < 0.35:
            key_risks.append("Only a narrow solvent set crosses the 1 mg/mL threshold.")
            recommendations.append("Expect limited formulation latitude; use targeted solvent-system design early.")
        if temperature_sensitivity_score < 0.60:
            key_risks.append("Temperature leverage around body temperature is modest.")
            recommendations.append("Do not rely on process heating alone; combine with solid-state or excipient strategies.")
        if not recommendations:
            recommendations.append("The compound looks broadly developable from a solubility-facing perspective; confirm permeability and stability experimentally.")

        return {
            "developability_score": developability,
            "component_scores": component_scores,
            "component_weights": weights,
            "traffic_light": traffic_light,
            "key_risks": key_risks,
            "recommendations": recommendations,
            "water_prediction": water_pred,
            "descriptor_profile": descriptors,
            "crystal_details": crystal_details,
            "solvent_diversity": {
                "count_above_1_mg_mL": solvent_diversity_count,
                "fraction_above_1_mg_mL": solvent_diversity_fraction,
            },
            "temperature_sensitivity": {
                "median_abs_dlnx2_dT": sensitivity,
                "scan": temp_scan.where(temp_scan.notna(), None).to_dict(orient="records"),
            },
            "screening_assumption": self.screener.MG_ML_ASSUMPTION,
        }

    def salt_cocrystal_impact(
        self,
        api_smiles: str,
        counterion_smiles_list: Sequence[str],
        T: float = 310.15,
    ) -> pd.DataFrame:
        """Estimate approximate salt / cocrystal leverage in water."""

        canonical_api = self.screener._require_smiles(api_smiles, "API")
        freeform = self._predict_in_water(canonical_api, T=T)
        rows: list[dict[str, Any]] = []
        for raw_counterion in counterion_smiles_list:
            counterion = _canonicalize_smiles(raw_counterion)
            if not counterion:
                rows.append(
                    {
                        "counterion": raw_counterion,
                        "salt_smiles": None,
                        "x2_freeform": freeform.get("x2"),
                        "x2_salt": None,
                        "solubility_advantage": None,
                        "confidence": 0.0,
                        "caveats": "Invalid counterion SMILES.",
                    }
                )
                continue
            salt_smiles = self._combine_components(canonical_api, counterion)
            if not salt_smiles:
                rows.append(
                    {
                        "counterion": counterion,
                        "salt_smiles": None,
                        "x2_freeform": freeform.get("x2"),
                        "x2_salt": None,
                        "solubility_advantage": None,
                        "confidence": 0.1,
                        "caveats": "Failed to build a disconnected API.counterion surrogate for approximate salt prediction.",
                    }
                )
                continue
            try:
                salt_pred = self._predict_in_water(salt_smiles, T=T)
                advantage = float(salt_pred["x2"]) / max(float(freeform["x2"]), 1e-12)
                rows.append(
                    {
                        "counterion": counterion,
                        "salt_smiles": salt_smiles,
                        "x2_freeform": freeform.get("x2"),
                        "x2_salt": salt_pred.get("x2"),
                        "solubility_advantage": advantage,
                        "confidence": 0.30,
                        "caveats": "Approximate only: TGNN-Solv is trained on neutral molecules, so disconnected salt/cocrystal surrogates should be interpreted qualitatively.",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "counterion": counterion,
                        "salt_smiles": salt_smiles,
                        "x2_freeform": freeform.get("x2"),
                        "x2_salt": None,
                        "solubility_advantage": None,
                        "confidence": 0.10,
                        "caveats": f"Approximate salt surrogate could not be scored: {type(exc).__name__}: {exc}",
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["solubility_advantage", "confidence"], ascending=[False, False], na_position="last").reset_index(drop=True)

    def pharma_media_profile(self, solute_smiles: str, T: float = 310.15) -> pd.DataFrame:
        """Score a small set of pharma-relevant solvent surrogates."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        solute_mw = self.screener._molecular_weight(canonical)
        water_x2 = None
        rows: list[dict[str, Any]] = []
        for label, smiles in PHARMA_MEDIA_LIBRARY.items():
            entry = self.screener._library_by_smiles.get(_canonicalize_smiles(smiles)) or {
                "name": label,
                "smiles": _canonicalize_smiles(smiles),
                "solvent_class": "pharma_medium",
                "density_g_mL": 1.0,
                "boiling_point_K": None,
                "ild_class": "not classified",
                "green_score": None,
                "cost_relative": "medium",
                "miscible_with_water": True,
                "h_bond_donor": False,
                "h_bond_acceptor": False,
                "protic": False,
                "hansen_d": None,
                "hansen_p": None,
                "hansen_h": None,
            }
            pred = self.screener._predict_one(canonical, str(entry["smiles"]), float(T))
            row = self.screener._prediction_to_row(canonical, solute_mw, entry, pred, temperature=T)
            if row is None:
                continue
            row["medium"] = label
            rows.append(row)
            if _canonicalize_smiles(smiles) == _canonicalize_smiles("O"):
                water_x2 = float(row["x2"])
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if water_x2 is not None and water_x2 > 0:
            df["fold_vs_water"] = df["x2"] / water_x2
        else:
            df["fold_vs_water"] = np.nan
        return df.sort_values("solubility_mg_mL", ascending=False).reset_index(drop=True)

    def compare_with_reference_drugs(self, solute_smiles: str, top_k: int = 5) -> pd.DataFrame:
        """Compare the candidate with a small built-in oral-drug reference panel."""

        canonical = self.screener._require_smiles(solute_smiles, "solute")
        candidate_desc = self._descriptor_profile(canonical)
        candidate_fp = self._morgan_fingerprint(canonical)
        rows: list[dict[str, Any]] = []
        for entry in REFERENCE_DRUG_LIBRARY:
            ref_smiles = _canonicalize_smiles(str(entry["smiles"]))
            ref_desc = self._descriptor_profile(ref_smiles)
            similarity = self._tanimoto(candidate_fp, self._morgan_fingerprint(ref_smiles))
            rows.append(
                {
                    "reference_name": entry["name"],
                    "reference_smiles": ref_smiles,
                    "reference_note": entry["reference_note"],
                    "tanimoto": similarity,
                    "delta_logp": self._delta(candidate_desc.get("MolLogP"), ref_desc.get("MolLogP")),
                    "delta_tpsa": self._delta(candidate_desc.get("TPSA"), ref_desc.get("TPSA")),
                    "delta_molwt": self._delta(candidate_desc.get("MolWt"), ref_desc.get("MolWt")),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("tanimoto", ascending=False).head(max(1, int(top_k))).reset_index(drop=True)

    def _predict_in_water(self, solute_smiles: str, *, T: float) -> dict[str, Any]:
        pred = self.screener._predict_one(solute_smiles, str(self._water_meta["smiles"]), float(T))
        solute_mw = self.screener._molecular_weight(solute_smiles)
        solvent_mw = self.screener._molecular_weight(str(self._water_meta["smiles"]))
        mg_ml = self.screener._x2_to_mg_ml(
            float(pred["x2"]),
            solute_mw=solute_mw,
            solvent_mw=solvent_mw,
            solvent_density_g_ml=self._water_meta.get("density_g_mL"),
        )
        payload = {
            "solute": solute_smiles,
            "solvent": str(self._water_meta["smiles"]),
            "solvent_name": "Water",
            "T": float(T),
            "solubility_mg_mL": mg_ml,
        }
        payload.update(pred)
        return payload

    def _descriptor_profile(self, smiles: str) -> dict[str, float]:
        if Chem is None or Descriptors is None:
            return {}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "MolWt": float(Descriptors.MolWt(mol)),
            "MolLogP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "HBA": float(Descriptors.NumHAcceptors(mol)),
            "HBD": float(Descriptors.NumHDonors(mol)),
            "RotBonds": float(Descriptors.NumRotatableBonds(mol)),
        }

    def _estimate_ionization(self, smiles: str) -> IonizationEstimate:
        if Chem is None:
            return IonizationEstimate(None, None, "neutral_or_unknown", "none", "RDKit unavailable.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return IonizationEstimate(None, None, "neutral_or_unknown", "none", "SMILES could not be parsed.")

        acid_rules = [
            ("carboxylic acid", "[CX3](=O)[OX2H1]", 4.5),
            ("sulfonic acid", "S(=O)(=O)[OX2H1]", -0.5),
            ("phosphonic acid", "P(=O)(O)O", 2.0),
            ("tetrazole", "[nH]1nnnc1", 4.8),
            ("phenol", "c[OX2H1]", 10.0),
        ]
        base_rules = [
            ("aliphatic amine", "[NX3;H2,H1,H0;!$(NC=O);!$(N[a])]", 9.5),
            ("aniline-like amine", "[NX3;H2,H1,H0]c", 5.0),
            ("pyridine-like nitrogen", "[nX2;r5,r6;+0]", 5.2),
            ("imidazole-like nitrogen", "[nH]1ccnc1", 6.8),
        ]

        acidic_hits = [pka for _, smarts, pka in acid_rules if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
        basic_hits = [pka for _, smarts, pka in base_rules if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
        acidic_pka = min(acidic_hits) if acidic_hits else None
        basic_pka = max(basic_hits) if basic_hits else None
        if acidic_pka is not None and basic_pka is not None:
            return IonizationEstimate(acidic_pka, basic_pka, "ampholyte", "rdkit_substructure_heuristic", "Both acidic and basic motifs were detected; pH correction uses a simplified combined factor.")
        if acidic_pka is not None:
            return IonizationEstimate(acidic_pka, None, "acid", "rdkit_substructure_heuristic", "Acidic motif detected from SMARTS-based heuristic.")
        if basic_pka is not None:
            return IonizationEstimate(None, basic_pka, "base", "rdkit_substructure_heuristic", "Basic motif detected from SMARTS-based heuristic.")
        return IonizationEstimate(None, None, "neutral_or_unknown", "none", "No obvious ionizable motif was detected.")

    def _apply_ph_correction(self, intrinsic_mg_ml: float, pH: float, estimate: IonizationEstimate) -> float | None:
        if not math.isfinite(intrinsic_mg_ml):
            return None
        factor = 1.0
        if estimate.acidic_pka is not None:
            factor += 10.0 ** clamp(float(pH) - float(estimate.acidic_pka), -8.0, 8.0)
        if estimate.basic_pka is not None:
            factor += 10.0 ** clamp(float(estimate.basic_pka) - float(pH), -8.0, 8.0)
        return float(intrinsic_mg_ml) * factor

    def _dose_number(self, dose_mg: float, volume_mL: float, solubility_mg_mL: float | None) -> float | None:
        if solubility_mg_mL is None or not math.isfinite(float(solubility_mg_mL)) or solubility_mg_mL <= 0:
            return None
        return float(dose_mg) / max(float(volume_mL) * float(solubility_mg_mL), 1e-12)

    def _high_permeability(self, descriptors: dict[str, float]) -> bool:
        logp = descriptors.get("MolLogP")
        tpsa = descriptors.get("TPSA")
        if logp is None or tpsa is None:
            return False
        return bool(float(tpsa) <= 140.0 and -1.0 <= float(logp) <= 5.5 and (float(logp) > 0.0 or float(tpsa) <= 90.0))

    def _assign_bcs_class(self, high_solubility: bool, high_permeability: bool) -> int:
        if high_solubility and high_permeability:
            return 1
        if (not high_solubility) and high_permeability:
            return 2
        if high_solubility and (not high_permeability):
            return 3
        return 4

    def _bcs_recommendations(self, bcs_class: int) -> list[str]:
        if bcs_class == 1:
            return [
                "Simple immediate-release formulation is the default starting point.",
                "Focus development effort on stability, manufacturability, and taste rather than enabling solubility technologies.",
            ]
        if bcs_class == 2:
            return [
                "Dissolution-rate limited profile: prioritize amorphous solid dispersions or nanocrystals.",
                "Run salt screening, lipid formulation scouting, and cyclodextrin / cosolvent studies early.",
            ]
        if bcs_class == 3:
            return [
                "Permeability-limited profile: investigate permeation enhancers, transporter effects, or prodrug options.",
                "Aqueous solubility looks adequate, so formulation should avoid unnecessarily slowing release.",
            ]
        return [
            "Class 4 profile: combine enabling formulations with chemistry or salt / prodrug redesign options.",
            "Expect both dissolution and permeability pressure; do not rely on one-axis fixes alone.",
        ]

    def _formulation_challenge(self, bcs_class: int) -> str:
        return {
            1: "Low formulation risk from the classic BCS perspective.",
            2: "Solubility and dissolution are the dominant oral-formulation bottlenecks.",
            3: "Permeability or intestinal transport is more likely to dominate than equilibrium solubility.",
            4: "Both solubility and permeability are problematic; this is the highest-risk developability regime.",
        }[int(bcs_class)]

    def _crystal_stability_score(self, water_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        tm = water_pred.get("T_m")
        dh = water_pred.get("dH_fus")
        if self.model_family != "tgnn_solv" or tm is None or dh is None:
            return 0.50, {
                "T_m_pred": tm,
                "dH_fus_pred": dh,
                "note": "Crystal-stability score falls back to a neutral prior because DirectGNN does not expose fusion parameters.",
            }
        tm = float(tm)
        dh = float(dh)
        score = 1.0 - _sigmoid((tm - 400.0) / 50.0) * _sigmoid((dh - 30000.0) / 5000.0)
        return clamp(score, 0.0, 1.0), {
            "T_m_pred": tm,
            "dH_fus_pred": dh,
            "note": "High T_m and high dH_fus imply a stiff crystal lattice and therefore a lower crystal-stability score.",
        }

    def _lipophilicity_score(self, descriptors: dict[str, float], water_pred: dict[str, Any]) -> float:
        logp = descriptors.get("MolLogP")
        if logp is None:
            return 0.5
        logp_score = _score_window(float(logp), -0.5, 4.5, soft_span=2.0)
        hansen_bonus = 0.5
        if self.model_family == "tgnn_solv":
            hansen = water_pred.get("hansen_sol") or [None, None, None]
            if len(hansen) >= 3 and hansen[2] is not None:
                hansen_bonus = _score_window(float(hansen[2]), 8.0, 22.0, soft_span=8.0)
        return 0.65 * logp_score + 0.35 * hansen_bonus

    def _combine_components(self, left_smiles: str, right_smiles: str) -> str | None:
        if Chem is None:
            return None
        mol_left = Chem.MolFromSmiles(left_smiles)
        mol_right = Chem.MolFromSmiles(right_smiles)
        if mol_left is None or mol_right is None:
            return None
        combined = Chem.CombineMols(mol_left, mol_right)
        return Chem.MolToSmiles(combined, canonical=True) if combined is not None else None

    def _morgan_fingerprint(self, smiles: str) -> Any:
        if Chem is None or AllChem is None:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    def _tanimoto(self, left_fp: Any, right_fp: Any) -> float | None:
        if DataStructs is None or left_fp is None or right_fp is None:
            return None
        return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))

    def _delta(self, left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return float(left) - float(right)


__all__ = ["DrugPropertyPredictor", "REFERENCE_DRUG_LIBRARY"]
