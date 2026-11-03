"""Lightweight application helpers built on top of solubility predictions."""

from __future__ import annotations

import math
from typing import Any


SYNTHESIS_SOLVENT_LIBRARY: dict[str, str] = {
    "Water": "O",
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Isopropanol": "CC(C)O",
    "Acetone": "CC(=O)C",
    "Acetonitrile": "CC#N",
    "Ethyl acetate": "CCOC(=O)C",
    "THF": "C1CCOC1",
    "2-MeTHF": "CC1CCCO1",
    "Toluene": "Cc1ccccc1",
    "DMSO": "CS(C)=O",
    "DMF": "CN(C)C=O",
}

PHARMA_MEDIA_LIBRARY: dict[str, str] = {
    "Water": "O",
    "Ethanol": "CCO",
    "Propylene glycol": "CC(O)CO",
    "Glycerol": "C(C(CO)O)O",
    "PEG surrogate": "OCCOCCOCCO",
    "DMSO": "CS(C)=O",
}

WATER_MOLARITY_MOL_L = 55.5


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def mole_fraction_to_molarity_in_water(x2: float) -> float:
    """Approximate molarity from aqueous mole-fraction solubility."""

    x2 = clamp(x2, 0.0, 0.999999)
    return WATER_MOLARITY_MOL_L * x2 / max(1e-9, 1.0 - x2)


def aqueous_max_supported_dose_mg(
    x2: float,
    mol_weight: float | None,
    *,
    dose_volume_l: float = 0.25,
) -> float | None:
    if mol_weight is None or mol_weight <= 0:
        return None
    molarity = mole_fraction_to_molarity_in_water(x2)
    return molarity * dose_volume_l * float(mol_weight) * 1000.0


def dose_margin(max_supported_dose_mg: float | None, dose_mg: float | None) -> float | None:
    if max_supported_dose_mg is None or dose_mg is None or dose_mg <= 0:
        return None
    return float(max_supported_dose_mg) / float(dose_mg)


def synthesis_window_metrics(hot_ln_x2: float, cold_ln_x2: float) -> dict[str, Any]:
    """Score a solvent for temperature-swing isolation."""

    hot_ln_x2 = float(hot_ln_x2)
    cold_ln_x2 = float(cold_ln_x2)
    delta_ln_x2 = hot_ln_x2 - cold_ln_x2
    swing_ratio = math.exp(clamp(delta_ln_x2, -30.0, 30.0))
    hot_loading = clamp((hot_ln_x2 + 6.0) / 4.0, 0.0, 1.0)
    cold_capture = clamp((-cold_ln_x2 - 2.0) / 4.5, 0.0, 1.0)
    swing_index = clamp(delta_ln_x2 / 3.0, 0.0, 1.0)
    route_score = 100.0 * (0.45 * swing_index + 0.35 * hot_loading + 0.20 * cold_capture)

    if route_score >= 75:
        regime = "Strong temperature-swing candidate"
    elif route_score >= 55:
        regime = "Workable crystallization window"
    elif route_score >= 35:
        regime = "Marginal; expect process tuning"
    else:
        regime = "Poor isolation window"

    return {
        "hot_ln_x2": hot_ln_x2,
        "cold_ln_x2": cold_ln_x2,
        "delta_ln_x2": delta_ln_x2,
        "swing_ratio": swing_ratio,
        "hot_loading_index": hot_loading,
        "cold_capture_index": cold_capture,
        "swing_index": swing_index,
        "route_score": route_score,
        "regime": regime,
    }


def solvent_swap_metrics(source_ln_x2: float, target_ln_x2: float) -> dict[str, Any]:
    """Estimate crash-out pressure when moving from a donor to a target solvent."""

    source_ln_x2 = float(source_ln_x2)
    target_ln_x2 = float(target_ln_x2)
    delta_ln_x2 = source_ln_x2 - target_ln_x2
    crash_ratio = math.exp(clamp(delta_ln_x2, -30.0, 30.0))
    transfer_score = 100.0 * (
        0.60 * clamp(delta_ln_x2 / 4.0, 0.0, 1.0)
        + 0.40 * clamp((-target_ln_x2 - 2.0) / 5.0, 0.0, 1.0)
    )
    if transfer_score >= 70:
        regime = "Strong precipitation trigger"
    elif transfer_score >= 45:
        regime = "Moderate crash-out leverage"
    else:
        regime = "Weak solvent-swap effect"
    return {
        "source_ln_x2": source_ln_x2,
        "target_ln_x2": target_ln_x2,
        "delta_ln_x2": delta_ln_x2,
        "crash_ratio": crash_ratio,
        "transfer_score": transfer_score,
        "regime": regime,
    }


def pharma_capability_matrix(
    *,
    water_margin: float | None,
    has_water_prediction: bool,
    best_cosolvent_uplift: float | None,
) -> list[dict[str, str]]:
    """Explain what can and cannot be inferred from a solubility model."""

    if not has_water_prediction:
        oral_readout = "Need water or aqueous surrogate"
        oral_conf = "low"
    elif water_margin is None:
        oral_readout = "Dose not specified"
        oral_conf = "medium"
    elif water_margin >= 1.0:
        oral_readout = "Water-only dose support looks plausible"
        oral_conf = "medium"
    elif water_margin >= 0.3:
        oral_readout = "Solubility pressure is material"
        oral_conf = "medium"
    else:
        oral_readout = "Dissolution-limited risk is high"
        oral_conf = "medium"

    if best_cosolvent_uplift is None:
        formulation_readout = "No cosolvent screen"
    elif best_cosolvent_uplift >= 10.0:
        formulation_readout = "Strong formulation leverage"
    elif best_cosolvent_uplift >= 3.0:
        formulation_readout = "Moderate formulation leverage"
    else:
        formulation_readout = "Limited cosolvent uplift"

    return [
        {
            "stage": "Solvent ranking / crystallization",
            "support": "strong",
            "reason": "Directly aligned with what the model predicts: equilibrium solubility in explicit solvents and at explicit temperatures.",
        },
        {
            "stage": "Preformulation / solvent screening",
            "support": "strong",
            "reason": formulation_readout,
        },
        {
            "stage": "Oral dose feasibility proxy",
            "support": oral_conf,
            "reason": oral_readout,
        },
        {
            "stage": "Absorption / bioavailability",
            "support": "weak",
            "reason": "Solubility is only one input; permeability, dissolution kinetics, precipitation, metabolism, and transporters are all missing.",
        },
        {
            "stage": "PK exposure (AUC, Cmax, CL, Vd)",
            "support": "weak",
            "reason": "PK requires external clearance, distribution, and absorption models. TGNN-Solv can only supply a solubility-facing prior.",
        },
        {
            "stage": "PD / efficacy",
            "support": "not supported",
            "reason": "Pharmacodynamics is outside the scope of equilibrium solubility prediction.",
        },
    ]


__all__ = [
    "PHARMA_MEDIA_LIBRARY",
    "SYNTHESIS_SOLVENT_LIBRARY",
    "WATER_MOLARITY_MOL_L",
    "aqueous_max_supported_dose_mg",
    "clamp",
    "dose_margin",
    "mole_fraction_to_molarity_in_water",
    "pharma_capability_matrix",
    "solvent_swap_metrics",
    "synthesis_window_metrics",
]
