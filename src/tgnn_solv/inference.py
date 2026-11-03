"""
Inference utilities for TGNN-Solv.

Functions:
  predict_solubility  — single (solute, solvent, T) prediction
  temperature_scan    — solubility vs temperature curve
  interpret_prediction — human-readable report
  save_model / load_model — checkpoint I/O
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from .config import TGNNSolvConfig
from .features import (
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    compute_molecular_descriptors,
    graph_feature_spec_from_config,
    smiles_to_descriptor_prior_features,
    smiles_to_graph,
    smiles_to_group_prior_features,
    smiles_to_morgan_fp,
)
from .group_contribution import GC_FALLBACK_PRIORS, compute_gc_priors
from .ionic_features import (
    applicability_domain_flags,
    compute_ionic_features,
    flag_no_melting_point,
)
from .model import TGNNSolv
from .baselines.direct_gnn import DirectGNN
from .data.solvent_types import solvent_type_id_from_smiles
from .unifac import modified_unifac_lngamma_inf


def _smiles_to_graph_for_model(
    smiles: str,
    model: TGNNSolv | DirectGNN,
):
    """Build a graph that matches the feature flags stored on the loaded model config."""
    return smiles_to_graph(
        smiles,
        use_gasteiger_charges=bool(
            getattr(model.cfg, "use_gasteiger_charges", False)
        ),
        use_phys_edge_features=bool(
            getattr(model.cfg, "use_phys_edge_features", False)
        ),
        explicit_h_small_molecules=bool(
            getattr(model.cfg, "explicit_h_small_molecules", False)
        ),
        explicit_h_max_heavy_atoms=int(
            getattr(model.cfg, "explicit_h_max_heavy_atoms", 3)
        ),
    )


# ================================================================== #
#  Single prediction                                                  #
# ================================================================== #

@torch.no_grad()
def predict_solubility(
    model: TGNNSolv,
    solute_smiles: str,
    solvent_smiles: str,
    T: float = 298.15,
    device: torch.device = None,
    has_T_m: bool | None = None,
) -> Dict:
    """
    Predict solubility for a single (solute, solvent, T) system.

    Returns a dict with all predicted quantities and intermediates.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    sol_graph = _smiles_to_graph_for_model(solute_smiles, model)
    slv_graph = _smiles_to_graph_for_model(solvent_smiles, model)
    if sol_graph is None:
        raise ValueError(f"Cannot parse solute SMILES: {solute_smiles}")
    if slv_graph is None:
        raise ValueError(f"Cannot parse solvent SMILES: {solvent_smiles}")

    sol_batch = Batch.from_data_list([sol_graph]).to(device)
    slv_batch = Batch.from_data_list([slv_graph]).to(device)
    T_tensor = torch.tensor([T], device=device)
    solute_morgan_fp = None
    solvent_morgan_fp = None
    solute_descriptors = None
    solvent_descriptors = None
    ionic_features = None
    solute_descriptor_prior_features = None
    solvent_descriptor_prior_features = None
    solute_group_prior_features = None
    solvent_group_prior_features = None
    unifac_ln_gamma_inf = None
    unifac_gamma_mask = None
    T_m_gc = None
    dH_fus_gc = None
    dCp_fus_gc = None
    if model.cfg.use_morgan_features:
        sol_fp = smiles_to_morgan_fp(
            solute_smiles,
            radius=model.cfg.morgan_radius,
            n_bits=model.cfg.morgan_n_bits,
        )
        slv_fp = smiles_to_morgan_fp(
            solvent_smiles,
            radius=model.cfg.morgan_radius,
            n_bits=model.cfg.morgan_n_bits,
        )
        if sol_fp is None or slv_fp is None:
            raise ValueError("Failed to compute Morgan fingerprints for inference.")
        solute_morgan_fp = torch.tensor(sol_fp, device=device).unsqueeze(0)
        solvent_morgan_fp = torch.tensor(slv_fp, device=device).unsqueeze(0)
    if model.cfg.use_descriptor_augmentation:
        sol_desc = compute_molecular_descriptors(solute_smiles)
        slv_desc = compute_molecular_descriptors(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute RDKit descriptors for inference.")
        solute_descriptors = torch.tensor(sol_desc, device=device).unsqueeze(0)
        solvent_descriptors = torch.tensor(slv_desc, device=device).unsqueeze(0)
    if model.cfg.use_ionic_features:
        ionic_features = torch.tensor(
            compute_ionic_features(solute_smiles, solvent_smiles),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
    if model.cfg.use_descriptor_priors:
        sol_desc = smiles_to_descriptor_prior_features(solute_smiles)
        slv_desc = smiles_to_descriptor_prior_features(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute descriptor priors for inference.")
        solute_descriptor_prior_features = torch.tensor(
            sol_desc,
            device=device,
        ).unsqueeze(0)
        solvent_descriptor_prior_features = torch.tensor(
            slv_desc,
            device=device,
        ).unsqueeze(0)
    if model.cfg.requires_group_prior_features:
        sol_group = smiles_to_group_prior_features(solute_smiles)
        slv_group = smiles_to_group_prior_features(solvent_smiles)
        if sol_group is None or slv_group is None:
            raise ValueError("Failed to compute fixed group priors for inference.")
        solute_group_prior_features = torch.tensor(
            sol_group,
            device=device,
        ).unsqueeze(0)
        solvent_group_prior_features = torch.tensor(
            slv_group,
            device=device,
        ).unsqueeze(0)
    if getattr(model.cfg, "use_unifac_gamma_prior", False):
        lng = modified_unifac_lngamma_inf(solute_smiles, solvent_smiles, float(T))
        has_unifac = lng is not None
        unifac_ln_gamma_inf = torch.tensor(
            [float(lng) if has_unifac else 0.0],
            device=device,
            dtype=torch.float32,
        )
        unifac_gamma_mask = torch.tensor(
            [has_unifac],
            device=device,
            dtype=torch.bool,
        )
    if model.cfg.use_gc_priors_crystal:
        gc_priors = compute_gc_priors(solute_smiles)
        if any(gc_priors[key] is None for key in ("T_m_gc", "dH_fus_gc", "dCp_fus_gc")):
            gc_priors = GC_FALLBACK_PRIORS
        T_m_gc = torch.tensor([gc_priors["T_m_gc"]], device=device)
        dH_fus_gc = torch.tensor([gc_priors["dH_fus_gc"]], device=device)
        dCp_fus_gc = torch.tensor([gc_priors["dCp_fus_gc"]], device=device)

    solvent_type = torch.tensor(
        [solvent_type_id_from_smiles(solvent_smiles)],
        device=device,
        dtype=torch.long,
    )
    decomposition_handling = flag_no_melting_point(solute_smiles)
    has_decomposition_T = decomposition_handling != "standard"
    has_valid_T_m = bool(has_T_m) and not has_decomposition_T
    inference_targets = {
        "T": T_tensor,
        "has_T_m": torch.tensor([has_valid_T_m], device=device, dtype=torch.bool),
        "has_valid_T_m": torch.tensor([has_valid_T_m], device=device, dtype=torch.bool),
        "has_decomposition_T": torch.tensor(
            [has_decomposition_T],
            device=device,
            dtype=torch.bool,
        ),
    }
    output = model(
        sol_batch,
        slv_batch,
        T_tensor,
        solvent_type=solvent_type,
        solute_morgan_fp=solute_morgan_fp,
        solvent_morgan_fp=solvent_morgan_fp,
        solute_descriptors=solute_descriptors,
        solvent_descriptors=solvent_descriptors,
        ionic_features=ionic_features,
        solute_descriptor_prior_features=solute_descriptor_prior_features,
        solvent_descriptor_prior_features=solvent_descriptor_prior_features,
        solute_group_prior_features=solute_group_prior_features,
        solvent_group_prior_features=solvent_group_prior_features,
        unifac_ln_gamma_inf=unifac_ln_gamma_inf,
        unifac_gamma_mask=unifac_gamma_mask,
        T_m_gc=T_m_gc,
        dH_fus_gc=dH_fus_gc,
        dCp_fus_gc=dCp_fus_gc,
        targets=inference_targets,
    )

    direct_sigma = None
    direct_log_sigma = None
    if "ln_x2_direct_sigma" in output:
        direct_sigma = output["ln_x2_direct_sigma"].item()
    if "ln_x2_direct_log_sigma" in output:
        direct_log_sigma = output["ln_x2_direct_log_sigma"].item()

    nrtl_params = output["nrtl_params"]
    nrtl_payload = {}
    if "tau_ref_12" in nrtl_params:
        nrtl_payload = {
            "tau_ref_12": nrtl_params["tau_ref_12"].item(),
            "tau_ref_21": nrtl_params["tau_ref_21"].item(),
            "tau_inv_12": nrtl_params["tau_inv_12"].item(),
            "tau_inv_21": nrtl_params["tau_inv_21"].item(),
        }
    elif "tau_a12" in nrtl_params:
        nrtl_payload = {
            "tau_a12": nrtl_params["tau_a12"].item(),
            "tau_b12": nrtl_params["tau_b12"].item(),
            "tau_c12": nrtl_params["tau_c12"].item(),
            "tau_a21": nrtl_params["tau_a21"].item(),
            "tau_b21": nrtl_params["tau_b21"].item(),
            "tau_c21": nrtl_params["tau_c21"].item(),
        }
    else:
        nrtl_payload = {
            "dg_12": nrtl_params["dg_12"].item(),
            "dg_21": nrtl_params["dg_21"].item(),
            "a_T12": nrtl_params["a_T12"].item(),
            "a_T21": nrtl_params["a_T21"].item(),
        }

    ln_x2 = output["ln_x2"].item()
    return {
        "solute": solute_smiles,
        "solvent": solvent_smiles,
        "T": T,
        "x2": output["x2"].item(),
        "ln_x2": ln_x2,
        "x_ideal": output["physics"]["x_ideal"].item(),
        "gamma_2": math.exp(output["physics"]["ln_gamma_2"].item()),
        "ln_gamma_2": output["physics"]["ln_gamma_2"].item(),
        "Phi": output["physics"]["Phi"].item(),
        "T_m": output["fusion_params"]["T_m"].item(),
        "dH_fus": output["fusion_params"]["dH_fus"].item(),
        "dCp_fus": output["fusion_params"]["dCp_fus"].item(),
        "T_m_gc": (
            output["fusion_gc_priors"]["T_m_gc"].item()
            if "fusion_gc_priors" in output
            else None
        ),
        "dH_fus_gc": (
            output["fusion_gc_priors"]["dH_fus_gc"].item()
            if "fusion_gc_priors" in output
            else None
        ),
        "dCp_fus_gc": (
            output["fusion_gc_priors"]["dCp_fus_gc"].item()
            if "fusion_gc_priors" in output
            else None
        ),
        "tau_12": output["physics"]["tau_12"].item(),
        "tau_21": output["physics"]["tau_21"].item(),
        "alpha_12": output["nrtl_params"]["alpha_12"].item(),
        "hansen_sol": output["hansen_sol"][0].tolist(),
        "hansen_slv": output["hansen_slv"][0].tolist(),
        "Ra": output["Ra"].item(),
        "correction": output["correction"].item(),
        "gate": output["gate"].item(),
        "direct_sigma": direct_sigma,
        "direct_log_sigma": direct_log_sigma,
        "direct_phi_mask": bool(
            output.get("solver_fusion_params", {})
            .get(
                "direct_phi_mask",
                torch.tensor([False], device=device),
            )[0]
            .detach()
            .cpu()
            .item()
        )
        if isinstance(output.get("solver_fusion_params"), dict)
        else False,
        "applicability_flags": applicability_domain_flags(
            solute_smiles,
            solvent_smiles,
            has_T_m=has_valid_T_m,
            predicted_ln_x2=ln_x2,
        ),
        **nrtl_payload,
    }


# ================================================================== #
#  Temperature scan                                                   #
# ================================================================== #

@torch.no_grad()
def temperature_scan(
    model: TGNNSolv,
    solute_smiles: str,
    solvent_smiles: str,
    T_min: float = 270.0,
    T_max: float = 340.0,
    n_points: int = 15,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Predict solubility at multiple temperatures.

    Returns DataFrame with columns:
      T, x2, ln_x2, x_ideal, gamma_2, correction
    """
    T_values = np.linspace(T_min, T_max, n_points)
    results = []
    for T_val in T_values:
        r = predict_solubility(
            model, solute_smiles, solvent_smiles, float(T_val), device,
        )
        results.append(r)

    df = pd.DataFrame(results)
    return df[["T", "x2", "ln_x2", "x_ideal", "gamma_2", "correction"]]


# ================================================================== #
#  Interpretation                                                     #
# ================================================================== #

def interpret_prediction(result: Dict) -> str:
    """Generate a human-readable report from a prediction dict."""
    lines = []
    lines.append("=" * 60)
    lines.append("Solubility Prediction Report")
    lines.append("=" * 60)
    lines.append(f"Solute:  {result['solute']}")
    lines.append(f"Solvent: {result['solvent']}")
    lines.append(
        f"T = {result['T']:.1f} K ({result['T'] - 273.15:.1f} °C)"
    )

    # Solubility
    lines.append("")
    lines.append("PREDICTED SOLUBILITY:")
    lines.append(f"  x₂ = {result['x2']:.5f} (mole fraction)")
    lines.append(f"  ln(x₂) = {result['ln_x2']:.3f}")
    if result["x2"] > 0.1:
        lines.append("  → High solubility (>10 mol%)")
    elif result["x2"] > 0.01:
        lines.append("  → Moderate solubility (1–10 mol%)")
    elif result["x2"] > 1e-4:
        lines.append("  → Low solubility (0.01–1 mol%)")
    else:
        lines.append("  → Very low solubility (<0.01 mol%)")

    # Decomposition
    lines.append("")
    lines.append("DECOMPOSITION:")
    lines.append("  ln(x₂) = -Φ - ln(γ₂) + correction")
    lines.append(
        f"  = {-result['Phi']:.3f} + "
        f"{-result['ln_gamma_2']:.3f} + "
        f"{result['correction']:.3f}"
    )
    total_mag = abs(result["Phi"]) + abs(result["ln_gamma_2"]) + 1e-8
    pct_ideal = abs(result["Phi"]) / total_mag * 100
    lines.append(
        f"  Crystal term (-Φ): {-result['Phi']:.3f} "
        f"({pct_ideal:.0f}% of total)"
    )
    lines.append(
        f"  Non-ideal term (-ln γ₂): {-result['ln_gamma_2']:.3f} "
        f"({100 - pct_ideal:.0f}%)"
    )
    if abs(result["correction"]) > 0.5:
        lines.append(
            f"  WARNING: Large correction ({result['correction']:.3f}) - "
            f"NRTL may be inadequate"
        )
    if result.get("direct_phi_mask"):
        lines.append("  NOTE: effective direct Phi(T) branch was used.")

    flags = result.get("applicability_flags") or []
    if flags:
        lines.append("")
        lines.append("APPLICABILITY DOMAIN FLAGS:")
        for flag in flags:
            lines.append(f"  - {flag}")

    # Crystal properties
    lines.append("")
    lines.append("CRYSTAL PROPERTIES (solute):")
    lines.append(
        f"  T_m = {result['T_m']:.1f} K "
        f"({result['T_m'] - 273.15:.1f} °C)"
    )
    lines.append(f"  ΔH_fus = {result['dH_fus']:.0f} J/mol")
    lines.append(f"  ΔCp_fus = {result['dCp_fus']:.1f} J/(mol·K)")
    dS = result["dH_fus"] / result["T_m"]
    ok = "OK" if 20 < dS < 120 else "⚠ unusual"
    lines.append(f"  ΔS_fus = {dS:.1f} J/(mol·K) ({ok}; Walden ≈ 56.5)")

    # Activity coefficient
    lines.append("")
    lines.append("ACTIVITY COEFFICIENT:")
    lines.append(f"  γ₂ = {result['gamma_2']:.3f}")
    if result["gamma_2"] > 100:
        lines.append("  → Strong positive deviation")
    elif result["gamma_2"] > 5:
        lines.append("  → Moderate positive deviation")
    elif result["gamma_2"] > 0.5:
        lines.append("  → Near-ideal to weak positive deviation")
    else:
        lines.append("  → Negative deviation (favorable interactions)")

    # Direct-path uncertainty (if available)
    if result.get("direct_sigma") is not None:
        lines.append("")
        lines.append("DIRECT-PATH UNCERTAINTY:")
        lines.append(
            f"  sigma_ln_x2 = {result['direct_sigma']:.3f}"
        )

    # NRTL
    lines.append("")
    lines.append("NRTL PARAMETERS:")
    lines.append(f"  tau_12(T) = {result['tau_12']:.3f}")
    lines.append(f"  tau_21(T) = {result['tau_21']:.3f}")
    lines.append(f"  alpha_12 = {result['alpha_12']:.3f}")

    # Hansen
    h_sol = result["hansen_sol"]
    h_slv = result["hansen_slv"]
    lines.append("")
    lines.append("HANSEN PARAMETERS:")
    lines.append(
        f"  Solute:  δd={h_sol[0]:.1f}, δp={h_sol[1]:.1f}, "
        f"δh={h_sol[2]:.1f}"
    )
    lines.append(
        f"  Solvent: δd={h_slv[0]:.1f}, δp={h_slv[1]:.1f}, "
        f"δh={h_slv[2]:.1f}"
    )
    lines.append(f"  Ra = {result['Ra']:.1f} MPa^½")
    if result["Ra"] < 8:
        lines.append("  → Good match ('like dissolves like')")
    elif result["Ra"] < 15:
        lines.append("  → Moderate match")
    else:
        lines.append("  → Poor match")

    return "\n".join(lines)


@torch.no_grad()
def predict_direct_solubility(
    model: DirectGNN,
    solute_smiles: str,
    solvent_smiles: str,
    T: float = 298.15,
    device: torch.device = None,
    has_T_m: bool | None = None,
) -> Dict:
    """Predict solubility for a single system with DirectGNN."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    sol_graph = _smiles_to_graph_for_model(solute_smiles, model)
    slv_graph = _smiles_to_graph_for_model(solvent_smiles, model)
    if sol_graph is None:
        raise ValueError(f"Cannot parse solute SMILES: {solute_smiles}")
    if slv_graph is None:
        raise ValueError(f"Cannot parse solvent SMILES: {solvent_smiles}")

    sol_batch = Batch.from_data_list([sol_graph]).to(device)
    slv_batch = Batch.from_data_list([slv_graph]).to(device)
    T_tensor = torch.tensor([T], device=device)

    solute_morgan_fp = None
    solvent_morgan_fp = None
    if model.cfg.use_morgan_features:
        sol_fp = smiles_to_morgan_fp(
            solute_smiles,
            radius=model.cfg.morgan_radius,
            n_bits=model.cfg.morgan_n_bits,
        )
        slv_fp = smiles_to_morgan_fp(
            solvent_smiles,
            radius=model.cfg.morgan_radius,
            n_bits=model.cfg.morgan_n_bits,
        )
        if sol_fp is None or slv_fp is None:
            raise ValueError("Failed to compute Morgan fingerprints for DirectGNN inference.")
        solute_morgan_fp = torch.tensor(sol_fp, device=device).unsqueeze(0)
        solvent_morgan_fp = torch.tensor(slv_fp, device=device).unsqueeze(0)

    solute_descriptors = None
    solvent_descriptors = None
    ionic_features = None
    if model.cfg.use_descriptor_augmentation:
        sol_desc = compute_molecular_descriptors(solute_smiles)
        slv_desc = compute_molecular_descriptors(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute RDKit descriptors for DirectGNN inference.")
        solute_descriptors = torch.tensor(sol_desc, device=device).unsqueeze(0)
        solvent_descriptors = torch.tensor(slv_desc, device=device).unsqueeze(0)
    if model.cfg.use_ionic_features:
        ionic_features = torch.tensor(
            compute_ionic_features(solute_smiles, solvent_smiles),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

    output = model(
        sol_batch,
        slv_batch,
        T_tensor,
        solute_morgan_fp=solute_morgan_fp,
        solvent_morgan_fp=solvent_morgan_fp,
        solute_descriptors=solute_descriptors,
        solvent_descriptors=solvent_descriptors,
        ionic_features=ionic_features,
    )

    ln_x2 = output["ln_x2"].item()
    return {
        "solute": solute_smiles,
        "solvent": solvent_smiles,
        "T": T,
        "x2": output["x2"].item(),
        "ln_x2": ln_x2,
        "model_family": "direct_gnn",
        "uses_morgan": bool(model.cfg.use_morgan_features),
        "uses_descriptors": bool(model.cfg.use_descriptor_augmentation),
        "uses_ionic_features": bool(model.cfg.use_ionic_features),
        "applicability_flags": applicability_domain_flags(
            solute_smiles,
            solvent_smiles,
            has_T_m=bool(has_T_m) if has_T_m is not None else False,
            predicted_ln_x2=ln_x2,
        ),
    }


@torch.no_grad()
def temperature_scan_direct(
    model: DirectGNN,
    solute_smiles: str,
    solvent_smiles: str,
    T_min: float = 270.0,
    T_max: float = 340.0,
    n_points: int = 15,
    device: torch.device = None,
) -> pd.DataFrame:
    """Predict DirectGNN solubility across temperature."""
    T_values = np.linspace(T_min, T_max, n_points)
    results = []
    for T_val in T_values:
        results.append(
            predict_direct_solubility(
                model,
                solute_smiles,
                solvent_smiles,
                float(T_val),
                device,
            )
        )
    df = pd.DataFrame(results)
    return df[["T", "x2", "ln_x2"]]


def interpret_direct_prediction(result: Dict) -> str:
    """Generate a readable report for DirectGNN predictions."""
    lines = []
    lines.append("=" * 60)
    lines.append("DirectGNN Prediction Report")
    lines.append("=" * 60)
    lines.append(f"Solute:  {result['solute']}")
    lines.append(f"Solvent: {result['solvent']}")
    lines.append(f"T = {result['T']:.1f} K ({result['T'] - 273.15:.1f} °C)")

    lines.append("")
    lines.append("PREDICTED SOLUBILITY:")
    lines.append(f"  x₂ = {result['x2']:.5f} (mole fraction)")
    lines.append(f"  ln(x₂) = {result['ln_x2']:.3f}")
    if result["x2"] > 0.1:
        lines.append("  → High solubility (>10 mol%)")
    elif result["x2"] > 0.01:
        lines.append("  → Moderate solubility (1–10 mol%)")
    elif result["x2"] > 1e-4:
        lines.append("  → Low solubility (0.01–1 mol%)")
    else:
        lines.append("  → Very low solubility (<0.01 mol%)")

    lines.append("")
    lines.append("MODEL PATH:")
    lines.append("  DirectGNN predicts ln(x₂) directly from the matched graph backbone.")
    lines.append("  No NRTL activity model, SLE solver, or bounded physics correction is used.")
    lines.append(
        f"  Morgan features: {'on' if result.get('uses_morgan') else 'off'}"
    )
    lines.append(
        f"  RDKit descriptor augmentation: {'on' if result.get('uses_descriptors') else 'off'}"
    )
    lines.append(
        f"  Ionic/context features: {'on' if result.get('uses_ionic_features') else 'off'}"
    )

    flags = result.get("applicability_flags") or []
    if flags:
        lines.append("")
        lines.append("APPLICABILITY DOMAIN FLAGS:")
        for flag in flags:
            lines.append(f"  - {flag}")

    return "\n".join(lines)


# ================================================================== #
#  Model I/O                                                          #
# ================================================================== #

def save_model(
    model: TGNNSolv,
    cfg: TGNNSolvConfig,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model checkpoint with config and metadata."""
    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "node_feat_dim": int(getattr(model, "node_feat_dim", NODE_FEAT_DIM)),
        "edge_feat_dim": int(getattr(model, "edge_feat_dim", EDGE_FEAT_DIM)),
        "metadata": metadata or {},
    }
    if cfg.use_descriptor_augmentation:
        checkpoint["descriptor_mean"] = model.descriptor_mean.detach().cpu()
        checkpoint["descriptor_std"] = model.descriptor_std.detach().cpu()
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    device: torch.device = None,
) -> Tuple[TGNNSolv, TGNNSolvConfig]:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format at {path}: expected a dict payload, "
            f"got {type(checkpoint).__name__}."
        )
    if "config" not in checkpoint or not isinstance(checkpoint["config"], dict):
        top_keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise ValueError(
            f"Checkpoint at {path} does not contain a TGNN-Solv config block. "
            f"Top-level keys: {top_keys or 'none'}. "
            "This loader only supports TGNN-Solv-style checkpoints saved with "
            "`config` and compatible model weights."
        )
    model_class = str(checkpoint.get("model_class", ""))
    model_type = str(checkpoint.get("model_type", ""))
    if "directgnn" in model_class.lower() or "direct" in model_type.lower():
        raise ValueError(
            f"Checkpoint at {path} appears to belong to {model_class or model_type}, "
            "not the TGNN-Solv physics model. The current inference helpers in "
            "`tgnn_solv.inference` only support TGNN-Solv checkpoints."
        )
    if checkpoint.get("descriptor_mean") is not None:
        descriptor_mean = checkpoint["descriptor_mean"]
        cfg_dict = dict(checkpoint["config"])
        cfg_dict["descriptor_dim"] = int(torch.as_tensor(descriptor_mean).numel())
        cfg = TGNNSolvConfig(**cfg_dict)
    else:
        cfg = TGNNSolvConfig(**checkpoint["config"])
    feature_spec = graph_feature_spec_from_config(cfg)
    node_feat_dim = int(checkpoint.get("node_feat_dim", feature_spec.node_dim))
    edge_feat_dim = int(checkpoint.get("edge_feat_dim", feature_spec.edge_dim))
    model = TGNNSolv(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        cfg=cfg,
    ).to(device)
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible_state, strict=False)
    if cfg.use_descriptor_augmentation:
        descriptor_mean = checkpoint.get("descriptor_mean")
        descriptor_std = checkpoint.get("descriptor_std")
        if descriptor_mean is not None and descriptor_std is not None:
            model.set_descriptor_normalization(descriptor_mean, descriptor_std)

    print(f"Model loaded from {path}")
    if checkpoint.get("metadata"):
        for k, v in checkpoint["metadata"].items():
            print(f"  {k}: {v}")

    return model, cfg


def load_directgnn_model(
    path: str,
    device: torch.device = None,
) -> Tuple[DirectGNN, TGNNSolvConfig]:
    """Load a DirectGNN baseline checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format at {path}: expected a dict payload, "
            f"got {type(checkpoint).__name__}."
        )
    if "config" not in checkpoint or not isinstance(checkpoint["config"], dict):
        top_keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise ValueError(
            f"Checkpoint at {path} does not contain a DirectGNN config block. "
            f"Top-level keys: {top_keys or 'none'}."
        )
    cfg = TGNNSolvConfig(**checkpoint["config"])
    feature_spec = graph_feature_spec_from_config(cfg)
    model = DirectGNN(
        node_feat_dim=int(checkpoint.get("node_feat_dim", feature_spec.node_dim)),
        edge_feat_dim=int(checkpoint.get("edge_feat_dim", feature_spec.edge_dim)),
        cfg=cfg,
    ).to(device)
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(compatible_state, strict=False)

    if cfg.use_descriptor_augmentation:
        descriptor_mean = checkpoint.get("descriptor_mean")
        descriptor_std = checkpoint.get("descriptor_std")
        if descriptor_mean is not None and descriptor_std is not None:
            model.set_descriptor_normalization(descriptor_mean, descriptor_std)

    print(f"DirectGNN model loaded from {path}")
    return model, cfg
