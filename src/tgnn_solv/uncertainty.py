"""
Uncertainty estimation for TGNN-Solv predictions.

Two complementary approaches:

1. **MC-Dropout** — Enable dropout at inference time and run
   multiple forward passes.  The variance of predictions
   approximates epistemic (model) uncertainty.
   Cost: N_samples × single forward pass.  No retraining needed.

2. **Deep Ensemble** — Train K independent models (different seeds)
   and average their predictions.  Disagreement = uncertainty.
   Cost: K × full training.  Best calibration.

Usage::

    from tgnn_solv.uncertainty import MCDropoutPredictor, EnsemblePredictor

    # MC-Dropout (single model)
    mc = MCDropoutPredictor(model, n_samples=30)
    result = mc.predict("CC(=O)Nc1ccc(O)cc1", "CCO", T=298.15)
    print(result["ln_x2_mean"], result["ln_x2_std"])

    # Ensemble (multiple models)
    ens = EnsemblePredictor([model1, model2, model3])
    result = ens.predict("CC(=O)Nc1ccc(O)cc1", "CCO", T=298.15)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .baselines.direct_gnn import DirectGNN
from .model import TGNNSolv
from .features import (
    compute_molecular_descriptors,
    smiles_to_descriptor_prior_features,
    smiles_to_graph,
    smiles_to_group_prior_features,
    smiles_to_morgan_fp,
)
from .group_contribution import GC_FALLBACK_PRIORS, compute_gc_priors
from .progress import progress, trange
from .data.solvent_types import solvent_type_id_from_smiles
from .unifac import modified_unifac_lngamma_inf


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _enable_dropout(model: nn.Module) -> None:
    """Force all Dropout layers into training mode (stochastic)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.MultiheadAttention)):
            m.train()


def _prepare_inputs(
    model: TGNNSolv | DirectGNN,
    solute_smiles: str,
    solvent_smiles: str,
    T: float,
    device: torch.device,
) -> tuple[Batch, Batch, torch.Tensor]:
    """Build batched graph inputs for a single system."""
    sol_g = smiles_to_graph(
        solute_smiles,
        use_gasteiger_charges=bool(getattr(model.cfg, "use_gasteiger_charges", False)),
        use_phys_edge_features=bool(getattr(model.cfg, "use_phys_edge_features", False)),
        explicit_h_small_molecules=bool(getattr(model.cfg, "explicit_h_small_molecules", False)),
        explicit_h_max_heavy_atoms=int(getattr(model.cfg, "explicit_h_max_heavy_atoms", 3)),
    )
    slv_g = smiles_to_graph(
        solvent_smiles,
        use_gasteiger_charges=bool(getattr(model.cfg, "use_gasteiger_charges", False)),
        use_phys_edge_features=bool(getattr(model.cfg, "use_phys_edge_features", False)),
        explicit_h_small_molecules=bool(getattr(model.cfg, "explicit_h_small_molecules", False)),
        explicit_h_max_heavy_atoms=int(getattr(model.cfg, "explicit_h_max_heavy_atoms", 3)),
    )
    if sol_g is None:
        raise ValueError(f"Invalid solute SMILES: {solute_smiles}")
    if slv_g is None:
        raise ValueError(f"Invalid solvent SMILES: {solvent_smiles}")

    sol_batch = Batch.from_data_list([sol_g]).to(device)
    slv_batch = Batch.from_data_list([slv_g]).to(device)
    T_tensor = torch.tensor([T], device=device)
    return sol_batch, slv_batch, T_tensor


def _prepare_forward_kwargs(
    model: TGNNSolv | DirectGNN,
    solute_smiles: str,
    solvent_smiles: str,
    device: torch.device,
    temperature: float,
) -> dict[str, Any]:
    """Build optional forward kwargs required by the current model config.

    ``temperature`` is required rather than defaulted: the UNIFAC prior below is temperature
    dependent, and until 2026-08-23 this function read a name ``T`` that was never in its scope --
    an F821 that raised NameError for anyone running with ``use_unifac_gamma_prior: true``, which
    two committed configs set. A default would have hidden the same bug behind a wrong number.
    """
    if isinstance(model, DirectGNN):
        kwargs: dict[str, Any] = {}
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
                raise ValueError("Failed to compute Morgan fingerprints for DirectGNN uncertainty inference.")
            kwargs["solute_morgan_fp"] = torch.tensor(sol_fp, device=device).unsqueeze(0)
            kwargs["solvent_morgan_fp"] = torch.tensor(slv_fp, device=device).unsqueeze(0)
        if model.cfg.use_descriptor_augmentation:
            sol_desc = compute_molecular_descriptors(solute_smiles)
            slv_desc = compute_molecular_descriptors(solvent_smiles)
            if sol_desc is None or slv_desc is None:
                raise ValueError("Failed to compute RDKit descriptors for DirectGNN uncertainty inference.")
            kwargs["solute_descriptors"] = torch.tensor(sol_desc, device=device).unsqueeze(0)
            kwargs["solvent_descriptors"] = torch.tensor(slv_desc, device=device).unsqueeze(0)
        return kwargs

    kwargs: dict[str, Any] = {
        "solvent_type": torch.tensor(
            [solvent_type_id_from_smiles(solvent_smiles)],
            device=device,
            dtype=torch.long,
        )
    }

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
            raise ValueError("Failed to compute Morgan fingerprints for uncertainty inference.")
        kwargs["solute_morgan_fp"] = torch.tensor(sol_fp, device=device).unsqueeze(0)
        kwargs["solvent_morgan_fp"] = torch.tensor(slv_fp, device=device).unsqueeze(0)

    if model.cfg.use_descriptor_augmentation:
        sol_desc = compute_molecular_descriptors(solute_smiles)
        slv_desc = compute_molecular_descriptors(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute RDKit descriptors for uncertainty inference.")
        kwargs["solute_descriptors"] = torch.tensor(
            sol_desc,
            device=device,
        ).unsqueeze(0)
        kwargs["solvent_descriptors"] = torch.tensor(
            slv_desc,
            device=device,
        ).unsqueeze(0)

    if model.cfg.use_descriptor_priors:
        sol_desc = smiles_to_descriptor_prior_features(solute_smiles)
        slv_desc = smiles_to_descriptor_prior_features(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute descriptor priors for uncertainty inference.")
        kwargs["solute_descriptor_prior_features"] = torch.tensor(
            sol_desc,
            device=device,
        ).unsqueeze(0)
        kwargs["solvent_descriptor_prior_features"] = torch.tensor(
            slv_desc,
            device=device,
        ).unsqueeze(0)
    if model.cfg.requires_group_prior_features:
        sol_group = smiles_to_group_prior_features(solute_smiles)
        slv_group = smiles_to_group_prior_features(solvent_smiles)
        if sol_group is None or slv_group is None:
            raise ValueError("Failed to compute fixed group priors for uncertainty inference.")
        kwargs["solute_group_prior_features"] = torch.tensor(
            sol_group,
            device=device,
        ).unsqueeze(0)
        kwargs["solvent_group_prior_features"] = torch.tensor(
            slv_group,
            device=device,
        ).unsqueeze(0)

    if getattr(model.cfg, "use_unifac_gamma_prior", False):
        lng = modified_unifac_lngamma_inf(solute_smiles, solvent_smiles, float(temperature))
        has_unifac = lng is not None
        kwargs["unifac_ln_gamma_inf"] = torch.tensor(
            [float(lng) if has_unifac else 0.0],
            device=device,
            dtype=torch.float32,
        )
        kwargs["unifac_gamma_mask"] = torch.tensor(
            [has_unifac],
            device=device,
            dtype=torch.bool,
        )

    if model.cfg.use_gc_priors_crystal:
        gc_priors = compute_gc_priors(solute_smiles)
        if any(gc_priors[key] is None for key in ("T_m_gc", "dH_fus_gc", "dCp_fus_gc")):
            gc_priors = GC_FALLBACK_PRIORS
        kwargs["T_m_gc"] = torch.tensor([gc_priors["T_m_gc"]], device=device)
        kwargs["dH_fus_gc"] = torch.tensor([gc_priors["dH_fus_gc"]], device=device)
        kwargs["dCp_fus_gc"] = torch.tensor([gc_priors["dCp_fus_gc"]], device=device)

    return kwargs


def _summarize_samples(samples: dict[str, list[float]]) -> dict[str, float]:
    """Compute mean, std, and confidence intervals from samples."""
    result = {}
    for key, values in samples.items():
        arr = np.array(values)
        result[f"{key}_mean"] = float(arr.mean())
        result[f"{key}_std"] = float(arr.std())
        result[f"{key}_q05"] = float(np.percentile(arr, 5))
        result[f"{key}_q95"] = float(np.percentile(arr, 95))
    return result


def _sample_outputs(
    model: TGNNSolv | DirectGNN,
    out: dict[str, Any],
) -> dict[str, float]:
    """Extract uncertainty-ready scalar outputs from one forward pass."""
    samples = {
        "ln_x2": float(out["ln_x2"].item()),
        "x2": float(out["x2"].item()),
    }
    if isinstance(model, DirectGNN):
        return samples
    samples.update(
        {
            "gamma_2": math.exp(out["physics"]["ln_gamma_2"].item()),
            "T_m": float(out["fusion_params"]["T_m"].item()),
            "dH_fus": float(out["fusion_params"]["dH_fus"].item()),
            "Phi": float(out["physics"]["Phi"].item()),
            "ln_gamma_2": float(out["physics"]["ln_gamma_2"].item()),
            "correction": float(out["correction"].item()),
        }
    )
    return samples


# ================================================================== #
#  MC-Dropout Predictor                                               #
# ================================================================== #

class MCDropoutPredictor:
    """
    Monte Carlo Dropout uncertainty estimation.

    Runs N forward passes with dropout enabled, then computes
    statistics over the predictions.

    Parameters
    ----------
    model : TGNNSolv
        Trained model (must have Dropout layers).
    n_samples : int
        Number of stochastic forward passes (default 30).
    """

    def __init__(self, model: TGNNSolv | DirectGNN, n_samples: int = 30) -> None:
        self.model = model
        self.n_samples = n_samples
        self.device = next(model.parameters()).device
        self.model_family = "direct_gnn" if isinstance(model, DirectGNN) else "tgnn_solv"

    @torch.no_grad()
    def predict(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        T: float = 298.15,
    ) -> dict[str, float | int | str]:
        """
        Predict with uncertainty.

        Returns dict with:
          solute, solvent, T,
          ln_x2_mean, ln_x2_std, ln_x2_q05, ln_x2_q95,
          x2_mean, x2_std,
          gamma_2_mean, gamma_2_std,
          T_m_mean, T_m_std,
          dH_fus_mean, dH_fus_std,
          n_samples
        """
        sol_b, slv_b, T_t = _prepare_inputs(
            self.model,
            solute_smiles,
            solvent_smiles,
            T,
            self.device,
        )
        forward_kwargs = _prepare_forward_kwargs(
            self.model,
            solute_smiles,
            solvent_smiles,
            self.device,
            T,
        )

        # Set model to eval, then re-enable dropout
        self.model.eval()
        _enable_dropout(self.model)

        samples = {
            "ln_x2": [],
            "x2": [],
        }
        if not isinstance(self.model, DirectGNN):
            samples.update(
                {
                    "gamma_2": [],
                    "T_m": [],
                    "dH_fus": [],
                    "Phi": [],
                    "ln_gamma_2": [],
                    "correction": [],
                }
            )

        for _ in trange(self.n_samples, desc="MC-dropout samples", leave=False):
            out = self.model(sol_b, slv_b, T_t, **forward_kwargs)
            sample = _sample_outputs(self.model, out)
            for key, value in sample.items():
                samples[key].append(value)

        # Restore full eval mode
        self.model.eval()

        result = {
            "solute": solute_smiles,
            "solvent": solvent_smiles,
            "T": T,
            "n_samples": self.n_samples,
            "model_family": self.model_family,
        }
        result.update(_summarize_samples(samples))
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        systems: list[tuple[str, str, float]],
    ) -> pd.DataFrame:
        """
        Predict for multiple (solute, solvent, T) systems.

        Parameters
        ----------
        systems : list of (solute_smiles, solvent_smiles, T) tuples

        Returns
        -------
        DataFrame with uncertainty estimates for each system.
        """
        results = []
        for sol, slv, T in progress(
            systems, desc="MC-dropout batch", leave=False
        ):
            try:
                r = self.predict(sol, slv, T)
                results.append(r)
            except ValueError as e:
                print(f"  Skipping {sol}/{slv}: {e}")
        return pd.DataFrame(results)


# ================================================================== #
#  Deep Ensemble Predictor                                            #
# ================================================================== #

class EnsemblePredictor:
    """
    Deep Ensemble uncertainty estimation.

    Averages predictions from K independently trained models.
    Disagreement between models captures epistemic uncertainty.

    Parameters
    ----------
    models : list of TGNNSolv
        K trained models (different random seeds / data folds).
    """

    def __init__(self, models: list[TGNNSolv | DirectGNN]) -> None:
        if len(models) < 2:
            raise ValueError("Ensemble requires ≥ 2 models")
        families = {"direct_gnn" if isinstance(model, DirectGNN) else "tgnn_solv" for model in models}
        if len(families) != 1:
            raise ValueError("All ensemble members must belong to the same model family.")
        self.models = models
        self.device = next(models[0].parameters()).device
        self.model_family = next(iter(families))

    @torch.no_grad()
    def predict(
        self,
        solute_smiles: str,
        solvent_smiles: str,
        T: float = 298.15,
    ) -> dict[str, float | int | str]:
        """Predict with ensemble uncertainty."""
        sol_b, slv_b, T_t = _prepare_inputs(
            self.models[0],
            solute_smiles,
            solvent_smiles,
            T,
            self.device,
        )

        samples = {
            "ln_x2": [],
            "x2": [],
        }
        if self.model_family != "direct_gnn":
            samples.update(
                {
                    "gamma_2": [],
                    "T_m": [],
                    "dH_fus": [],
                    "correction": [],
                }
            )

        for model in progress(
            self.models, desc="Ensemble models", leave=False
        ):
            model.eval()
            forward_kwargs = _prepare_forward_kwargs(
                model,
                solute_smiles,
                solvent_smiles,
                self.device,
                T,
            )
            out = model(sol_b, slv_b, T_t, **forward_kwargs)
            sample = _sample_outputs(model, out)
            for key, value in sample.items():
                samples[key].append(value)

        result = {
            "solute": solute_smiles,
            "solvent": solvent_smiles,
            "T": T,
            "n_models": len(self.models),
            "model_family": self.model_family,
        }
        result.update(_summarize_samples(samples))
        return result


# ================================================================== #
#  Calibration check                                                  #
# ================================================================== #

def calibration_report(
    predictions: list[dict[str, float | int | str]],
    true_ln_x2: list[float],
) -> dict[str, float | int]:
    """
    Check if predicted confidence intervals are well-calibrated.

    A perfectly calibrated model should have ~90% of true values
    falling within the 5th–95th percentile interval.

    Parameters
    ----------
    predictions : list of dicts from MCDropoutPredictor/EnsemblePredictor
    true_ln_x2 : corresponding experimental values

    Returns
    -------
    dict with coverage_90, mean_interval_width, PICP, MPIW, RMSE, MAE
    """
    n = len(predictions)
    inside = 0
    widths = []
    errors = []

    for pred, true_val in zip(predictions, true_ln_x2):
        q05 = pred["ln_x2_q05"]
        q95 = pred["ln_x2_q95"]
        mean = pred["ln_x2_mean"]

        if q05 <= true_val <= q95:
            inside += 1
        widths.append(q95 - q05)
        errors.append(mean - true_val)

    errors = np.array(errors)
    widths = np.array(widths)

    return {
        "n_samples": n,
        "PICP_90": inside / n,                   # Prediction Interval Coverage Probability
        "MPIW": float(widths.mean()),              # Mean Prediction Interval Width
        "MAE": float(np.abs(errors).mean()),
        "RMSE": float(np.sqrt((errors ** 2).mean())),
        "sharpness": float(widths.std()),          # Lower = more consistent
    }
