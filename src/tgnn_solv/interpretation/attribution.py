"""Gradient-based atomic attribution for TGNN-Solv and DirectGNN."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from tgnn_solv.baselines.direct_gnn import DirectGNN
from tgnn_solv.data.solvent_types import solvent_type_id_from_smiles
from tgnn_solv.features import (
    compute_molecular_descriptors,
    smiles_to_descriptor_prior_features,
    smiles_to_graph,
    smiles_to_group_prior_features,
    smiles_to_morgan_fp,
)
from tgnn_solv.group_contribution import GC_FALLBACK_PRIORS, compute_gc_priors
from tgnn_solv.model import TGNNSolv
from tgnn_solv.unifac import modified_unifac_lngamma_inf


def _as_device(device: str | torch.device | None, model: torch.nn.Module) -> torch.device:
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _to_batch(graph: Data | Batch, device: torch.device) -> Batch:
    """Convert a single PyG Data graph into a Batch and move it to device."""
    if isinstance(graph, Batch):
        return graph.to(device)
    return Batch.from_data_list([graph]).to(device)


def _tensor_1d(value: float | Tensor, device: torch.device, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.to(device=device, dtype=dtype)
        return tensor.reshape(1) if tensor.ndim == 0 else tensor
    return torch.tensor([value], device=device, dtype=dtype)


def atom_labels_from_smiles(smiles: str) -> list[str]:
    """Return heavy-atom labels that match the graph atom order."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    return [f"{atom.GetSymbol()}{atom.GetIdx() + 1}" for atom in mol.GetAtoms()]


def build_single_system_inputs(
    model: TGNNSolv | DirectGNN,
    solute_smiles: str,
    solvent_smiles: str,
    temperature: float,
    *,
    device: str | torch.device | None = None,
) -> tuple[Batch, Batch, dict[str, Tensor]]:
    """Build graph batches and optional feature tensors for one prediction."""
    dev = _as_device(device, model)
    cfg = model.cfg
    solute_graph = smiles_to_graph(
        solute_smiles,
        use_gasteiger_charges=bool(getattr(cfg, "use_gasteiger_charges", False)),
        use_phys_edge_features=bool(getattr(cfg, "use_phys_edge_features", False)),
        explicit_h_small_molecules=bool(getattr(cfg, "explicit_h_small_molecules", False)),
        explicit_h_max_heavy_atoms=int(getattr(cfg, "explicit_h_max_heavy_atoms", 3)),
    )
    solvent_graph = smiles_to_graph(
        solvent_smiles,
        use_gasteiger_charges=bool(getattr(cfg, "use_gasteiger_charges", False)),
        use_phys_edge_features=bool(getattr(cfg, "use_phys_edge_features", False)),
        explicit_h_small_molecules=bool(getattr(cfg, "explicit_h_small_molecules", False)),
        explicit_h_max_heavy_atoms=int(getattr(cfg, "explicit_h_max_heavy_atoms", 3)),
    )
    if solute_graph is None:
        raise ValueError(f"Cannot build graph for solute SMILES: {solute_smiles}")
    if solvent_graph is None:
        raise ValueError(f"Cannot build graph for solvent SMILES: {solvent_smiles}")

    targets: dict[str, Tensor] = {
        "T": torch.tensor([float(temperature)], device=dev, dtype=torch.float32),
        "solvent_type": torch.tensor(
            [solvent_type_id_from_smiles(solvent_smiles)],
            device=dev,
            dtype=torch.long,
        ),
    }

    if getattr(cfg, "use_morgan_features", False):
        sol_fp = smiles_to_morgan_fp(
            solute_smiles,
            radius=cfg.morgan_radius,
            n_bits=cfg.morgan_n_bits,
        )
        slv_fp = smiles_to_morgan_fp(
            solvent_smiles,
            radius=cfg.morgan_radius,
            n_bits=cfg.morgan_n_bits,
        )
        if sol_fp is None or slv_fp is None:
            raise ValueError("Failed to compute Morgan fingerprints.")
        targets["solute_morgan_fp"] = torch.tensor(sol_fp, device=dev, dtype=torch.float32).unsqueeze(0)
        targets["solvent_morgan_fp"] = torch.tensor(slv_fp, device=dev, dtype=torch.float32).unsqueeze(0)

    if getattr(cfg, "use_descriptor_augmentation", False):
        sol_desc = compute_molecular_descriptors(solute_smiles)
        slv_desc = compute_molecular_descriptors(solvent_smiles)
        if sol_desc is None or slv_desc is None:
            raise ValueError("Failed to compute RDKit descriptors.")
        targets["solute_descriptors"] = torch.tensor(sol_desc, device=dev, dtype=torch.float32).unsqueeze(0)
        targets["solvent_descriptors"] = torch.tensor(slv_desc, device=dev, dtype=torch.float32).unsqueeze(0)

    if isinstance(model, TGNNSolv):
        if getattr(cfg, "use_descriptor_priors", False):
            sol_desc_prior = smiles_to_descriptor_prior_features(solute_smiles)
            slv_desc_prior = smiles_to_descriptor_prior_features(solvent_smiles)
            if sol_desc_prior is None or slv_desc_prior is None:
                raise ValueError("Failed to compute descriptor-prior features.")
            targets["solute_descriptor_prior_features"] = torch.tensor(
                sol_desc_prior, device=dev, dtype=torch.float32
            ).unsqueeze(0)
            targets["solvent_descriptor_prior_features"] = torch.tensor(
                slv_desc_prior, device=dev, dtype=torch.float32
            ).unsqueeze(0)
        if getattr(cfg, "requires_group_prior_features", False):
            sol_group = smiles_to_group_prior_features(solute_smiles)
            slv_group = smiles_to_group_prior_features(solvent_smiles)
            if sol_group is None or slv_group is None:
                raise ValueError("Failed to compute group-prior features.")
            targets["solute_group_prior_features"] = torch.tensor(sol_group, device=dev, dtype=torch.float32).unsqueeze(0)
            targets["solvent_group_prior_features"] = torch.tensor(slv_group, device=dev, dtype=torch.float32).unsqueeze(0)
        if getattr(cfg, "use_unifac_gamma_prior", False):
            lng = modified_unifac_lngamma_inf(solute_smiles, solvent_smiles, float(temperature))
            has_unifac = lng is not None
            targets["unifac_ln_gamma_inf"] = torch.tensor(
                [float(lng) if has_unifac else 0.0],
                device=dev,
                dtype=torch.float32,
            )
            targets["unifac_gamma_mask"] = torch.tensor(
                [has_unifac],
                device=dev,
                dtype=torch.bool,
            )
        if getattr(cfg, "use_gc_priors_crystal", False):
            priors = compute_gc_priors(solute_smiles)
            if any(priors[key] is None for key in ("T_m_gc", "dH_fus_gc", "dCp_fus_gc")):
                priors = GC_FALLBACK_PRIORS
            targets["T_m_gc"] = torch.tensor([priors["T_m_gc"]], device=dev, dtype=torch.float32)
            targets["dH_fus_gc"] = torch.tensor([priors["dH_fus_gc"]], device=dev, dtype=torch.float32)
            targets["dCp_fus_gc"] = torch.tensor([priors["dCp_fus_gc"]], device=dev, dtype=torch.float32)

    return _to_batch(solute_graph, dev), _to_batch(solvent_graph, dev), targets


class AtomAttribution:
    """Integrated Gradients attribution over molecular node features."""

    def __init__(self, model: TGNNSolv | DirectGNN, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def _move_targets(self, targets: Mapping[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in targets.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _forward_ln_x2(
        self,
        solute_graph: Batch,
        solvent_graph: Batch,
        targets: Mapping[str, Any],
    ) -> Tensor:
        T = _tensor_1d(targets["T"], self.device)
        kwargs = {
            "solute_morgan_fp": targets.get("solute_morgan_fp"),
            "solvent_morgan_fp": targets.get("solvent_morgan_fp"),
            "solute_descriptors": targets.get("solute_descriptors"),
            "solvent_descriptors": targets.get("solvent_descriptors"),
        }
        if isinstance(self.model, DirectGNN):
            output = self.model(solute_graph, solvent_graph, T, **kwargs)
            return output["ln_x2"]

        output = self.model(
            solute_graph,
            solvent_graph,
            T,
            solvent_type=targets.get("solvent_type"),
            solute_descriptor_prior_features=targets.get("solute_descriptor_prior_features"),
            solvent_descriptor_prior_features=targets.get("solvent_descriptor_prior_features"),
            solute_group_prior_features=targets.get("solute_group_prior_features"),
            solvent_group_prior_features=targets.get("solvent_group_prior_features"),
            T_m_gc=targets.get("T_m_gc"),
            dH_fus_gc=targets.get("dH_fus_gc"),
            dCp_fus_gc=targets.get("dCp_fus_gc"),
            targets=dict(targets),
            **kwargs,
        )
        return output.get("ln_x2_final", output["ln_x2"])

    def integrated_gradients(
        self,
        solute_graph: Data | Batch,
        solvent_graph: Data | Batch,
        targets: Mapping[str, Any],
        n_steps: int = 50,
        attribute_to: str = "solute",
    ) -> Tensor:
        """
        Compute per-atom Integrated Gradients attribution for predicted ``ln_x2``.

        Parameters
        ----------
        solute_graph, solvent_graph:
            Single-example PyG ``Data`` or ``Batch`` objects.
        targets:
            Mapping with at least ``T`` and any optional feature tensors required
            by the model config.
        n_steps:
            Number of Riemann integration steps.
        attribute_to:
            ``"solute"`` or ``"solvent"``.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if attribute_to not in {"solute", "solvent"}:
            raise ValueError("attribute_to must be 'solute' or 'solvent'.")

        targets_on_device = self._move_targets(targets)
        real_solute = _to_batch(solute_graph, self.device)
        real_solvent = _to_batch(solvent_graph, self.device)
        selected = real_solute if attribute_to == "solute" else real_solvent
        real_x = selected.x.detach()
        baseline_x = torch.zeros_like(real_x)
        avg_grad = torch.zeros_like(real_x)

        for alpha in torch.linspace(
            1.0 / n_steps,
            1.0,
            steps=n_steps,
            device=self.device,
            dtype=real_x.dtype,
        ):
            interp_x = (baseline_x + alpha * (real_x - baseline_x)).detach()
            interp_x.requires_grad_(True)
            interp_graph = selected.clone()
            interp_graph.x = interp_x
            sol_graph = interp_graph if attribute_to == "solute" else real_solute.clone()
            slv_graph = interp_graph if attribute_to == "solvent" else real_solvent.clone()

            self.model.zero_grad(set_to_none=True)
            ln_x2 = self._forward_ln_x2(sol_graph, slv_graph, targets_on_device)
            ln_x2.sum().backward()
            if interp_x.grad is None:
                raise RuntimeError("Integrated Gradients failed: input gradient is None.")
            avg_grad += interp_x.grad.detach()

        avg_grad /= float(n_steps)
        feature_attr = (real_x - baseline_x) * avg_grad
        return feature_attr.sum(dim=1).detach().cpu()
