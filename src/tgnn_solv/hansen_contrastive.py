"""Hansen-distance contrastive objectives and pseudo-Hansen helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def hansen_distance_matrix(params: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return pairwise Hansen Ra distances for ``(delta_d, delta_p, delta_h)``."""
    if params.size(-1) != 3:
        raise ValueError("Hansen distance requires last dimension == 3")
    dd = params[:, 0:1]
    dp = params[:, 1:2]
    dh = params[:, 2:3]
    return torch.sqrt(
        4.0 * (dd - dd.T).pow(2)
        + (dp - dp.T).pow(2)
        + (dh - dh.T).pow(2)
        + eps
    )


def euclidean_property_distance_matrix(params: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return pairwise Euclidean distances for 1D/2D property targets."""
    return torch.cdist(params, params, p=2)


def pairwise_alignment_loss(
    latent_dist: Tensor,
    target_dist: Tensor,
    pair_weight: Tensor | None = None,
) -> Tensor:
    """Weighted pairwise MSE excluding diagonal self-pairs."""
    if latent_dist.shape != target_dist.shape:
        raise ValueError("latent_dist and target_dist must have identical shapes")
    if latent_dist.ndim != 2 or latent_dist.size(0) != latent_dist.size(1):
        raise ValueError("pairwise distance matrices must be square")
    if latent_dist.size(0) < 2:
        return latent_dist.sum() * 0.0

    mask = torch.ones_like(latent_dist, dtype=torch.bool)
    mask.fill_diagonal_(False)
    diff_sq = (latent_dist - target_dist).pow(2)
    if pair_weight is not None:
        weights = pair_weight.to(diff_sq).clamp_min(0.0)
        weights = weights * mask.to(weights)
        denom = weights.sum().clamp_min(1.0)
        return (diff_sq * weights).sum() / denom
    return diff_sq[mask].mean()


class HansenContrastiveLoss(nn.Module):
    """Soft contrastive loss aligning latent distances with Hansen distances."""

    def __init__(self, temperature: float = 0.1, margin_scale: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.margin_scale = float(margin_scale)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def _zero(self, embeddings: Tensor) -> Tensor:
        return embeddings.sum() * 0.0

    def _scale_target(self, target_dist: Tensor) -> Tensor:
        scale = F.softplus(self.alpha) * self.margin_scale
        return scale * target_dist.detach() + self.beta

    def _forward_with_target_dist(
        self,
        embeddings: Tensor,
        target_dist: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        if embeddings.size(0) < 2:
            return self._zero(embeddings)
        latent_dist = torch.cdist(embeddings, embeddings, p=2)
        pair_weight = None
        if sample_weight is not None:
            w = sample_weight.to(embeddings).flatten().clamp_min(0.0)
            pair_weight = w[:, None] * w[None, :]
        return pairwise_alignment_loss(
            latent_dist,
            self._scale_target(target_dist.to(embeddings)),
            pair_weight=pair_weight,
        )

    def forward(
        self,
        embeddings: Tensor,
        hansen_params: Tensor,
        hansen_mask: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        idx = hansen_mask.to(embeddings.device).bool().nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return self._zero(embeddings)
        emb = embeddings[idx]
        han = hansen_params.to(embeddings)[idx]
        weights = sample_weight.to(embeddings)[idx] if sample_weight is not None else None
        return self._forward_with_target_dist(
            emb,
            hansen_distance_matrix(han),
            sample_weight=weights,
        )

    def forward_1d(
        self,
        embeddings: Tensor,
        values: Tensor,
        mask: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        idx = mask.to(embeddings.device).bool().nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return self._zero(embeddings)
        emb = embeddings[idx]
        vals = values.to(embeddings)[idx].reshape(idx.numel(), -1)
        weights = sample_weight.to(embeddings)[idx] if sample_weight is not None else None
        return self._forward_with_target_dist(
            emb,
            euclidean_property_distance_matrix(vals),
            sample_weight=weights,
        )

    def forward_2d(
        self,
        embeddings: Tensor,
        values: Tensor,
        mask: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        return self.forward_1d(embeddings, values, mask, sample_weight=sample_weight)


class ChannelHansenContrastiveLoss(nn.Module):
    """Channel-specific Hansen contrastive loss for TIMP embeddings."""

    def __init__(self, temperature: float = 0.1, margin_scale: float = 1.0) -> None:
        super().__init__()
        self.disp_loss = HansenContrastiveLoss(temperature, margin_scale)
        self.polar_loss = HansenContrastiveLoss(temperature, margin_scale)

    def forward(
        self,
        g_disp: Tensor | None,
        g_polar: Tensor | None,
        hansen_params: Tensor,
        hansen_mask: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        if g_disp is None or g_polar is None:
            ref = hansen_params
            return ref.sum() * 0.0
        disp_target = hansen_params[:, 0:1]
        polar_target = hansen_params[:, 1:3]
        l_disp = self.disp_loss.forward_1d(
            g_disp,
            disp_target,
            hansen_mask,
            sample_weight=sample_weight,
        )
        l_polar = self.polar_loss.forward_2d(
            g_polar,
            polar_target,
            hansen_mask,
            sample_weight=sample_weight,
        )
        return l_disp + l_polar


class PairHansenContrastiveLoss(nn.Module):
    """Pair-level contrastive loss using Hansen compatibility Ra values."""

    def __init__(self, temperature: float = 0.1, margin_scale: float = 1.0) -> None:
        super().__init__()
        self.loss = HansenContrastiveLoss(temperature, margin_scale)

    def forward(
        self,
        pair_embeddings: Tensor,
        Ra_values: Tensor,
        pair_hansen_mask: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        idx = pair_hansen_mask.to(pair_embeddings.device).bool().nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return pair_embeddings.sum() * 0.0
        emb = pair_embeddings[idx]
        ra = Ra_values.to(pair_embeddings)[idx].reshape(idx.numel(), 1)
        weights = sample_weight.to(pair_embeddings)[idx] if sample_weight is not None else None
        target_dist = euclidean_property_distance_matrix(ra)
        return self.loss._forward_with_target_dist(
            emb,
            target_dist,
            sample_weight=weights,
        )


def channel_orthogonality_penalty(g_disp: Tensor | None, g_polar: Tensor | None) -> Tensor:
    """Penalize collapse of TIMP dispersive and polar channel means."""
    if g_disp is None or g_polar is None:
        if g_disp is not None:
            return g_disp.sum() * 0.0
        if g_polar is not None:
            return g_polar.sum() * 0.0
        return torch.zeros(())
    if g_disp.numel() == 0 or g_polar.numel() == 0:
        return g_disp.sum() * 0.0 + g_polar.sum() * 0.0
    disp_mean = g_disp.mean(dim=0, keepdim=True)
    polar_mean = g_polar.mean(dim=0, keepdim=True)
    dim = min(disp_mean.size(-1), polar_mean.size(-1))
    if dim == 0:
        return g_disp.sum() * 0.0 + g_polar.sum() * 0.0
    return F.cosine_similarity(disp_mean[:, :dim], polar_mean[:, :dim], dim=-1).abs().mean()


def pseudo_hansen_from_descriptors(
    *,
    mol_logp: float,
    mol_mr: float,
    heavy_atom_count: float,
    tpsa: float,
    num_h_acceptors: float,
    num_h_donors: float,
) -> tuple[float, float, float]:
    """Deterministic RDKit-descriptor heuristic for pseudo-Hansen parameters."""
    hac = max(float(heavy_atom_count), 1.0)
    logp = float(mol_logp)
    mr_per_atom = float(mol_mr) / hac
    tpsa = max(float(tpsa), 0.0)
    hba = max(float(num_h_acceptors), 0.0)
    hbd = max(float(num_h_donors), 0.0)

    delta_d = 15.0 + 0.20 * logp + 0.22 * mr_per_atom
    delta_p = 2.0 + 0.18 * tpsa + 2.50 * hba - 0.10 * logp
    delta_h = 1.0 + 0.35 * tpsa + 10.0 * hbd + 1.20 * hba
    return (
        float(np.clip(delta_d, 10.0, 25.0)),
        float(np.clip(delta_p, 0.0, 30.0)),
        float(np.clip(delta_h, 0.0, 45.0)),
    )


def pseudo_hansen_from_smiles(smiles: str) -> tuple[float, float, float] | None:
    """Compute pseudo-Hansen parameters from a SMILES string using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return pseudo_hansen_from_descriptors(
            mol_logp=Crippen.MolLogP(mol),
            mol_mr=Crippen.MolMR(mol),
            heavy_atom_count=rdMolDescriptors.CalcNumHeavyAtoms(mol),
            tpsa=Descriptors.TPSA(mol),
            num_h_acceptors=Lipinski.NumHAcceptors(mol),
            num_h_donors=Lipinski.NumHDonors(mol),
        )
    except Exception:
        return None


@dataclass
class PseudoHansenLinearModel:
    """Small train-only linear regressor for descriptor-to-Hansen calibration."""

    coef: np.ndarray
    intercept: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return x @ self.coef + self.intercept


def fit_pseudo_hansen_linear(
    descriptor_rows: Iterable[Iterable[float]],
    hansen_rows: Iterable[Iterable[float]],
    *,
    ridge: float = 1.0e-3,
) -> PseudoHansenLinearModel:
    """Fit a ridge-linear pseudo-Hansen model on train-only rows."""
    x = np.asarray(list(descriptor_rows), dtype=np.float64)
    y = np.asarray(list(hansen_rows), dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("Expected descriptor matrix [N, d] and Hansen matrix [N, 3]")
    if x.shape[0] < 3:
        raise ValueError("At least three labelled molecules are required to fit pseudo-Hansen")
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1.0e-8] = 1.0
    xz = (x - x_mean) / x_std
    design = np.concatenate([xz, np.ones((xz.shape[0], 1))], axis=1)
    eye = np.eye(design.shape[1])
    eye[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + ridge * eye, design.T @ y)
    coef = beta[:-1] / x_std[:, None]
    intercept = beta[-1] - x_mean @ coef
    return PseudoHansenLinearModel(coef=coef, intercept=intercept)


compute_pseudo_hansen = pseudo_hansen_from_smiles
