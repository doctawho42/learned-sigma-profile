"""Grouped namespace wrapper for Hansen-contrastive training objectives."""

from __future__ import annotations

from ..hansen_contrastive import (
    ChannelHansenContrastiveLoss,
    HansenContrastiveLoss,
    PairHansenContrastiveLoss,
    PseudoHansenLinearModel,
    channel_orthogonality_penalty,
    compute_pseudo_hansen,
    fit_pseudo_hansen_linear,
    hansen_distance_matrix,
    pairwise_alignment_loss,
    pseudo_hansen_from_descriptors,
    pseudo_hansen_from_smiles,
)

__all__ = [
    "ChannelHansenContrastiveLoss",
    "HansenContrastiveLoss",
    "PairHansenContrastiveLoss",
    "PseudoHansenLinearModel",
    "channel_orthogonality_penalty",
    "compute_pseudo_hansen",
    "fit_pseudo_hansen_linear",
    "hansen_distance_matrix",
    "pairwise_alignment_loss",
    "pseudo_hansen_from_descriptors",
    "pseudo_hansen_from_smiles",
]
