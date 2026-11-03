"""Grouped namespace package."""

from __future__ import annotations

from .direct_gnn import DirectGNN as DirectGNN
from .direct_gnn import DirectGNNTrainer as DirectGNNTrainer
from .heads import FusionHead as FusionHead
from .layers import GNNEncoder as GNNEncoder
from .layers import GPSEncoder as GPSEncoder
from .tgnn import TGNNSolv as TGNNSolv

__all__ = [
    "TGNNSolv",
    "DirectGNN",
    "DirectGNNTrainer",
    "FusionHead",
    "GNNEncoder",
    "GPSEncoder",
]
