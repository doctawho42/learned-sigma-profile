"""
Baseline: Dual-GNN with thermometer temperature encoding.

Same GNN backbone for solute and solvent, graph-level vectors
concatenated, temperature encoded as thermometer bins.
No physics layer, no NRTL, no SLE — pure data-driven.
"""

from .direct_gnn import DirectGNN as DirectGNN
from .direct_gnn import DirectGNNTrainer as DirectGNNTrainer
from .runner import compare_with_tgnn as compare_with_tgnn
from .runner import run_baseline as run_baseline

__all__ = [
    "DirectGNN",
    "DirectGNNTrainer",
    "run_baseline",
    "compare_with_tgnn",
]
