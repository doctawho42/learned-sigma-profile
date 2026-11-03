"""Grouped namespace package."""

from __future__ import annotations

from .ablation import run_ablation_study as run_ablation_study
from .ablation import run_single_ablation as run_single_ablation

__all__ = [
    "run_ablation_study",
    "run_single_ablation",
]
