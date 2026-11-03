"""Grouped namespace package."""

from __future__ import annotations

from .evaluator import Evaluator as Evaluator
from .inference import load_model as load_model
from .inference import predict_solubility as predict_solubility
from .applications import pharma_capability_matrix as pharma_capability_matrix
from .applications import solvent_swap_metrics as solvent_swap_metrics
from .applications import synthesis_window_metrics as synthesis_window_metrics
from .reporting import build_report_payload as build_report_payload
from .reporting import normalize_report_payload as normalize_report_payload

__all__ = [
    "Evaluator",
    "load_model",
    "predict_solubility",
    "pharma_capability_matrix",
    "solvent_swap_metrics",
    "synthesis_window_metrics",
    "build_report_payload",
    "normalize_report_payload",
]
