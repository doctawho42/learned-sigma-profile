"""
Data pipeline for TGNN-Solv.

Modules:
  utils    — SMILES canonicalization, download, scaffold
  sources  — BigSolDB, Bradley MP, Hansen, IDAC, CombiSolv loaders
  builder  — DataBuilder merging all sources
  dataset  — PyTorch Dataset and DataLoader factories
  split    — Scaffold-based train/val/test splitting
"""

from .utils import DATA_DIR as DATA_DIR
from .utils import PROCESSED_DIR as PROCESSED_DIR
from .utils import RAW_DIR as RAW_DIR
from .utils import canonicalize as canonicalize
from .utils import get_scaffold as get_scaffold
from .sources import (
    load_bigsoldb as load_bigsoldb,
    load_fusion_enthalpies as load_fusion_enthalpies,
    load_hansen as load_hansen,
    load_idac as load_idac,
    load_melting_points as load_melting_points,
)
from .builder import DataBuilder as DataBuilder
from .builder import filter_for_sle as filter_for_sle
from .dataset import TGNNSolvDataset as TGNNSolvDataset
from .dataset import collate_fn as collate_fn
from .dataset import make_loaders as make_loaders
from .split import scaffold_split as scaffold_split
from .split_registry import (
    SPLIT_DISPLAY_NAMES as SPLIT_DISPLAY_NAMES,
)
from .split_registry import (
    SPLIT_FILE_SUFFIXES as SPLIT_FILE_SUFFIXES,
)
from .split_registry import SPLIT_MODES as SPLIT_MODES
from .split_registry import build_split_metadata as build_split_metadata
from .split_registry import get_split_display_name as get_split_display_name
from .split_registry import infer_split_mode_from_path as infer_split_mode_from_path
from .split_registry import normalize_split_mode as normalize_split_mode
from .split_registry import resolve_split_modes as resolve_split_modes
from .split_registry import split_filename as split_filename
from .split_registry import split_paths as split_paths

__all__ = [
    "canonicalize",
    "get_scaffold",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "load_bigsoldb",
    "load_melting_points",
    "load_fusion_enthalpies",
    "load_hansen",
    "load_idac",
    "DataBuilder",
    "filter_for_sle",
    "TGNNSolvDataset",
    "collate_fn",
    "make_loaders",
    "scaffold_split",
    "SPLIT_MODES",
    "SPLIT_FILE_SUFFIXES",
    "SPLIT_DISPLAY_NAMES",
    "normalize_split_mode",
    "resolve_split_modes",
    "get_split_display_name",
    "split_filename",
    "split_paths",
    "infer_split_mode_from_path",
    "build_split_metadata",
]
