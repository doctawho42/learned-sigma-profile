"""VT-2005 sigma-profile oracle: SMILES -> (p_sigma, area) for eval-time injection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data.utils import canonicalize


def load_sigma_profiles(csv_path: str, n_bins: int = 51) -> dict[str, tuple[np.ndarray, float]]:
    """Map canonical SMILES -> (p_sigma[n_bins] area-weighted, area). Skips unparseable."""
    df = pd.read_csv(csv_path)
    cols = [f"sigma_p_{i}" for i in range(n_bins)]
    table: dict[str, tuple[np.ndarray, float]] = {}
    for rec in df.itertuples(index=False):
        d = rec._asdict()
        key = canonicalize(str(d.get("smiles", "")))
        if key is None:
            continue
        p = np.array([float(d[c]) for c in cols], dtype=float)
        area = float(d.get("sigma_area", p.sum()))
        table[key] = (p, area)
    return table


def build_oracle_tensors(
    smiles_list, table: dict[str, tuple[np.ndarray, float]], n_bins: int = 51
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row (p_sigma (B,n_bins), area (B,), mask (B,) bool). Unmatched -> zeros + False."""
    B = len(smiles_list)
    p = torch.zeros(B, n_bins)
    A = torch.zeros(B)
    mask = torch.zeros(B, dtype=torch.bool)
    for i, smi in enumerate(smiles_list):
        key = canonicalize(str(smi))
        hit = table.get(key) if key is not None else None
        if hit is not None:
            p[i] = torch.tensor(hit[0], dtype=torch.float)
            A[i] = float(hit[1])
            mask[i] = True
    return p, A, mask
