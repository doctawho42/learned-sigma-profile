"""Shared helpers for sigma-grounding tests: a tiny cosmo_sac trainer + batch.

Deviation from the task-2-brief skeleton: uses ``make_loader`` (same pattern as
``test_sigma_aux_stream.py``) rather than ``TGNNSolvDataset + DataLoader``
directly, because ``make_loader`` wires the correct collate function and
``expected_sigma_bins`` consistently with the rest of the test suite.
Also mirrors the exact config fields used by ``_cosmo_cfg()`` in
``test_sigma_aux_stream.py`` so the model builds without shape errors.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from tgnn_solv.config import TGNNSolvConfig
from tgnn_solv.data.dataset import make_loader
from tgnn_solv.model import TGNNSolv
from tgnn_solv.trainer import TGNNSolvTrainer

N_BINS = 51
_SMILES = ["CCO", "CCCCCC", "c1ccccc1", "CC(C)O"]


def tiny_cosmo_config() -> TGNNSolvConfig:
    """Minimal cosmo_sac config that matches the existing sigma-stream tests."""
    return TGNNSolvConfig(
        activity_model="cosmo_sac",
        hidden_dim=32,
        n_gnn_layers=2,
        n_cross_attn_layers=1,
        n_attn_heads=4,
        pair_dim=64,
        solvent_moe_hidden=64,
        solvent_type_emb_dim=8,
        n_iter_train=2,
        n_iter_eval=2,
        set2set_steps=2,
        cosmo_sac_gamma_iter_train=4,
        cosmo_sac_gamma_iter_eval=4,
        sigma_aux_steps_per_epoch=2,
        # encoder_role_mode defaults to "shared_residual" — role adapters exist
    )


def _sigma_pool_df(n: int = 4) -> pd.DataFrame:
    """Tiny sigma-profile-only data frame (no solubility target)."""
    rng = np.random.RandomState(0)
    shape = rng.dirichlet(np.ones(N_BINS), size=n)
    rows = []
    for i, smi in enumerate(_SMILES[:n]):
        row: dict = {
            "solute_smiles": smi,
            "solvent_smiles": smi,
            "temperature": 298.15,
            "has_solubility": False,
            "has_sigma_profile": True,
            "sigma_area": 40.0 + 10.0 * i,
            # Fields required by TGNNSolvDataset even when has_solubility is False
            "ln_x2": 0.0,
            "source": "t",
            "T_m": 0.0,
            "has_T_m": False,
            "dH_fus": 0.0,
            "has_dH_fus": False,
            "hansen_d": 0.0,
            "hansen_p": 0.0,
            "hansen_h": 0.0,
            "has_hansen": False,
            "ln_gamma_inf": 0.0,
            "has_gamma_inf": False,
        }
        for b in range(N_BINS):
            row[f"sigma_p_{b}"] = float(shape[i, b])
        rows.append(row)
    return pd.DataFrame(rows)


def make_tiny_cosmo_trainer_and_loader():
    """Return a (TGNNSolvTrainer, DataLoader) pair for sigma-grounding tests.

    The loader yields (sol_batch, slv_batch, targets) tuples where every sample
    has ``has_sigma_profile=True``, so ``targets["sigma_profile_mask"]`` is
    all-True and both ``_sigma_forward_loss`` and ``_train_sigma_aux_batch``
    exercise the real code path.
    """
    cfg = tiny_cosmo_config()
    model = TGNNSolv(cfg=cfg)
    trainer = TGNNSolvTrainer(model, cfg)
    loader = make_loader(
        _sigma_pool_df(),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        cache=False,
        use_pair_temperature_batching=False,
        expected_sigma_bins=cfg.cosmo_sac_n_bins,
    )
    return trainer, loader


def first_batch(loader):
    """Return the first batch from a loader."""
    return next(iter(loader))
