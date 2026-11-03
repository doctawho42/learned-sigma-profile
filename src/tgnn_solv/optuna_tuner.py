"""
Optuna hyperparameter tuning for TGNN-Solv and baselines.
"""

from __future__ import annotations

import gc
import random
import time
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TGNNSolvConfig
from .data.dataset import TGNNSolvDataset, make_loader
from .features import compute_descriptor_normalization_stats
from .model import TGNNSolv
from .trainer import TGNNSolvTrainer
from .baselines.direct_gnn import DirectGNN, DirectGNNTrainer

try:
    import optuna
except ImportError:  # pragma: no cover - optuna is optional
    optuna = None

AVAILABLE_MODELS = (
    "tgnn_solv",
    "tgnn_solv_gps",
    "tgnn_solv_descriptors",
    "no_cross_attn",
    "no_nrtl",
    "no_curriculum",
    "no_aux_losses",
    "no_correction",
    "no_implicit_diff",
    "small_128",
    "large_512",
    "direct_gnn",
    "direct_gnn_descriptors",
)


class OptunaTuner:
    """Object-oriented wrapper for Optuna hyperparameter tuning."""

    def __init__(
        self,
        datasets: Tuple[TGNNSolvDataset, TGNNSolvDataset, TGNNSolvDataset],
        device: torch.device,
        seed: int = 42,
        num_workers: int = 0,
        fixed_batch_size: Optional[int] = None,
        tune_arch: bool = True,
        base_cfg: TGNNSolvConfig | None = None,
        cfg_overrides: Optional[Dict[str, Optional[float]]] = None,
        baseline_epochs: int = 200,
        baseline_patience: int = 20,
    ) -> None:
        self.train_ds, self.val_ds, self.test_ds = datasets
        self.train_df = getattr(self.train_ds, "df", None)
        self.val_df = getattr(self.val_ds, "df", None)
        self.test_df = getattr(self.test_ds, "df", None)
        self.device = device
        self.seed = seed
        self.num_workers = num_workers
        self.fixed_batch_size = fixed_batch_size
        self.tune_arch = tune_arch
        self.baseline_epochs = baseline_epochs
        self.baseline_patience = baseline_patience

        cfg = replace(base_cfg) if base_cfg is not None else TGNNSolvConfig()
        if cfg_overrides:
            cfg = self._apply_overrides(cfg, cfg_overrides)
        self.base_cfg = cfg

    @staticmethod
    def require_optuna() -> None:
        if optuna is None:
            raise ImportError(
                "Optuna is not installed. Install with `pip install optuna`."
            )

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def load_csv_splits(
        train_csv: str, val_csv: str, test_csv: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        return train_df, val_df, test_df

    @staticmethod
    def build_datasets(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cfg: TGNNSolvConfig | None = None,
        cache: bool = True,
    ) -> Tuple[TGNNSolvDataset, TGNNSolvDataset, TGNNSolvDataset]:
        cfg = cfg or TGNNSolvConfig()
        train_ds = TGNNSolvDataset(
            train_df,
            cache=cache,
            use_morgan_features=cfg.use_morgan_features,
            morgan_radius=cfg.morgan_radius,
            morgan_n_bits=cfg.morgan_n_bits,
            use_descriptor_augmentation=cfg.use_descriptor_augmentation,
            use_ionic_features=cfg.use_ionic_features,
            use_descriptor_priors=cfg.use_descriptor_priors,
            use_group_priors=cfg.requires_group_prior_features,
            use_gc_priors_crystal=cfg.use_gc_priors_crystal,
            use_gasteiger_charges=cfg.use_gasteiger_charges,
            use_phys_edge_features=cfg.use_phys_edge_features,
            explicit_h_small_molecules=cfg.explicit_h_small_molecules,
            explicit_h_max_heavy_atoms=cfg.explicit_h_max_heavy_atoms,
            source_uncertainty_csv=(
                cfg.source_uncertainty_csv
                if cfg.use_source_uncertainty_weights
                else ""
            ),
            source_uncertainty_weight_mode=cfg.source_uncertainty_weight_mode,
            source_uncertainty_default_sigma_ln_x2=cfg.source_uncertainty_default_sigma_ln_x2,
            source_uncertainty_min_sigma_ln_x2=cfg.source_uncertainty_min_sigma_ln_x2,
            source_uncertainty_min_weight=cfg.source_uncertainty_min_weight,
            source_uncertainty_max_weight=cfg.source_uncertainty_max_weight,
        )
        val_ds = TGNNSolvDataset(
            val_df,
            cache=cache,
            use_morgan_features=cfg.use_morgan_features,
            morgan_radius=cfg.morgan_radius,
            morgan_n_bits=cfg.morgan_n_bits,
            use_descriptor_augmentation=cfg.use_descriptor_augmentation,
            use_ionic_features=cfg.use_ionic_features,
            use_descriptor_priors=cfg.use_descriptor_priors,
            use_group_priors=cfg.requires_group_prior_features,
            use_gc_priors_crystal=cfg.use_gc_priors_crystal,
            use_gasteiger_charges=cfg.use_gasteiger_charges,
            use_phys_edge_features=cfg.use_phys_edge_features,
            explicit_h_small_molecules=cfg.explicit_h_small_molecules,
            explicit_h_max_heavy_atoms=cfg.explicit_h_max_heavy_atoms,
            source_uncertainty_csv=(
                cfg.source_uncertainty_csv
                if cfg.use_source_uncertainty_weights
                else ""
            ),
            source_uncertainty_weight_mode=cfg.source_uncertainty_weight_mode,
            source_uncertainty_default_sigma_ln_x2=cfg.source_uncertainty_default_sigma_ln_x2,
            source_uncertainty_min_sigma_ln_x2=cfg.source_uncertainty_min_sigma_ln_x2,
            source_uncertainty_min_weight=cfg.source_uncertainty_min_weight,
            source_uncertainty_max_weight=cfg.source_uncertainty_max_weight,
        )
        test_ds = TGNNSolvDataset(
            test_df,
            cache=cache,
            use_morgan_features=cfg.use_morgan_features,
            morgan_radius=cfg.morgan_radius,
            morgan_n_bits=cfg.morgan_n_bits,
            use_descriptor_augmentation=cfg.use_descriptor_augmentation,
            use_ionic_features=cfg.use_ionic_features,
            use_descriptor_priors=cfg.use_descriptor_priors,
            use_group_priors=cfg.requires_group_prior_features,
            use_gc_priors_crystal=cfg.use_gc_priors_crystal,
            use_gasteiger_charges=cfg.use_gasteiger_charges,
            use_phys_edge_features=cfg.use_phys_edge_features,
            explicit_h_small_molecules=cfg.explicit_h_small_molecules,
            explicit_h_max_heavy_atoms=cfg.explicit_h_max_heavy_atoms,
            source_uncertainty_csv=(
                cfg.source_uncertainty_csv
                if cfg.use_source_uncertainty_weights
                else ""
            ),
            source_uncertainty_weight_mode=cfg.source_uncertainty_weight_mode,
            source_uncertainty_default_sigma_ln_x2=cfg.source_uncertainty_default_sigma_ln_x2,
            source_uncertainty_min_sigma_ln_x2=cfg.source_uncertainty_min_sigma_ln_x2,
            source_uncertainty_min_weight=cfg.source_uncertainty_min_weight,
            source_uncertainty_max_weight=cfg.source_uncertainty_max_weight,
        )
        return train_ds, val_ds, test_ds

    def _build_loaders(
        self,
        cfg: TGNNSolvConfig,
        batch_size: int,
        seed: int,
        include_test: bool = False,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        if not isinstance(self.train_df, pd.DataFrame) or not isinstance(
            self.val_df, pd.DataFrame
        ):
            raise ValueError(
                "OptunaTuner requires dataset objects with an attached `.df` dataframe."
            )

        def _make(frame: pd.DataFrame, *, shuffle: bool) -> DataLoader:
            return make_loader(
                frame,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                cache=True,
                drop_last=shuffle and len(frame) > batch_size,
                use_pair_temperature_batching=(
                    shuffle and cfg.use_pair_temperature_batching
                ),
                pair_temperature_min_group_size=(
                    cfg.pair_temperature_min_group_size
                ),
                pair_temperature_group_chunk_size=(
                    cfg.pair_temperature_group_chunk_size
                ),
                use_morgan_features=cfg.use_morgan_features,
                morgan_radius=cfg.morgan_radius,
                morgan_n_bits=cfg.morgan_n_bits,
                use_descriptor_augmentation=cfg.use_descriptor_augmentation,
                use_ionic_features=cfg.use_ionic_features,
                use_descriptor_priors=cfg.use_descriptor_priors,
                use_group_priors=cfg.requires_group_prior_features,
                use_gc_priors_crystal=cfg.use_gc_priors_crystal,
                use_gasteiger_charges=cfg.use_gasteiger_charges,
                use_phys_edge_features=cfg.use_phys_edge_features,
                explicit_h_small_molecules=cfg.explicit_h_small_molecules,
                explicit_h_max_heavy_atoms=cfg.explicit_h_max_heavy_atoms,
                use_pseudo_hansen=(cfg.use_hansen_contrastive or cfg.use_hansen_delta_loss) and cfg.use_pseudo_hansen,
                pseudo_hansen_weight_discount=cfg.pseudo_hansen_weight_discount,
                source_uncertainty_csv=(
                    cfg.source_uncertainty_csv
                    if cfg.use_source_uncertainty_weights
                    else ""
                ),
                source_uncertainty_weight_mode=cfg.source_uncertainty_weight_mode,
                source_uncertainty_default_sigma_ln_x2=cfg.source_uncertainty_default_sigma_ln_x2,
                source_uncertainty_min_sigma_ln_x2=cfg.source_uncertainty_min_sigma_ln_x2,
                source_uncertainty_min_weight=cfg.source_uncertainty_min_weight,
                source_uncertainty_max_weight=cfg.source_uncertainty_max_weight,
                seed=seed,
            )

        train_loader = _make(self.train_df, shuffle=True)
        val_loader = _make(self.val_df, shuffle=False)
        test_loader = None
        if include_test:
            if not isinstance(self.test_df, pd.DataFrame):
                raise ValueError(
                    "OptunaTuner requires a test dataframe when include_test=True."
                )
            test_loader = _make(self.test_df, shuffle=False)
        return train_loader, val_loader, test_loader

    @torch.no_grad()
    def _compute_val_mae(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> float:
        model.eval()
        all_pred = []
        all_true = []
        for sol_b, slv_b, tgt in loader:
            sol_b = sol_b.to(self.device)
            slv_b = slv_b.to(self.device)
            mask = tgt["has_solubility"]
            if not mask.any().item():
                continue
            T = tgt["T"].to(self.device)
            solvent_type = tgt.get("solvent_type")
            solute_morgan_fp = tgt.get("solute_morgan_fp")
            solvent_morgan_fp = tgt.get("solvent_morgan_fp")
            solute_descriptors = tgt.get("solute_descriptors")
            solvent_descriptors = tgt.get("solvent_descriptors")
            if isinstance(model, DirectGNN):
                out = model(
                    sol_b,
                    slv_b,
                    T,
                    solvent_type=solvent_type,
                    solute_morgan_fp=(
                        solute_morgan_fp.to(self.device)
                        if isinstance(solute_morgan_fp, torch.Tensor)
                        else None
                    ),
                    solvent_morgan_fp=(
                        solvent_morgan_fp.to(self.device)
                        if isinstance(solvent_morgan_fp, torch.Tensor)
                        else None
                    ),
                    solute_descriptors=(
                        solute_descriptors.to(self.device)
                        if isinstance(solute_descriptors, torch.Tensor)
                        else None
                    ),
                    solvent_descriptors=(
                        solvent_descriptors.to(self.device)
                        if isinstance(solvent_descriptors, torch.Tensor)
                        else None
                    ),
                )
                pred = out["ln_x2"].detach().cpu()[mask]
                true = tgt["ln_x2"][mask]
                all_pred.append(pred)
                all_true.append(true)
                continue
            solute_descriptor_prior_features = tgt.get(
                "solute_descriptor_prior_features"
            )
            solvent_descriptor_prior_features = tgt.get(
                "solvent_descriptor_prior_features"
            )
            solute_group_prior_features = tgt.get(
                "solute_group_prior_features"
            )
            solvent_group_prior_features = tgt.get(
                "solvent_group_prior_features"
            )
            T_m_gc = tgt.get("T_m_gc")
            dH_fus_gc = tgt.get("dH_fus_gc")
            dCp_fus_gc = tgt.get("dCp_fus_gc")
            out = model(
                sol_b,
                slv_b,
                T,
                solvent_type=solvent_type,
                solute_morgan_fp=(
                    solute_morgan_fp.to(self.device)
                    if isinstance(solute_morgan_fp, torch.Tensor)
                    else None
                ),
                solvent_morgan_fp=(
                    solvent_morgan_fp.to(self.device)
                    if isinstance(solvent_morgan_fp, torch.Tensor)
                    else None
                ),
                solute_descriptors=(
                    solute_descriptors.to(self.device)
                    if isinstance(solute_descriptors, torch.Tensor)
                    else None
                ),
                solvent_descriptors=(
                    solvent_descriptors.to(self.device)
                    if isinstance(solvent_descriptors, torch.Tensor)
                    else None
                ),
                solute_descriptor_prior_features=(
                    solute_descriptor_prior_features.to(self.device)
                    if isinstance(solute_descriptor_prior_features, torch.Tensor)
                    else None
                ),
                solvent_descriptor_prior_features=(
                    solvent_descriptor_prior_features.to(self.device)
                    if isinstance(solvent_descriptor_prior_features, torch.Tensor)
                    else None
                ),
                solute_group_prior_features=(
                    solute_group_prior_features.to(self.device)
                    if isinstance(solute_group_prior_features, torch.Tensor)
                    else None
                ),
                solvent_group_prior_features=(
                    solvent_group_prior_features.to(self.device)
                    if isinstance(solvent_group_prior_features, torch.Tensor)
                    else None
                ),
                T_m_gc=(
                    T_m_gc.to(self.device)
                    if isinstance(T_m_gc, torch.Tensor)
                    else None
                ),
                dH_fus_gc=(
                    dH_fus_gc.to(self.device)
                    if isinstance(dH_fus_gc, torch.Tensor)
                    else None
                ),
                dCp_fus_gc=(
                    dCp_fus_gc.to(self.device)
                    if isinstance(dCp_fus_gc, torch.Tensor)
                    else None
                ),
            )
            pred = out["ln_x2"].detach().cpu()[mask]
            true = tgt["ln_x2"][mask]
            all_pred.append(pred)
            all_true.append(true)
        if not all_pred:
            return float("inf")
        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        return (pred - true).abs().mean().item()

    @staticmethod
    def _apply_overrides(
        cfg: TGNNSolvConfig, overrides: Dict[str, Optional[float]]
    ) -> TGNNSolvConfig:
        updates = {k: v for k, v in overrides.items() if v is not None}
        if not updates:
            return cfg
        return replace(cfg, **updates)

    def _suggest_shared_arch(
        self,
        trial: "optuna.Trial",
        tune_arch: bool,
        tune_dropout: bool = True,
    ) -> Dict[str, int | float]:
        params: Dict[str, int | float] = {}
        if not tune_arch and not tune_dropout:
            return params
        if tune_arch:
            hidden_dim = trial.suggest_categorical(
                "hidden_dim", [128, 256, 384, 512]
            )
            n_gnn_layers = trial.suggest_int("n_gnn_layers", 4, 8)
            n_cross_attn_layers = trial.suggest_int(
                "n_cross_attn_layers", 2, 4
            )
            n_attn_heads = trial.suggest_categorical("n_attn_heads", [4, 8])
            pair_dim_mult = trial.suggest_categorical(
                "pair_dim_mult", [2, 3, 4]
            )
            pair_dim = hidden_dim * pair_dim_mult
            set2set_steps = trial.suggest_int("set2set_steps", 2, 4)
            moe_experts = trial.suggest_categorical(
                "solvent_moe_experts", [4, 6, 8]
            )
            moe_hidden = trial.suggest_categorical(
                "solvent_moe_hidden", [128, 256]
            )
            params.update(
                {
                    "hidden_dim": hidden_dim,
                    "n_gnn_layers": n_gnn_layers,
                    "n_cross_attn_layers": n_cross_attn_layers,
                    "n_attn_heads": n_attn_heads,
                    "pair_dim": pair_dim,
                    "set2set_steps": set2set_steps,
                    "solvent_moe_experts": moe_experts,
                    "solvent_moe_hidden": moe_hidden,
                }
            )
        if tune_dropout:
            params["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        return params

    def _suggest_tgnn_params(
        self,
        trial: "optuna.Trial",
        tune_arch: bool,
    ) -> Tuple[TGNNSolvConfig, int]:
        cfg = replace(self.base_cfg)
        arch = self._suggest_shared_arch(trial, tune_arch, tune_dropout=True)
        for k, v in arch.items():
            setattr(cfg, k, v)

        cfg.nrtl_tau_mode = trial.suggest_categorical(
            "nrtl_tau_mode", ["ref_invT", "legacy", "abc", "gamma_inf"]
        )

        lr_base = trial.suggest_float("lr_base", 5e-5, 5e-4, log=True)
        lr_phase1_mult = trial.suggest_categorical(
            "lr_phase1_mult", [2.0, 3.0, 4.0]
        )
        lr_phase3_mult = trial.suggest_categorical(
            "lr_phase3_mult", [0.01, 0.02, 0.05, 0.1, 0.2]
        )
        cfg.lr_phase2 = lr_base
        cfg.lr_phase1 = lr_base * lr_phase1_mult
        cfg.lr_phase3 = lr_base * lr_phase3_mult
        cfg.warmup_epochs = trial.suggest_categorical(
            "warmup_epochs", [0, 3, 5, 10]
        )
        cfg.phase2_correction_unfreeze_epoch = trial.suggest_categorical(
            "phase2_correction_unfreeze_epoch", [0, 10, 20, 40]
        )
        cfg.grad_clip = trial.suggest_categorical(
            "grad_clip", [0.5, 1.0, 2.0]
        )
        if cfg.encoder_type == "gps":
            cfg.gps_num_heads = trial.suggest_categorical(
                "gps_num_heads",
                [4, 8],
            )
            cfg.gps_positional_encoding = trial.suggest_categorical(
                "gps_positional_encoding",
                ["laplacian", "rwse"],
            )
            cfg.gps_pe_dim = trial.suggest_categorical(
                "gps_pe_dim",
                [8, 16, 24],
            )
        if cfg.use_descriptor_augmentation:
            cfg.descriptor_augmentation_hidden_dim = trial.suggest_categorical(
                "descriptor_augmentation_hidden_dim",
                [64, 128, 256],
            )

        if self.fixed_batch_size is None:
            batch_size = trial.suggest_categorical(
                "batch_size", [32, 64, 128]
            )
        else:
            batch_size = self.fixed_batch_size
        cfg.batch_size = batch_size
        return cfg, batch_size

    def _suggest_direct_params(
        self,
        trial: "optuna.Trial",
        tune_arch: bool,
    ) -> Tuple[TGNNSolvConfig, int, float]:
        cfg = replace(self.base_cfg)
        cfg.dropout = trial.suggest_float("dropout", 0.0, 0.3)
        if tune_arch:
            cfg.hidden_dim = trial.suggest_categorical(
                "hidden_dim", [128, 256]
            )
            cfg.n_gnn_layers = trial.suggest_categorical(
                "n_gnn_layers", [4, 6]
            )

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        cfg.direct_weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-3,
            log=True,
        )
        cfg.grad_clip = trial.suggest_float("grad_clip", 1.0, 5.0)
        if cfg.use_descriptor_augmentation:
            cfg.descriptor_augmentation_hidden_dim = trial.suggest_categorical(
                "descriptor_augmentation_hidden_dim",
                [64, 128, 256],
            )
        if self.fixed_batch_size is None:
            batch_size = trial.suggest_categorical(
                "batch_size", [32, 64]
            )
        else:
            batch_size = self.fixed_batch_size
        cfg.batch_size = batch_size
        return cfg, batch_size, lr

    def _resolve_model_spec(
        self, name: str, cfg: TGNNSolvConfig
    ) -> Tuple[TGNNSolvConfig, type, type]:
        if name == "tgnn_solv":
            return cfg, TGNNSolv, TGNNSolvTrainer
        if name == "tgnn_solv_gps":
            cfg = replace(cfg, encoder_type="gps")
            return cfg, TGNNSolv, TGNNSolvTrainer
        if name == "tgnn_solv_descriptors":
            cfg = replace(cfg, use_descriptor_augmentation=True)
            return cfg, TGNNSolv, TGNNSolvTrainer

        from .ablation import (
            TGNNSolvNoCrossAttn,
            TGNNSolvNoNRTL,
            TGNNSolvNoCorrection,
            NoCurriculumTrainer,
            NoAuxLossTrainer,
        )

        if name == "no_cross_attn":
            return cfg, TGNNSolvNoCrossAttn, TGNNSolvTrainer
        if name == "no_nrtl":
            return cfg, TGNNSolvNoNRTL, TGNNSolvTrainer
        if name == "no_curriculum":
            return cfg, TGNNSolv, NoCurriculumTrainer
        if name == "no_aux_losses":
            return cfg, TGNNSolv, NoAuxLossTrainer
        if name == "no_correction":
            return cfg, TGNNSolvNoCorrection, TGNNSolvTrainer
        if name == "no_implicit_diff":
            cfg = replace(cfg, use_implicit_diff=False)
            return cfg, TGNNSolv, TGNNSolvTrainer
        if name == "small_128":
            cfg = replace(
                cfg,
                hidden_dim=128,
                pair_dim=256,
                n_gnn_layers=4,
                n_cross_attn_layers=2,
            )
            return cfg, TGNNSolv, TGNNSolvTrainer
        if name == "large_512":
            cfg = replace(
                cfg,
                hidden_dim=512,
                pair_dim=1024,
                n_gnn_layers=8,
                n_cross_attn_layers=4,
            )
            return cfg, TGNNSolv, TGNNSolvTrainer

        raise ValueError(f"Unknown model: {name}")

    def _tune_arch_for_model(self, model_name: str) -> bool:
        if model_name in ("small_128", "large_512"):
            return False
        return self.tune_arch

    def _clear_memory(self, model: torch.nn.Module) -> None:
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _objective(self, model_name: str, trial: "optuna.Trial") -> float:
        trial_seed = self.seed + trial.number
        self.set_seed(trial_seed)
        tune_arch = self._tune_arch_for_model(model_name)

        if model_name in {"direct_gnn", "direct_gnn_descriptors"}:
            cfg, batch_size, lr = self._suggest_direct_params(
                trial, tune_arch
            )
            if model_name == "direct_gnn_descriptors":
                cfg = replace(cfg, use_descriptor_augmentation=True)
            train_loader, val_loader, _ = self._build_loaders(
                cfg,
                batch_size,
                trial_seed,
                include_test=False,
            )
            if cfg.use_descriptor_augmentation:
                if not isinstance(self.train_df, pd.DataFrame):
                    raise ValueError(
                        "Descriptor augmentation requires access to the training dataframe."
                    )
                descriptor_mean, descriptor_std = compute_descriptor_normalization_stats(
                    pd.concat(
                        [
                            self.train_df["solute_smiles"].astype(str),
                            self.train_df["solvent_smiles"].astype(str),
                        ],
                        axis=0,
                        ignore_index=True,
                    ).tolist()
                )
                cfg = replace(cfg, descriptor_dim=int(descriptor_mean.shape[0]))
            model = DirectGNN(
                cfg=cfg,
            ).to(self.device)
            if cfg.use_descriptor_augmentation:
                model.set_descriptor_normalization(descriptor_mean, descriptor_std)
            trainer = DirectGNNTrainer(model, self.device)
            t0 = time.time()
            trainer.train(
                train_loader,
                val_loader,
                n_epochs=self.baseline_epochs,
                lr=lr,
                patience=self.baseline_patience,
            )
        else:
            cfg, batch_size = self._suggest_tgnn_params(trial, tune_arch)
            cfg, model_cls, trainer_cls = self._resolve_model_spec(
                model_name, cfg
            )
            train_loader, val_loader, _ = self._build_loaders(
                cfg,
                batch_size,
                trial_seed,
                include_test=False,
            )
            descriptor_mean = None
            descriptor_std = None
            if cfg.use_descriptor_augmentation:
                if not isinstance(self.train_df, pd.DataFrame):
                    raise ValueError(
                        "Descriptor augmentation requires access to the training dataframe."
                    )
                descriptor_mean, descriptor_std = compute_descriptor_normalization_stats(
                    pd.concat(
                        [
                            self.train_df["solute_smiles"].astype(str),
                            self.train_df["solvent_smiles"].astype(str),
                        ],
                        axis=0,
                        ignore_index=True,
                    ).tolist()
                )
                cfg = replace(cfg, descriptor_dim=int(descriptor_mean.shape[0]))
            model = model_cls(cfg=cfg).to(self.device)
            if cfg.use_descriptor_augmentation:
                if descriptor_mean is None or descriptor_std is None:
                    raise ValueError(
                        "Descriptor normalization statistics were not computed."
                    )
                model.set_descriptor_normalization(descriptor_mean, descriptor_std)
            trainer = trainer_cls(model, cfg)
            t0 = time.time()
            trainer.train_full(train_loader, val_loader)

        val_mae = self._compute_val_mae(model, val_loader)
        n_params = sum(p.numel() for p in model.parameters())
        train_time = time.time() - t0

        trial.set_user_attr("val_mae", val_mae)
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("train_time_s", train_time)
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("batch_size", batch_size)

        self._clear_memory(model)
        return val_mae

    def run_study(
        self,
        model_name: str,
        n_trials: int,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> "optuna.study.Study":
        self.require_optuna()

        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choices: {AVAILABLE_MODELS}"
            )

        study = optuna.create_study(
            direction="minimize",
            storage=storage,
            study_name=study_name,
            load_if_exists=bool(storage and study_name),
        )
        study.optimize(
            lambda trial: self._objective(model_name, trial),
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
        )
        return study


def load_csv_splits(
    train_csv: str, val_csv: str, test_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return OptunaTuner.load_csv_splits(train_csv, val_csv, test_csv)


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: TGNNSolvConfig | None = None,
    cache: bool = True,
) -> Tuple[TGNNSolvDataset, TGNNSolvDataset, TGNNSolvDataset]:
    return OptunaTuner.build_datasets(
        train_df,
        val_df,
        test_df,
        cfg=cfg,
        cache=cache,
    )
