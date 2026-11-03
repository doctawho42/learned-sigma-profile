#!/usr/bin/env python3
"""
CLI wrapper for training TGNN-Solv with curriculum learning.

Supports:
  - YAML configuration loading and CLI parameter overrides
  - Three-phase training (property pretraining → SLE → fine-tuning)
  - Experiment logging with metrics, artifacts, and model checkpoints
  - Optional test set evaluation
"""

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tgnn_solv.artifacts import build_model_card, build_run_manifest, write_json
from tgnn_solv.config import TGNNSolvConfig
from tgnn_solv.data.dataset import make_loader
from tgnn_solv.device import default_device, resolve_device
from tgnn_solv.experiment_logger import ExperimentLogger
from tgnn_solv.features import (
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    DescriptorNormalizer,
)
from tgnn_solv.group_contribution import (
    GC_FALLBACK_PRIORS,
    compute_gc_priors,
    fit_tm_gc_calibration,
)
from tgnn_solv.model import TGNNSolv
from tgnn_solv.stream_provenance import certify_streams_or_raise
from tgnn_solv.pretrain_pipeline import (
    apply_pretrained_encoder_checkpoint,
    derive_pretrain_checkpoint_path,
    load_pretrained_encoder_checkpoint,
    run_stage0_pretraining,
)
from tgnn_solv.seed import set_seed
from tgnn_solv.trainer import TGNNSolvTrainer


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """Opt-in torch.compile (env TGNN_COMPILE) to cut kernel-launch overhead: the small
    GNN + iterative SLE/COSMO-SAC solver forward is launch-bound on big GPUs. Uses
    in-place model.compile(dynamic=True) so state_dict keys are unchanged (preemption-
    resume safe). The custom IFT-autograd solver may graph-break and stay eager, which
    is safe (the encoder still compiles). Override mode via TGNN_COMPILE_MODE."""
    if device.type != "cuda" or os.environ.get("TGNN_COMPILE", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        return model
    mode = os.environ.get("TGNN_COMPILE_MODE") or "default"
    model.compile(mode=mode, dynamic=True)
    print(f"   torch.compile enabled in-place (mode={mode}, dynamic=True).")
    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TGNN-Solv with curriculum learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_config.yaml",
        help="Path to YAML configuration file",
    )
    
    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation CSV file",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test CSV file (optional; evaluate after training if provided)",
    )
    parser.add_argument(
        "--idac-train-data",
        type=str,
        default=None,
        help=(
            "Optional activity auxiliary training CSV. "
            "Compatible with gamma_inf and finite-composition ln_gamma_2 rows. "
            "Batches from this file are trained through a fast activity-only "
            "path instead of being appended to the main SLE training CSV."
        ),
    )
    
    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Path to save trained model checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved training checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-extend",
        action="store_true",
        help="Load the checkpoint's WEIGHTS but continue/finetune training under a freshly "
             "supplied schedule (--epochs-phase*, --lr, --set ...): apply CLI schedule "
             "overrides on resume and re-run the phase loop from the loaded weights instead "
             "of honouring a 'completed' resume-state. Use to extend or fine-tune a finished "
             "checkpoint (e.g. warm-up -> unfrozen SLE); plain --resume continues an "
             "interrupted run under its saved schedule.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a resumable checkpoint every N epochs per phase (0 disables periodic saves)",
    )
    
    # Training settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Device to use (cuda or cpu); defaults to whichever this box has.",
    )
    
    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Root directory for experiment logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated from timestamp if not provided)",
    )
    
    # CLI overrides for configuration
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override hidden_dim from config",
    )
    parser.add_argument(
        "--n-gnn-layers",
        type=int,
        default=None,
        help="Override n_gnn_layers from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (0 = main process). >0 parallelizes featurization; "
             "persistent workers keep the per-SMILES cache warm across epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (lr_phase2) from config",
    )
    parser.add_argument(
        "--epochs-phase1",
        type=int,
        default=None,
        help="Override Phase 1 epoch count from config.",
    )
    parser.add_argument(
        "--epochs-phase2",
        type=int,
        default=None,
        help="Override Phase 2 epoch count from config.",
    )
    parser.add_argument(
        "--epochs-phase3",
        type=int,
        default=None,
        help="Override Phase 3 epoch count from config.",
    )
    parser.add_argument(
        "--branch-training-mode",
        type=str,
        default=None,
        help=(
            "Override branch training mode from config. "
            "Supported values: standard, coordinate_descent."
        ),
    )
    parser.add_argument(
        "--idac-steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Number of activity-only auxiliary batches per epoch. "
            "Requires --idac-train-data; 0 disables the auxiliary stream."
        ),
    )
    parser.add_argument(
        "--idac-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for --idac-train-data. Defaults to the main batch size. "
            "Use a smaller value on MPS because gamma-only cross-attention still "
            "pads atom sequences across the batch."
        ),
    )
    parser.add_argument(
        "--crystal-train-data",
        type=str,
        default=None,
        help=(
            "Optional external single-component crystal auxiliary CSV "
            "(T_m / dH_fus pool), built by scripts/data/build_crystal_aux_stream.py. "
            "Grounds the crystal branch on a much larger pure-component label pool "
            "than the solubility pairs, enabling scaffold transfer of Phi(T)."
        ),
    )
    parser.add_argument(
        "--crystal-steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Number of external-crystal auxiliary batches per epoch. "
            "Requires --crystal-train-data; 0 disables the auxiliary stream."
        ),
    )
    parser.add_argument(
        "--crystal-batch-size",
        type=int,
        default=None,
        help="Batch size for --crystal-train-data. Defaults to the main batch size.",
    )
    parser.add_argument(
        "--sigma-train-data",
        type=str,
        default=None,
        help=(
            "Optional external sigma-profile auxiliary CSV (COSMO-SAC activity "
            "grounding), built by scripts/data/build_sigma_profile_aux_stream.py. "
            "Grounds the sigma-profile head on a single-component label pool."
        ),
    )
    parser.add_argument(
        "--sigma-steps-per-epoch",
        type=int,
        default=None,
        help="Number of sigma-profile auxiliary batches per epoch (0 disables).",
    )
    parser.add_argument(
        "--sigma-batch-size",
        type=int,
        default=None,
        help="Batch size for --sigma-train-data. Defaults to the main batch size.",
    )
    parser.add_argument(
        "--sigma-val-data",
        type=str,
        default=None,
        help=(
            "Optional sigma-profile validation CSV for warmup early-stop "
            "(mirrors --sigma-train-data format). Used by sigma-warmup pretraining."
        ),
    )
    parser.add_argument(
        "--allow-stream-scaffold-overlap",
        action="store_true",
        help=(
            "Override the fail-closed point-of-use check that no auxiliary grounding "
            "stream shares a canonical SMILES or a Bemis-Murcko scaffold with the "
            "validation/test files this run is scored against. The override is "
            "recorded in the run manifest. Only for pools deliberately not used in a "
            "scaffold-split evaluation."
        ),
    )
    parser.add_argument(
        "--sigma-warmup-epochs",
        type=int,
        default=None,
        help=(
            "Number of sigma-warmup pretraining epochs before the SLE curriculum "
            "(overrides config.sigma_warmup_epochs; 0 disables)."
        ),
    )
    parser.add_argument(
        "--freeze-sigma-head-during-sle",
        action="store_true",
        help=(
            "Freeze head_sigma during SLE phases 2/3 to hold the σ-manifold "
            "established by warmup (sets config.freeze_sigma_head_during_sle=True)."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Generic config overrides applied after the YAML/checkpoint config, "
            "e.g. --set use_gc_priors_crystal=false detach_crystal_params_in_sle=false. "
            "Values are coerced to the dataclass field type. Used by ablation runners."
        ),
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Run Stage 0 pretraining on an external SMILES corpus before standard TGNN training.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=30,
        help="Number of Stage 0 pretraining epochs.",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=128,
        help="Stage 0 pretraining batch size.",
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=3.0e-4,
        help="Learning rate for Stage 0 pretraining.",
    )
    parser.add_argument(
        "--pretrain-data",
        type=str,
        default="zinc250k",
        help="Stage 0 pretraining source (`zinc250k` or a local CSV/text file with SMILES).",
    )
    parser.add_argument(
        "--pretrain-max-molecules",
        type=int,
        default=None,
        help="Optional cap on the number of Stage 0 molecules to load.",
    )
    parser.add_argument(
        "--pretrain-pairwise-contrastive-data",
        type=str,
        default=None,
        help=(
            "Optional pairwise solubility-contrastive CSV produced by "
            "scripts/data/build_pairwise_contrastive_pretrain.py."
        ),
    )
    parser.add_argument(
        "--pretrain-pairwise-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Stage 0 weight for pairwise compatibility BCE. "
            "Requires --pretrain-pairwise-contrastive-data."
        ),
    )
    parser.add_argument(
        "--pretrain-pairwise-contrastive-batch-size",
        type=int,
        default=None,
        help="Optional Stage 0 batch size for pairwise contrastive rows.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default=None,
        help="Load a previously saved Stage 0 encoder/readout checkpoint instead of rerunning pretraining.",
    )
    parser.add_argument(
        "--pretrain-output",
        type=str,
        default=None,
        help="Where to save the Stage 0 encoder/readout checkpoint when `--pretrain` is used.",
    )
    parser.add_argument(
        "--run-descriptor-probe",
        action="store_true",
        help="After training, run the existing Ridge linear probe from `g_sol` to RDKit descriptors.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Skip training and run only the descriptor probe for --checkpoint.",
    )
    parser.add_argument(
        "--descriptor-probe-output-dir",
        type=str,
        default=None,
        help="Output directory for descriptor probe artifacts. Defaults to `<checkpoint_stem>_descriptor_probe`.",
    )
    parser.add_argument(
        "--descriptor-probe-device",
        type=str,
        default="cpu",
        help="Device for the descriptor probe. Defaults to CPU to avoid contending with training.",
    )
    parser.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help=(
            "During training, run the descriptor linear probe every N epochs "
            "for the selected phases (0 disables probe evolution tracking)."
        ),
    )
    parser.add_argument(
        "--probe-phases",
        type=str,
        default="2",
        help="Comma-separated phases for --probe-every, for example '2' or '1,2,3'.",
    )
    parser.add_argument(
        "--probe-evolution-output",
        type=str,
        default=None,
        help="CSV path for in-training probe evolution. Defaults to <checkpoint>_probe_evolution.csv.",
    )
    
    return parser.parse_args()


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Atomically save a checkpoint to avoid partial files on preemption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
        suffix=".pt",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def build_checkpoint_payload(
    *,
    model: TGNNSolv,
    config: TGNNSolvConfig,
    seed: int,
    experiment_name: str,
    trainer: TGNNSolvTrainer,
    resume_state: dict | None,
    pretrain_info: dict[str, Any] | None = None,
) -> dict:
    """Build a training checkpoint that supports both inference and resume."""
    checkpoint = {
        "model_state": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(config),
        "node_feat_dim": int(getattr(model, "node_feat_dim", NODE_FEAT_DIM)),
        "edge_feat_dim": int(getattr(model, "edge_feat_dim", EDGE_FEAT_DIM)),
        "seed": seed,
        "experiment_name": experiment_name,
        "trainer_state_dict": trainer.state_dict(),
        "resume_state": resume_state,
        "pretrain_info": pretrain_info,
    }
    if config.use_descriptor_augmentation:
        checkpoint["descriptor_mean"] = model.descriptor_mean.detach().cpu()
        checkpoint["descriptor_std"] = model.descriptor_std.detach().cpu()
    return checkpoint


def maybe_run_descriptor_probe(
    *,
    checkpoint_path: Path,
    train_data: str,
    test_data: str | None,
    output_dir: str | None,
    device: str,
    logger: ExperimentLogger,
) -> None:
    """Run the existing `g_sol -> descriptors` Ridge probe and log its summary."""
    if test_data is None:
        print("   Skipping descriptor probe because no --test-data path was provided.")
        return

    from probe_gsol_descriptor_recovery import (
        CORE_DESCRIPTOR_NAMES,
        build_descriptor_matrix,
        build_markdown_report,
        extract_solute_embeddings,
        fit_descriptor_probes,
        load_unique_solutes,
    )
    from tgnn_solv.inference import load_model
    from tgnn_solv.reporting import json_safe

    probe_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_descriptor_probe")
    )
    probe_output_dir.mkdir(parents=True, exist_ok=True)

    # Default allow_tf32_from_env=False, as before: this call used to borrow
    # probe_gsol_descriptor_recovery's resolver, which never touched TGNN_MATMUL_TF32.
    # (If the training run above enabled TF32, that process-global setting stands; this
    # only declines to re-apply and re-announce it.)
    probe_device = resolve_device(device)
    model, cfg = load_model(str(checkpoint_path), device=probe_device)
    model.eval()

    train_path = Path(train_data).resolve()
    test_path = Path(test_data).resolve()
    train_solutes = load_unique_solutes(train_path)
    test_solutes = load_unique_solutes(test_path)

    train_smiles, X_train = extract_solute_embeddings(
        model=model,
        smiles_list=train_solutes,
        device=probe_device,
        batch_size=512,
        temperature_K=298.15,
    )
    test_smiles, X_test = extract_solute_embeddings(
        model=model,
        smiles_list=test_solutes,
        device=probe_device,
        batch_size=512,
        temperature_K=298.15,
    )
    Y_train = build_descriptor_matrix(train_smiles)
    Y_test = build_descriptor_matrix(test_smiles)
    results_df, probe_summary = fit_descriptor_probes(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        alpha=1.0,
    )

    core_df = (
        results_df[results_df["descriptor"].isin(CORE_DESCRIPTOR_NAMES)]
        .set_index("descriptor")
        .reindex(CORE_DESCRIPTOR_NAMES)
        .reset_index()
    )

    import math
    import numpy as np

    np.savez_compressed(
        probe_output_dir / "train_solute_embeddings.npz",
        smiles=np.asarray(train_smiles, dtype=object),
        embeddings=X_train.astype(np.float32, copy=False),
    )
    np.savez_compressed(
        probe_output_dir / "test_solute_embeddings.npz",
        smiles=np.asarray(test_smiles, dtype=object),
        embeddings=X_test.astype(np.float32, copy=False),
    )
    results_df.to_csv(probe_output_dir / "descriptor_r2.csv", index=False)

    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "config": {
            "hidden_dim": int(cfg.hidden_dim),
            "encoder_type": str(getattr(cfg, "encoder_type", "mpnn")),
            "use_temperature_in_encoder": bool(cfg.use_temperature_in_encoder),
            "use_morgan_features": bool(cfg.use_morgan_features),
            "use_descriptor_augmentation": bool(cfg.use_descriptor_augmentation),
            "encoder_role_mode": str(cfg.encoder_role_mode),
        },
        "probe": {
            "embedding_name": "g_sol_pre",
            "embedding_temperature_K": 298.15,
            "ridge_alpha": 1.0,
            "device": str(probe_device),
        },
        "dataset": {
            "train_data": str(train_path),
            "test_data": str(test_path),
            "n_train_unique_solutes": int(len(train_smiles)),
            "n_test_unique_solutes": int(len(test_smiles)),
        },
        "summary": probe_summary,
        "core_descriptors": {
            row["descriptor"]: {
                "r2_test": (
                    None if not math.isfinite(float(row["r2_test"])) else float(row["r2_test"])
                ),
                "r2_train": (
                    None if not math.isfinite(float(row["r2_train"])) else float(row["r2_train"])
                ),
                "status": str(row["status"]),
            }
            for _, row in core_df.iterrows()
        },
    }
    (probe_output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary_payload), indent=2),
        encoding="utf-8",
    )
    (probe_output_dir / "report.md").write_text(
        build_markdown_report(
            summary=probe_summary,
            results_df=results_df,
            core_df=core_df,
            checkpoint_path=checkpoint_path,
            device=str(probe_device),
            train_count=len(train_smiles),
            test_count=len(test_smiles),
        ),
        encoding="utf-8",
    )
    logger.log_artifact("descriptor_probe_summary", summary_payload)
    print(f"   Descriptor probe written to {probe_output_dir}")


def parse_probe_phases(raw: str) -> set[int]:
    """Parse a comma-separated phase list for in-training descriptor probes."""
    phases: set[int] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        phase = int(item)
        if phase not in {1, 2, 3}:
            raise ValueError("--probe-phases may only contain 1, 2, and 3.")
        phases.add(phase)
    return phases


def maybe_run_probe_evolution_snapshot(
    *,
    model: TGNNSolv,
    config: TGNNSolvConfig,
    phase: int,
    epoch_in_phase: int,
    global_epoch: int,
    train_data: str,
    test_data: str | None,
    output_csv: Path,
    output_root: Path,
    logger: ExperimentLogger,
) -> None:
    """Run a descriptor linear probe on the live model and append one CSV row."""
    if test_data is None:
        print("   Skipping probe evolution snapshot because no --test-data path was provided.")
        return

    from probe_gsol_descriptor_recovery import (
        CORE_DESCRIPTOR_NAMES,
        build_descriptor_matrix,
        extract_solute_embeddings,
        fit_descriptor_probes,
        load_unique_solutes,
    )
    from tgnn_solv.reporting import json_safe

    probe_device = next(model.parameters()).device
    snapshot_dir = output_root / f"phase{phase}_epoch{epoch_in_phase + 1:04d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        train_path = Path(train_data).resolve()
        test_path = Path(test_data).resolve()
        train_solutes = load_unique_solutes(train_path)
        test_solutes = load_unique_solutes(test_path)

        train_smiles, X_train = extract_solute_embeddings(
            model=model,
            smiles_list=train_solutes,
            device=probe_device,
            batch_size=512,
            temperature_K=298.15,
        )
        test_smiles, X_test = extract_solute_embeddings(
            model=model,
            smiles_list=test_solutes,
            device=probe_device,
            batch_size=512,
            temperature_K=298.15,
        )
        Y_train = build_descriptor_matrix(train_smiles)
        Y_test = build_descriptor_matrix(test_smiles)
        results_df, probe_summary = fit_descriptor_probes(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            alpha=1.0,
        )
    finally:
        model.train(was_training)

    results_df.to_csv(snapshot_dir / "descriptor_r2.csv", index=False)
    summary_payload = {
        "phase": int(phase),
        "epoch_in_phase": int(epoch_in_phase + 1),
        "global_epoch": int(global_epoch),
        "config": {
            "encoder_type": str(getattr(config, "encoder_type", "mpnn")),
            "hidden_dim": int(config.hidden_dim),
        },
        "summary": probe_summary,
    }
    (snapshot_dir / "summary.json").write_text(
        json.dumps(json_safe(summary_payload), indent=2),
        encoding="utf-8",
    )

    core_df = (
        results_df[results_df["descriptor"].isin(CORE_DESCRIPTOR_NAMES)]
        .set_index("descriptor")
    )
    tracked = ("MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "FractionCSP3", "MolWt")
    row: dict[str, float | int | str | None] = {
        "global_epoch": int(global_epoch),
        "phase": int(phase),
        "epoch_in_phase": int(epoch_in_phase + 1),
        "median_R2": probe_summary.get("median_r2_test"),
        "mean_R2": probe_summary.get("mean_r2_test"),
        "n_descriptors": probe_summary.get("n_descriptors_with_finite_r2"),
    }
    for descriptor in tracked:
        if descriptor in core_df.index:
            value = core_df.loc[descriptor, "r2_test"]
            row[f"{descriptor}_R2"] = None if pd.isna(value) else float(value)
        else:
            row[f"{descriptor}_R2"] = None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if output_csv.exists():
        row_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        row_df.to_csv(output_csv, index=False)

    logger.log_artifact(f"probe_evolution_phase{phase}_epoch{epoch_in_phase + 1}", row)
    median = row.get("median_R2")
    median_str = "NA" if median is None else f"{float(median):.3f}"
    print(
        "   Probe evolution snapshot: "
        f"phase={phase}, epoch={epoch_in_phase + 1}, "
        f"global={global_epoch}, median_R2={median_str}"
    )


def load_data(
    csv_path: str,
    config: TGNNSolvConfig,
    *,
    shuffle: bool,
    seed: int,
    batch_size: int | None = None,
) -> DataLoader:
    """Load a CSV split and build the configured DataLoader."""
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)

    return make_loader(
        df,
        batch_size=int(batch_size or config.batch_size),
        shuffle=shuffle,
        num_workers=config.num_workers,
        cache=True,
        use_pair_temperature_batching=(
            shuffle and config.use_pair_temperature_batching
        ),
        pair_temperature_min_group_size=(
            config.pair_temperature_min_group_size
        ),
        pair_temperature_group_chunk_size=(
            config.pair_temperature_group_chunk_size
        ),
        use_morgan_features=config.use_morgan_features,
        morgan_radius=config.morgan_radius,
        morgan_n_bits=config.morgan_n_bits,
        use_descriptor_augmentation=config.use_descriptor_augmentation,
        use_ionic_features=config.use_ionic_features,
        use_descriptor_priors=config.use_descriptor_priors,
        use_group_priors=config.requires_group_prior_features,
        use_gc_priors_crystal=config.use_gc_priors_crystal,
        use_gasteiger_charges=config.use_gasteiger_charges,
        use_phys_edge_features=config.use_phys_edge_features,
        explicit_h_small_molecules=config.explicit_h_small_molecules,
        explicit_h_max_heavy_atoms=config.explicit_h_max_heavy_atoms,
        use_pseudo_hansen=(
            (config.use_hansen_contrastive or config.use_hansen_delta_loss)
            and config.use_pseudo_hansen
        ),
        pseudo_hansen_weight_discount=config.pseudo_hansen_weight_discount,
        source_uncertainty_csv=(
            config.source_uncertainty_csv
            if config.use_source_uncertainty_weights
            else ""
        ),
        source_uncertainty_weight_mode=config.source_uncertainty_weight_mode,
        source_uncertainty_default_sigma_ln_x2=config.source_uncertainty_default_sigma_ln_x2,
        source_uncertainty_min_sigma_ln_x2=config.source_uncertainty_min_sigma_ln_x2,
        source_uncertainty_min_weight=config.source_uncertainty_min_weight,
        source_uncertainty_max_weight=config.source_uncertainty_max_weight,
        expected_sigma_bins=getattr(config, "cosmo_sac_n_bins", None),
        seed=seed,
    )


def _base_scalar_type(ftype: object) -> object:
    """Base scalar of a field annotation, unwrapping ``Optional[X]``/``X | None``.

    ``f.type`` is a real typing object here (config.py does not use PEP 563), so
    ``Optional[float].__args__ == (float, NoneType)``. Returns the non-None member
    (e.g. ``float``) so coercion keys off the DECLARED type rather than the current
    value -- the latter is ``None`` for every ``Optional`` field, which previously
    made ``--set`` store such fields as raw strings.
    """
    args = getattr(ftype, "__args__", ())
    non_none = [a for a in args if a is not type(None)]
    return non_none[0] if non_none else ftype


def apply_set_overrides(
    config: TGNNSolvConfig,
    set_overrides: list[str] | None,
) -> None:
    """Apply generic ``--set key=value`` overrides, coercing to field types."""
    if not set_overrides:
        return
    field_types = {f.name: f.type for f in dataclasses.fields(config)}
    for item in set_overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key not in field_types:
            raise ValueError(f"--set unknown config field: {key!r}")
        base = _base_scalar_type(field_types[key])
        if base is bool:
            value: object = raw.lower() in {"true", "1", "yes", "on"}
        elif base is int:
            value = int(raw)
        elif base is float:
            value = float(raw)
        else:
            value = raw
        setattr(config, key, value)
        print(f"   --set override: {key} = {value!r}")


def maybe_fit_gc_tm_calibration(
    train_csv_path: str,
    config: TGNNSolvConfig,
) -> None:
    """Fit and store a train-only affine calibration for the GC melting prior."""
    if not config.use_gc_priors_crystal:
        return

    import numpy as np
    import pandas as pd

    df = pd.read_csv(train_csv_path, low_memory=False)
    required_cols = {"solute_smiles", "T_m", "has_T_m"}
    if not required_cols.issubset(df.columns):
        print(
            "   Skipping GC T_m calibration; "
            f"missing columns: {sorted(required_cols - set(df.columns))}"
        )
        return

    unique_tm = (
        df.loc[df["has_T_m"].fillna(False).astype(bool), ["solute_smiles", "T_m"]]
        .drop_duplicates(subset=["solute_smiles"], keep="first")
    )
    if unique_tm.empty:
        print("   Skipping GC T_m calibration; no training rows with T_m labels.")
        return

    raw_tm_gc: list[float] = []
    tm_true: list[float] = []
    fallback_count = 0
    for smiles, tm_value in unique_tm.itertuples(index=False):
        priors = compute_gc_priors(str(smiles))
        tm_gc = priors["T_m_gc"]
        if tm_gc is None or not np.isfinite(tm_gc) or not np.isfinite(tm_value):
            continue
        raw_tm_gc.append(float(tm_gc))
        tm_true.append(float(tm_value))
        if abs(float(tm_gc) - GC_FALLBACK_PRIORS["T_m_gc"]) < 1.0e-6:
            fallback_count += 1

    if not raw_tm_gc:
        print("   Skipping GC T_m calibration; no usable GC priors in training split.")
        return

    scale, bias = fit_tm_gc_calibration(raw_tm_gc, tm_true)
    config.gc_prior_tm_scale = scale
    config.gc_prior_tm_bias = bias

    raw_arr = np.asarray(raw_tm_gc, dtype=float)
    true_arr = np.asarray(tm_true, dtype=float)
    calibrated_arr = scale * raw_arr + bias
    raw_mae = float(np.mean(np.abs(raw_arr - true_arr)))
    calibrated_mae = float(np.mean(np.abs(calibrated_arr - true_arr)))
    fallback_frac = fallback_count / len(raw_tm_gc)

    print(
        "   GC T_m calibration: "
        f"n_unique={len(raw_tm_gc)}, "
        f"fallback={fallback_frac:.1%}, "
        f"raw_mae={raw_mae:.2f} K, "
        f"calibrated_mae={calibrated_mae:.2f} K, "
        f"scale={scale:.6f}, "
        f"bias={bias:.3f}"
    )


def main() -> None:
    """Main training pipeline."""
    try:
        # Parse arguments
        args = parse_args()
        if args.pretrain and args.pretrain_checkpoint is not None:
            raise ValueError(
                "Use either --pretrain or --pretrain-checkpoint, not both."
            )
        if args.probe_only and not args.run_descriptor_probe:
            args.run_descriptor_probe = True
        if not args.probe_only and args.val_data is None:
            raise ValueError("--val-data is required unless --probe-only is used.")
        if args.probe_only and args.test_data is None:
            raise ValueError("--test-data is required for --probe-only descriptor probes.")
        
        print("=" * 70)
        print("TGNN-Solv Training Pipeline")
        print("=" * 70)
        
        # Load configuration from YAML
        resume_checkpoint = None
        pretrain_info: dict[str, Any] | None = None
        if args.resume is not None:
            print(f"\n1. Loading resume checkpoint from {args.resume}...")
            resume_checkpoint = torch.load(args.resume, map_location="cpu")
            config = TGNNSolvConfig(**resume_checkpoint["config"])
            args.seed = int(resume_checkpoint.get("seed", args.seed))
            pretrain_info = resume_checkpoint.get("pretrain_info")
        else:
            print(f"\n1. Loading configuration from {args.config}...")
            config = TGNNSolvConfig.from_yaml(args.config)
        # allow_tf32_from_env: this is a training entry point, so TGNN_MATMUL_TF32 may
        # switch the process to TF32 matmul. Evaluation callers leave it off.
        device = resolve_device(args.device, allow_tf32_from_env=True)
        descriptor_mean = None
        descriptor_std = None
        if config.use_descriptor_augmentation:
            if resume_checkpoint is not None:
                descriptor_mean = resume_checkpoint.get("descriptor_mean")
                descriptor_std = resume_checkpoint.get("descriptor_std")
            if descriptor_mean is None or descriptor_std is None:
                normalizer = DescriptorNormalizer().fit(
                    pd.read_csv(args.train_data, low_memory=False)
                )
                descriptor_mean = normalizer.mean
                descriptor_std = normalizer.std
            config.descriptor_dim = int(torch.as_tensor(descriptor_mean).shape[0])
            print(
                "   Descriptor augmentation enabled: "
                f"{config.descriptor_dim} RDKit descriptors."
            )

        # Apply CLI overrides. Resumed runs keep model-shape/training-schedule
        # settings from the checkpoint, but data-loader knobs like batch size
        # are safe to override to fit the available device.
        if resume_checkpoint is None:
            if args.hidden_dim is not None:
                config.hidden_dim = args.hidden_dim
            if args.n_gnn_layers is not None:
                config.n_gnn_layers = args.n_gnn_layers
            if args.batch_size is not None:
                config.batch_size = args.batch_size
            if args.num_workers is not None:
                config.num_workers = int(args.num_workers)
            if args.lr is not None:
                config.lr_phase2 = args.lr
            if args.epochs_phase1 is not None:
                config.epochs_phase1 = int(args.epochs_phase1)
            if args.epochs_phase2 is not None:
                config.epochs_phase2 = int(args.epochs_phase2)
            if args.epochs_phase3 is not None:
                config.epochs_phase3 = int(args.epochs_phase3)
            if args.branch_training_mode is not None:
                config.branch_training_mode = str(args.branch_training_mode)
            if args.idac_steps_per_epoch is not None:
                config.idac_aux_steps_per_epoch = int(args.idac_steps_per_epoch)
            if args.crystal_steps_per_epoch is not None:
                config.crystal_aux_steps_per_epoch = int(args.crystal_steps_per_epoch)
            if args.sigma_steps_per_epoch is not None:
                config.sigma_aux_steps_per_epoch = int(args.sigma_steps_per_epoch)
            if args.sigma_warmup_epochs is not None:
                config.sigma_warmup_epochs = args.sigma_warmup_epochs
            if args.freeze_sigma_head_during_sle:
                config.freeze_sigma_head_during_sle = True
            if args.probe_every > 0:
                config.probe_every = int(args.probe_every)
        elif any(
            override is not None
            for override in (
                args.hidden_dim,
                args.n_gnn_layers,
                args.lr,
                args.epochs_phase1,
                args.epochs_phase2,
                args.epochs_phase3,
                args.branch_training_mode,
                args.idac_steps_per_epoch,
                args.idac_batch_size,
            )
        ):
            if not args.resume_extend:
                print(
                    "   Ignoring CLI config overrides for resumed training "
                    "(pass --resume-extend to apply them); using the checkpoint's saved config."
                )
        # --resume-extend: re-apply the CLI training schedule on top of the loaded weights so a
        # finished checkpoint can be extended / fine-tuned (e.g. warm-up -> unfrozen SLE).
        if resume_checkpoint is not None and args.resume_extend:
            if args.epochs_phase1 is not None:
                config.epochs_phase1 = int(args.epochs_phase1)
            if args.epochs_phase2 is not None:
                config.epochs_phase2 = int(args.epochs_phase2)
            if args.epochs_phase3 is not None:
                config.epochs_phase3 = int(args.epochs_phase3)
            if args.lr is not None:
                config.lr_phase2 = args.lr
            if args.branch_training_mode is not None:
                config.branch_training_mode = str(args.branch_training_mode)
            if args.sigma_steps_per_epoch is not None:
                config.sigma_aux_steps_per_epoch = int(args.sigma_steps_per_epoch)
            if args.crystal_steps_per_epoch is not None:
                config.crystal_aux_steps_per_epoch = int(args.crystal_steps_per_epoch)
            if args.idac_steps_per_epoch is not None:
                config.idac_aux_steps_per_epoch = int(args.idac_steps_per_epoch)
            if args.sigma_warmup_epochs is not None:
                config.sigma_warmup_epochs = args.sigma_warmup_epochs
            if args.freeze_sigma_head_during_sle:
                config.freeze_sigma_head_during_sle = True
            print(
                "   --resume-extend: continuing from loaded weights under the CLI schedule "
                f"(epochs {config.epochs_phase1}/{config.epochs_phase2}/{config.epochs_phase3})."
            )
        if resume_checkpoint is not None and args.batch_size is not None:
            config.batch_size = args.batch_size
            print(
                "   Applying resume-time batch_size override for data loading: "
                f"{config.batch_size}"
            )
        if resume_checkpoint is not None and args.probe_every > 0:
            config.probe_every = int(args.probe_every)
        if resume_checkpoint is not None and args.sigma_steps_per_epoch is not None:
            config.sigma_aux_steps_per_epoch = int(args.sigma_steps_per_epoch)
        if resume_checkpoint is not None and args.crystal_steps_per_epoch is not None:
            config.crystal_aux_steps_per_epoch = int(args.crystal_steps_per_epoch)
        if resume_checkpoint is not None and args.idac_steps_per_epoch is not None:
            config.idac_aux_steps_per_epoch = int(args.idac_steps_per_epoch)
            print(
                "   Applying resume-time IDAC auxiliary step override: "
                f"{config.idac_aux_steps_per_epoch}"
            )

        apply_set_overrides(config, args.set_overrides)

        if resume_checkpoint is None:
            maybe_fit_gc_tm_calibration(args.train_data, config)
        
        print(f"   Device: {device}")
        print(f"   Seed: {args.seed}")
        print(
            "   Epoch budget: "
            f"{config.epochs_phase1}/{config.epochs_phase2}/{config.epochs_phase3}"
        )
        print(f"   Branch training mode: {config.branch_training_mode}")
        
        # Set random seed
        print("\n2. Setting random seed...")
        set_seed(args.seed, deterministic=True)
        
        # Create experiment logger
        print("\n3. Initializing experiment logger...")
        experiment_name = args.experiment_name
        if experiment_name is None and resume_checkpoint is not None:
            experiment_name = resume_checkpoint.get("experiment_name")
        logger = ExperimentLogger(args.log_dir, experiment_name)
        logger.log_config(config)
        print(f"   Experiment: {logger.experiment_name}")
        print(f"   Log directory: {logger.exp_dir}")

        if args.probe_only:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint for --probe-only not found: {checkpoint_path}")
            print("\n4. Running descriptor probe only...")
            maybe_run_descriptor_probe(
                checkpoint_path=checkpoint_path,
                train_data=args.train_data,
                test_data=args.test_data,
                output_dir=args.descriptor_probe_output_dir,
                device=args.descriptor_probe_device,
                logger=logger,
            )
            logger.finalize(None)
            print("\nDescriptor probe completed.")
            return
        
        # Load datasets
        print("\n4. Loading datasets...")
        print("   Train:")
        train_loader = load_data(
            args.train_data,
            config,
            shuffle=True,
            seed=args.seed,
        )
        print("   Val:")
        val_loader = load_data(
            args.val_data,
            config,
            shuffle=False,
            seed=args.seed,
        )
        idac_train_loader = None
        if args.idac_train_data is not None:
            if config.idac_aux_steps_per_epoch <= 0:
                print(
                    "   IDAC auxiliary data was provided, but "
                    "idac_aux_steps_per_epoch <= 0; auxiliary stream is disabled."
                )
            else:
                print("   IDAC auxiliary train:")
                idac_train_loader = load_data(
                    args.idac_train_data,
                    config,
                    shuffle=True,
                    seed=args.seed + 17,
                    batch_size=args.idac_batch_size,
                )

        crystal_train_loader = None
        if args.crystal_train_data is not None:
            if config.crystal_aux_steps_per_epoch <= 0:
                print(
                    "   Crystal auxiliary data was provided, but "
                    "crystal_aux_steps_per_epoch <= 0; auxiliary stream is disabled."
                )
            else:
                print("   External crystal auxiliary train:")
                crystal_train_loader = load_data(
                    args.crystal_train_data,
                    config,
                    shuffle=True,
                    seed=args.seed + 23,
                    batch_size=args.crystal_batch_size,
                )

        # Point-of-use certification of every grounding stream, before a GPU-hour is
        # spent on it. The builder's scaffold guard is a property of the builder; this
        # is a property of THIS run, checked against the very val/test files this run
        # will be scored against. Fail-closed; the records travel into the manifest.
        held_out_for_streams = [p for p in (args.val_data, args.test_data) if p]
        stream_provenance: list[dict[str, Any]] = []
        if held_out_for_streams:
            print("\n4b. Certifying grounding streams against the held-out splits...")
            stream_provenance = certify_streams_or_raise(
                [
                    ("sigma_train", args.sigma_train_data),
                    ("sigma_val", args.sigma_val_data),
                    ("crystal_train", args.crystal_train_data),
                    ("idac_train", args.idac_train_data),
                ],
                held_out_for_streams,
                allow_overlap=bool(args.allow_stream_scaffold_overlap),
            )
            for record in stream_provenance:
                if not record.get("present"):
                    print(f"   {record['role']}: none (this arm carries no such stream)")
                    continue
                print(
                    f"   {record['role']}: {record['n_rows']} rows "
                    f"sha256={record['sha256'][:16]}... "
                    f"leak by SMILES={record['leak_by_canonical_smiles']} "
                    f"by scaffold={record['leak_by_murcko_scaffold']}"
                    + ("" if record["certified_no_leak"] else "  [OVERRIDDEN]")
                )

        sigma_train_loader = None
        if args.sigma_train_data is not None:
            if config.sigma_aux_steps_per_epoch <= 0:
                print(
                    "   Sigma-profile auxiliary data was provided, but "
                    "sigma_aux_steps_per_epoch <= 0; auxiliary stream is disabled."
                )
            else:
                print("   External sigma-profile auxiliary train:")
                sigma_train_loader = load_data(
                    args.sigma_train_data,
                    config,
                    shuffle=True,
                    seed=args.seed + 29,
                    batch_size=args.sigma_batch_size,
                )

        sigma_val_loader = None
        if args.sigma_val_data:
            print("   External sigma-profile auxiliary val:")
            sigma_val_loader = load_data(
                args.sigma_val_data,
                config,
                shuffle=False,
                seed=args.seed + 31,
                batch_size=args.sigma_batch_size,
            )

        # Initialize model
        print("\n5. Initializing model...")
        model = TGNNSolv(cfg=config).to(device)
        model = maybe_compile_model(model, device)
        if resume_checkpoint is not None:
            if args.pretrain or args.pretrain_checkpoint is not None:
                print(
                    "   Ignoring Stage 0 options because --resume loads a full model checkpoint."
                )
            saved_state = resume_checkpoint.get(
                "model_state_dict",
                resume_checkpoint["model_state"],
            )
            # A fork may deliberately add parameters the parent never had (e.g. turning
            # cosmo_sac_kernel_residual_rank 0 -> 1 adds kernel_B/kernel_a). Load by
            # presence+shape and report the difference, so a genuine architecture
            # mismatch cannot slip through as a silent strict=False.
            current_state = model.state_dict()
            compatible = {
                k: v
                for k, v in saved_state.items()
                if k in current_state and tuple(current_state[k].shape) == tuple(v.shape)
            }
            fresh = [k for k in current_state if k not in compatible]
            dropped = [k for k in saved_state if k not in compatible]
            if not fresh and not dropped:
                model.load_state_dict(saved_state)
            else:
                if fresh:
                    n_fresh = sum(current_state[k].numel() for k in fresh)
                    print(
                        f"   Fork: {len(fresh)} parameter tensor(s) "
                        f"({n_fresh:,} values) absent from the checkpoint, "
                        f"freshly initialized: {fresh}"
                    )
                if dropped:
                    print(
                        f"   WARNING: {len(dropped)} checkpoint tensor(s) do not fit "
                        f"this model and were DROPPED (name or shape mismatch): {dropped}"
                    )
                model.load_state_dict(compatible, strict=False)
        if config.use_descriptor_augmentation:
            if descriptor_mean is None or descriptor_std is None:
                raise ValueError(
                    "Descriptor normalization statistics were not computed."
                )
            model.set_descriptor_normalization(descriptor_mean, descriptor_std)
        logger.log_model_summary(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        if resume_checkpoint is None:
            if args.pretrain_checkpoint is not None:
                print("\n5a. Loading Stage 0 encoder/readout checkpoint...")
                loaded_pretrain = load_pretrained_encoder_checkpoint(
                    args.pretrain_checkpoint,
                    map_location="cpu",
                )
                pretrain_info = apply_pretrained_encoder_checkpoint(
                    model,
                    loaded_pretrain,
                    strict=True,
                )
                pretrain_info["checkpoint_path"] = str(
                    Path(args.pretrain_checkpoint).expanduser().resolve()
                )
                print(
                    "   Loaded pretrained encoder from "
                    f"{pretrain_info['checkpoint_path']}"
                )
            elif args.pretrain:
                print("\n5a. Running Stage 0 pretraining...")
                pretrain_output = (
                    Path(args.pretrain_output).expanduser().resolve()
                    if args.pretrain_output is not None
                    else derive_pretrain_checkpoint_path(args.checkpoint)
                )
                pretrain_info = run_stage0_pretraining(
                    model,
                    config,
                    device=device,
                    pretrain_source=args.pretrain_data,
                    pretrain_epochs=args.pretrain_epochs,
                    pretrain_batch_size=args.pretrain_batch_size,
                    pretrain_lr=args.pretrain_lr,
                    pretrain_max_molecules=args.pretrain_max_molecules,
                    pairwise_contrastive_csv=args.pretrain_pairwise_contrastive_data,
                    pairwise_contrastive_weight=(
                        args.pretrain_pairwise_contrastive_weight
                    ),
                    pairwise_contrastive_batch_size=(
                        args.pretrain_pairwise_contrastive_batch_size
                    ),
                    save_path=pretrain_output,
                )
                print(
                    "   Stage 0 complete: "
                    f"{pretrain_info['smiles_count']:,} molecules, "
                    f"{pretrain_info['epochs']} epochs."
                )
                if pretrain_info.get("checkpoint_path") is not None:
                    print(
                        "   Saved pretrained encoder to "
                        f"{pretrain_info['checkpoint_path']}"
                    )
            if pretrain_info is not None:
                logger.log_artifact(
                    "pretrain_summary",
                    {
                        key: value
                        for key, value in pretrain_info.items()
                        if key != "history"
                    },
                )

        # Initialize trainer
        print("\n6. Initializing trainer...")
        trainer = TGNNSolvTrainer(model, config)
        if resume_checkpoint is not None:
            trainer.load_state_dict(
                resume_checkpoint.get("trainer_state_dict")
            )

        # Train model
        print("\n7. Starting training (3-phase curriculum)...")
        checkpoint_path = Path(args.checkpoint)

        def maybe_save_resume_checkpoint(state: dict) -> None:
            if args.checkpoint_every <= 0:
                return
            next_epoch = int(state["next_epoch_in_phase"])
            phase_epochs = int(state["phase_epochs"])
            if next_epoch % args.checkpoint_every != 0 and next_epoch < phase_epochs:
                return
            payload = build_checkpoint_payload(
                model=model,
                config=config,
                seed=args.seed,
                experiment_name=logger.experiment_name,
                trainer=trainer,
                resume_state=state,
                pretrain_info=pretrain_info,
            )
            atomic_torch_save(payload, checkpoint_path)
            print(
                "  Saved resume checkpoint "
                f"(phase {state['phase']} epoch {next_epoch}/{phase_epochs}) "
                f"to {checkpoint_path}"
            )

        probe_phases = parse_probe_phases(args.probe_phases)
        probe_evolution_csv = (
            Path(args.probe_evolution_output).expanduser().resolve()
            if args.probe_evolution_output is not None
            else checkpoint_path.with_name(f"{checkpoint_path.stem}_probe_evolution.csv")
        )
        probe_evolution_root = probe_evolution_csv.with_suffix("")

        def on_epoch_end(state: dict) -> None:
            maybe_save_resume_checkpoint(state)
            probe_every = int(getattr(config, "probe_every", 0))
            if probe_every <= 0:
                return
            phase = int(state["phase"])
            next_epoch = int(state["next_epoch_in_phase"])
            if phase not in probe_phases or next_epoch <= 0:
                return
            if next_epoch % probe_every != 0:
                return
            phase_offsets = {
                1: 0,
                2: int(config.epochs_phase1),
                3: int(config.epochs_phase1 + config.epochs_phase2),
            }
            maybe_run_probe_evolution_snapshot(
                model=model,
                config=config,
                phase=phase,
                epoch_in_phase=next_epoch - 1,
                global_epoch=phase_offsets[phase] + next_epoch,
                train_data=args.train_data,
                test_data=args.test_data,
                output_csv=probe_evolution_csv,
                output_root=probe_evolution_root,
                logger=logger,
            )

        resume_state = (
            resume_checkpoint.get("resume_state")
            if resume_checkpoint is not None
            else None
        )
        if args.resume_extend:
            # continue/finetune: ignore the checkpoint's phase/epoch cursor and 'completed'
            # status, running the phase loop fresh under the CLI schedule from the loaded weights.
            resume_state = None
        if resume_state and resume_state.get("status") == "completed":
            print("   Resume checkpoint already marks training as completed; skipping fit.")
        else:
            if getattr(config, "sigma_warmup_epochs", 0) > 0 and sigma_train_loader is not None:
                if resume_checkpoint is not None:
                    print("Skipping sigma-warmup on resume (already grounded).")
                else:
                    from tgnn_solv.pretrain_pipeline import run_sigma_warmup_pretraining
                    print("\n6a. Running sigma-warmup pretraining (head_sigma grounding)...")
                    warm_meta = run_sigma_warmup_pretraining(
                        model, config, device=device,
                        sigma_train_loader=sigma_train_loader,
                        sigma_val_loader=sigma_val_loader,
                    )
                    if not warm_meta.get("skipped"):
                        print(
                            f"    sigma-warmup: {warm_meta['epochs_run']} epochs, "
                            f"best_val={warm_meta['best_val']:.4f}, "
                            f"area_mae={warm_meta['area_mae']:.2f} Å², "
                            f"gate_passed={warm_meta['area_gate_passed']}"
                        )
                    else:
                        print(f"    sigma-warmup skipped: {warm_meta}")
            trainer.train_full(
                train_loader,
                val_loader,
                idac_train_loader=idac_train_loader,
                crystal_train_loader=crystal_train_loader,
                sigma_train_loader=sigma_train_loader,
                sigma_val_loader=sigma_val_loader,
                resume_state=resume_state,
                on_epoch_end=on_epoch_end,
            )

        # Save checkpoint
        print("\n8. Saving checkpoint...")
        checkpoint = build_checkpoint_payload(
            model=model,
            config=config,
            seed=args.seed,
            experiment_name=logger.experiment_name,
            trainer=trainer,
            resume_state={
                "status": "completed",
                "phase": 4,
                "next_epoch_in_phase": 0,
                "trainer_state_dict": trainer.state_dict(),
            },
            pretrain_info=pretrain_info,
        )
        atomic_torch_save(checkpoint, checkpoint_path)
        print(f"   Model saved to {checkpoint_path}")
        
        # Optional test evaluation
        test_metrics = None
        if args.test_data is not None:
            print("\n9. Evaluating on test set...")
            print("   Test:")
            test_loader = load_data(
                args.test_data,
                config,
                shuffle=False,
                seed=args.seed,
            )
            eval_phase = (
                int(trainer.best_phase)
                if trainer.best_phase is not None
                else (3 if int(config.epochs_phase3) > 0 else 2)
            )
            print(f"   Using validation phase weights from Phase {eval_phase}.")
            test_metrics = trainer.validate(test_loader, phase=eval_phase)
            
            print("\n   Test Metrics:")
            print(json.dumps(test_metrics, indent=2))
            logger.log_artifact("test_metrics", test_metrics)

        if args.run_descriptor_probe:
            print("\n9a. Running descriptor linear probe...")
            maybe_run_descriptor_probe(
                checkpoint_path=checkpoint_path,
                train_data=args.train_data,
                test_data=args.test_data,
                output_dir=args.descriptor_probe_output_dir,
                device=args.descriptor_probe_device,
                logger=logger,
            )

        manifest = build_run_manifest(
            "training_run",
            model_name=checkpoint_path.name,
            model_family="tgnn_solv",
            inputs={
                "config": args.config,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "test_data": args.test_data,
                "idac_train_data": args.idac_train_data,
                "crystal_train_data": args.crystal_train_data,
                "sigma_train_data": args.sigma_train_data,
                "sigma_val_data": args.sigma_val_data,
                "resume_checkpoint": args.resume,
                "pretrain_enabled": bool(args.pretrain),
                "pretrain_checkpoint": args.pretrain_checkpoint,
            },
            outputs={
                "checkpoint": checkpoint_path,
            },
            metadata={
                "experiment_name": logger.experiment_name,
                "log_dir": str(logger.exp_dir),
                "seed": int(args.seed),
                "device": str(device),
                "pretrain_info": pretrain_info,
                # The stream build hash, its row count and the recomputed leak counts,
                # recorded beside the split hashes that `inputs` carries. This is what
                # lets an artifact say which build fed which arm.
                "grounding_streams": stream_provenance,
                "stream_scaffold_overlap_override": bool(args.allow_stream_scaffold_overlap),
            },
        )
        model_card = build_model_card(
            checkpoint_path=checkpoint_path,
            model_family="tgnn_solv",
            config=dataclasses.asdict(config),
            metrics=test_metrics or {},
            metadata={
                "experiment_name": logger.experiment_name,
                "log_dir": str(logger.exp_dir),
                "seed": int(args.seed),
                "pretrain_info": pretrain_info,
            },
        )
        write_json(checkpoint_path.with_suffix(".manifest.json"), manifest)
        write_json(checkpoint_path.with_suffix(".model_card.json"), model_card)
        
        # Finalize logging
        print("\n10. Finalizing experiment...")
        logger.finalize(test_metrics if test_metrics else None)
        
        # Print summary
        print("\n" + "=" * 70)
        print("✓ Training completed successfully!")
        print("=" * 70)
        print(f"Model saved to: {checkpoint_path}")
        print(f"Logs saved to: {logger.exp_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print("Training failed with error:", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
