"""Utilities for optional Stage 0 pretraining before TGNN task training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import torch

from .config import TGNNSolvConfig
from .model import TGNNSolv
from .pretrain import Pretrainer, download_zinc250k


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Atomically save a checkpoint to avoid partial files on interruption."""
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


def load_pretraining_smiles(
    source: str,
    *,
    max_molecules: int | None = None,
) -> list[str]:
    """Load a SMILES corpus for Stage 0 pretraining."""
    normalized = str(source).strip()
    if not normalized:
        raise ValueError("Pretraining source must be a non-empty string.")

    if normalized.lower() == "zinc250k":
        return download_zinc250k(
            max_molecules=max_molecules if max_molecules is not None else 250000
        )

    path = Path(normalized).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pretraining source does not exist: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
        smiles_col = next(
            (column for column in df.columns if "smi" in column.lower()),
            None,
        )
        if smiles_col is None:
            smiles_col = df.columns[0]
        smiles = df[smiles_col].astype(str).tolist()
    else:
        smiles = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if max_molecules is not None:
        smiles = smiles[:max_molecules]
    return smiles


def derive_pretrain_checkpoint_path(final_checkpoint_path: str | Path) -> Path:
    """Derive a default Stage 0 checkpoint path from the final model checkpoint."""
    checkpoint_path = Path(final_checkpoint_path).expanduser().resolve()
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_pretrained_encoder.pt")


def build_pretrain_checkpoint_payload(
    *,
    model: TGNNSolv,
    config: TGNNSolvConfig,
    pretrain_history: dict[str, list[float]],
    pretrain_source: str,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_lr: float,
    smiles_count: int,
    pairwise_contrastive_csv: str | None = None,
    pairwise_contrastive_weight: float = 0.0,
) -> dict[str, Any]:
    """Serialize the Stage 0 artifacts needed to warm-start a new model."""
    payload: dict[str, Any] = {
        "format": "tgnn_solv_stage0_pretrain",
        "config": asdict(config),
        "gnn_state_dict": model.gnn.state_dict(),
        "readout_state_dict": model.readout.state_dict(),
        "pretrain_history": pretrain_history,
        "pretrain_metadata": {
            "source": pretrain_source,
            "epochs": int(pretrain_epochs),
            "batch_size": int(pretrain_batch_size),
            "lr": float(pretrain_lr),
            "smiles_count": int(smiles_count),
            "pairwise_contrastive_csv": pairwise_contrastive_csv,
            "pairwise_contrastive_weight": float(pairwise_contrastive_weight),
        },
    }
    if getattr(model, "head_sigma", None) is not None:
        payload["sigma_head_state_dict"] = model.head_sigma.state_dict()
    return payload


def save_pretrained_encoder_checkpoint(
    *,
    model: TGNNSolv,
    config: TGNNSolvConfig,
    output_path: str | Path,
    pretrain_history: dict[str, list[float]],
    pretrain_source: str,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_lr: float,
    smiles_count: int,
    pairwise_contrastive_csv: str | None = None,
    pairwise_contrastive_weight: float = 0.0,
) -> Path:
    """Save the Stage 0-pretrained encoder/readout weights."""
    path = Path(output_path).expanduser().resolve()
    payload = build_pretrain_checkpoint_payload(
        model=model,
        config=config,
        pretrain_history=pretrain_history,
        pretrain_source=pretrain_source,
        pretrain_epochs=pretrain_epochs,
        pretrain_batch_size=pretrain_batch_size,
        pretrain_lr=pretrain_lr,
        smiles_count=smiles_count,
        pairwise_contrastive_csv=pairwise_contrastive_csv,
        pairwise_contrastive_weight=pairwise_contrastive_weight,
    )
    atomic_torch_save(payload, path)
    return path


def load_pretrained_encoder_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a saved Stage 0 checkpoint."""
    resolved = Path(path).expanduser().resolve()
    checkpoint = torch.load(resolved, map_location=map_location)
    required = {"gnn_state_dict", "readout_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Pretrain checkpoint {resolved} is missing keys: {sorted(missing)}"
        )
    return checkpoint


def apply_pretrained_encoder_checkpoint(
    model: TGNNSolv,
    checkpoint: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load Stage 0 weights into `model.gnn` and `model.readout`."""
    gnn_load = model.gnn.load_state_dict(
        checkpoint["gnn_state_dict"],
        strict=strict,
    )
    readout_load = model.readout.load_state_dict(
        checkpoint["readout_state_dict"],
        strict=strict,
    )
    if (getattr(model, "head_sigma", None) is not None
            and "sigma_head_state_dict" in checkpoint):
        model.head_sigma.load_state_dict(checkpoint["sigma_head_state_dict"])
    metadata = dict(checkpoint.get("pretrain_metadata") or {})
    metadata["gnn_missing_keys"] = list(gnn_load.missing_keys)
    metadata["gnn_unexpected_keys"] = list(gnn_load.unexpected_keys)
    metadata["readout_missing_keys"] = list(readout_load.missing_keys)
    metadata["readout_unexpected_keys"] = list(readout_load.unexpected_keys)
    metadata["history"] = checkpoint.get("pretrain_history")
    return metadata


def run_sigma_warmup_pretraining(
    model: TGNNSolv,
    config: TGNNSolvConfig,
    *,
    device: torch.device,
    sigma_train_loader,
    sigma_val_loader=None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pretrain head_sigma (+ encoder) on the sigma pool with aux-VAL early-stop
    and an area-anchor gate, BEFORE the SLE curriculum.

    Guards ``head_sigma is None`` (returns ``{"skipped": True}`` for non-COSMO
    models). Trains with its OWN AdamW over head_sigma + gnn + readout params.
    Early-stopping is LOCAL to this routine (independent of the main trainer's
    best_state / patience). Returns metadata dict with keys:
    ``{"history", "best_val", "area_mae", "area_gate_passed", "epochs_run"}``.
    """
    import logging

    from .trainer import TGNNSolvTrainer

    log = logging.getLogger(__name__)

    if getattr(model, "head_sigma", None) is None:
        log.warning("sigma-warmup skipped: model has no head_sigma (non-cosmo).")
        return {"skipped": True}

    if int(config.sigma_warmup_epochs) == 0:
        log.info("sigma-warmup skipped: sigma_warmup_epochs=0.")
        return {"skipped": True}

    # Move model to device first; trainer infers device from model.parameters().
    model.to(device)
    trainer = TGNNSolvTrainer(model, config)

    params = (
        list(model.head_sigma.parameters())
        + list(model.gnn.parameters())
        + list(model.readout.parameters())
    )
    opt = torch.optim.AdamW(
        params,
        lr=float(config.sigma_warmup_lr),
        weight_decay=float(config.weight_decay),
    )

    best_val: float = float("inf")
    best_state: dict | None = None
    patience: int = 0
    history: list[float] = []
    val_loader = sigma_val_loader if sigma_val_loader is not None else sigma_train_loader

    for epoch in range(int(config.sigma_warmup_epochs)):
        model.train()
        for batch in sigma_train_loader:
            opt.zero_grad()
            loss_sol, _ = trainer._sigma_forward_loss(batch, role="solute")
            if getattr(config, "sigma_aux_symmetrize", True):
                loss_slv, _ = trainer._sigma_forward_loss(batch, role="solvent")
                loss = 0.5 * (loss_sol + loss_slv)
            else:
                loss = loss_sol
            # Empty-mask guard: matches the Task 2/4 pattern in _train_sigma_aux_batch
            if not loss.requires_grad or not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip))
            opt.step()

        vmetrics = trainer.validate_sigma(val_loader)
        v: float = vmetrics["sigma_profile"]
        history.append(v)

        if v < best_val:
            best_val = v
            best_state = trainer._clone_model_state()
            patience = 0
        else:
            patience += 1

        if (
            (epoch + 1) >= int(config.sigma_warmup_min_epochs)
            and patience >= int(config.sigma_warmup_patience)
        ):
            log.info(
                "sigma-warmup early-stopped at epoch %d (patience=%d)", epoch + 1, patience
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    area_mae: float = trainer.validate_sigma(val_loader)["sigma_area_mae"]
    passed: bool = area_mae <= float(config.sigma_area_anchor_mae_tol)
    if not passed:
        msg = (
            f"sigma area-anchor gate FAILED: area MAE {area_mae:.1f} Å² > "
            f"tol {config.sigma_area_anchor_mae_tol} Å²"
        )
        if getattr(config, "sigma_area_anchor_strict", False):
            raise RuntimeError(msg)
        log.warning(msg)

    meta: dict[str, Any] = {
        "history": history,
        "best_val": best_val,
        "area_mae": area_mae,
        "area_gate_passed": bool(passed),
        "epochs_run": len(history),
    }

    if save_path is not None:
        payload = build_pretrain_checkpoint_payload(
            model=model,
            config=config,
            pretrain_history={"sigma_warmup": history},
            pretrain_source="sigma_warmup",
            pretrain_epochs=len(history),
            pretrain_batch_size=0,
            pretrain_lr=float(config.sigma_warmup_lr),
            smiles_count=0,
        )
        payload["sigma_warmup_meta"] = meta
        atomic_torch_save(payload, Path(save_path))

    return meta


def run_stage0_pretraining(
    model: TGNNSolv,
    config: TGNNSolvConfig,
    *,
    device: torch.device,
    pretrain_source: str = "zinc250k",
    pretrain_epochs: int = 30,
    pretrain_batch_size: int = 128,
    pretrain_lr: float = 3.0e-4,
    pretrain_max_molecules: int | None = None,
    pairwise_contrastive_csv: str | Path | None = None,
    pairwise_contrastive_weight: float = 0.0,
    pairwise_contrastive_batch_size: int | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run Stage 0 pretraining and optionally save the encoder checkpoint."""
    smiles_list = load_pretraining_smiles(
        pretrain_source,
        max_molecules=pretrain_max_molecules,
    )
    pretrainer = Pretrainer(
        model.gnn,
        model.readout,
        config,
        device=device,
    )
    history = pretrainer.pretrain(
        smiles_list,
        n_epochs=pretrain_epochs,
        batch_size=pretrain_batch_size,
        lr=pretrain_lr,
        pairwise_contrastive_csv=pairwise_contrastive_csv,
        pairwise_contrastive_weight=pairwise_contrastive_weight,
        pairwise_contrastive_batch_size=pairwise_contrastive_batch_size,
    )
    metadata: dict[str, Any] = {
        "source": pretrain_source,
        "epochs": int(pretrain_epochs),
        "batch_size": int(pretrain_batch_size),
        "lr": float(pretrain_lr),
        "smiles_count": int(len(smiles_list)),
        "pairwise_contrastive_csv": (
            str(Path(pairwise_contrastive_csv).expanduser().resolve())
            if pairwise_contrastive_csv is not None
            else None
        ),
        "pairwise_contrastive_weight": float(pairwise_contrastive_weight),
        "history": history,
        "checkpoint_path": None,
    }
    if save_path is not None:
        checkpoint_path = save_pretrained_encoder_checkpoint(
            model=model,
            config=config,
            output_path=save_path,
            pretrain_history=history,
            pretrain_source=pretrain_source,
            pretrain_epochs=pretrain_epochs,
            pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr,
            smiles_count=len(smiles_list),
            pairwise_contrastive_csv=metadata["pairwise_contrastive_csv"],
            pairwise_contrastive_weight=pairwise_contrastive_weight,
        )
        metadata["checkpoint_path"] = str(checkpoint_path)
    return metadata
