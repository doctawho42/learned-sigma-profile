"""Helpers for machine-readable run manifests, model cards, and benchmark cards."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reporting import normalize_report_payload


ARTIFACT_SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for artifact metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def git_commit() -> str | None:
    """Return the current git commit, if available."""
    return _git_output("rev-parse", "HEAD")


def git_branch() -> str | None:
    """Return the current git branch, if available."""
    return _git_output("rev-parse", "--abbrev-ref", "HEAD")


def git_is_dirty() -> bool | None:
    """Return whether the working tree is dirty."""
    status = _git_output("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute a SHA256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: str | Path, *, role: str | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Build a machine-readable description for one file."""
    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    root_path = Path(root).expanduser().resolve() if root else None
    try:
        relative = str(file_path.relative_to(root_path)) if root_path is not None else str(file_path.relative_to(_repo_root()))
    except Exception:
        relative = str(file_path)
    return {
        "role": role,
        "path": str(file_path),
        "relative_path": relative,
        "sha256": sha256_file(file_path),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    }


def _records_from_mapping(paths: Mapping[str, str | Path | None] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, value in (paths or {}).items():
        if value is None:
            continue
        try:
            path = Path(value).expanduser().resolve()
        except Exception:
            continue
        if path.exists() and path.is_file():
            records.append(describe_file(path, role=str(role)))
    return records


#: Момент импорта модуля. `train.py` импортирует его на старте, поэтому разница до записи
#: манифеста близка к настенному времени прогона -- но это НЕ оно, и ключ назван буквально.
_IMPORTED_AT = time.monotonic()


def compute_snapshot() -> dict[str, Any]:
    """Чем именно считали: версия torch, карта, драйвер, ПИК VRAM.

    ЗАЧЕМ. В манифестах депонированных армов (`checkpoints/e5_leakfree/*.manifest.json`) записаны
    `python_version` и `platform`, а версии torch, имени карты, драйвера и пика памяти НЕТ. Поэтому
    про те чекпойнты нельзя сказать, на каком torch они обучены, -- а при переносе на другую машину
    это первое, что спрашивают. Снимок добавлен 2026-10-22 при переезде на кластер.

    Функция обязана быть безопасной там, где torch или `nvidia-smi` отсутствуют: `runtime_snapshot`
    вызывают и чисто процессорные скрипты.
    """
    out: dict[str, Any] = {
        "elapsed_since_module_import_s": round(time.monotonic() - _IMPORTED_AT, 1),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        out["torch"] = None
        out["note"] = "torch не импортируется -- снимок вычислителя пуст"
        return out
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:                                    # драйвер может отвалиться на ходу
        out["cuda_available"] = False
    if out["cuda_available"]:
        try:
            out["device_count"] = int(torch.cuda.device_count())
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            out["gpu_mem_total_gib"] = round(torch.cuda.mem_get_info(0)[1] / 2**30, 2)
            # ПИК -- главное из того, чего не было: он отвечает на «влезет ли на другую карту».
            out["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            out["peak_vram_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        except Exception as exc:                         # не ронять манифест из-за провенанса
            out["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out["driver"] = subprocess.run(
                [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=True
            ).stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, OSError, IndexError):
            out["driver"] = None
    return out


def runtime_snapshot() -> dict[str, Any]:
    """Capture the current runtime and repo state."""
    return {
        "created_at": utc_now_iso(),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_is_dirty(),
        "repo_root": str(_repo_root()),
        "compute": compute_snapshot(),
    }


def build_run_manifest(
    run_type: str,
    *,
    model_name: str | None = None,
    model_family: str | None = None,
    command: Sequence[str] | None = None,
    inputs: Mapping[str, str | Path | None] | None = None,
    outputs: Mapping[str, str | Path | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic run manifest that can be attached to any artifact bundle."""
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_type": str(run_type),
        "model_name": model_name,
        "model_family": model_family,
        "command": list(command) if command is not None else None,
        "runtime": runtime_snapshot(),
        "inputs": _records_from_mapping(inputs),
        "outputs": _records_from_mapping(outputs),
        "metadata": dict(metadata or {}),
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON to disk with stable formatting."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return out


def checkpoint_capabilities(model_family: str | None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Infer checkpoint capabilities from family and config."""
    cfg = dict(config or {})
    family = str(model_family or "unknown")
    return {
        "inference": family in {"tgnn_solv", "direct_gnn"},
        "temperature_scan": family in {"tgnn_solv", "direct_gnn"},
        "uncertainty": family in {"tgnn_solv", "direct_gnn"},
        "calibration": family in {"tgnn_solv", "direct_gnn"},
        "ood_screening": family == "tgnn_solv",
        "physics_decomposition": family == "tgnn_solv",
        "descriptor_augmentation": bool(cfg.get("use_descriptor_augmentation")),
        "morgan_features": bool(cfg.get("use_morgan_features")),
        "descriptor_priors": bool(cfg.get("use_descriptor_priors")),
        "group_priors": bool(cfg.get("use_group_priors")),
        "gc_priors_crystal": bool(cfg.get("use_gc_priors_crystal")),
    }


def build_model_card(
    *,
    checkpoint_path: str | Path,
    model_family: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable model card for a checkpoint."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    card = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "card_type": "model_card",
        "checkpoint": describe_file(checkpoint, role="checkpoint"),
        "model_family": str(model_family),
        "config": dict(config),
        "capabilities": checkpoint_capabilities(model_family, config),
        "metrics": dict(metrics or {}),
        "metadata": dict(metadata or {}),
        "runtime": runtime_snapshot(),
    }
    return card


def _benchmark_capabilities(
    *,
    model_family: str | None,
    predictions_columns: Sequence[str] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    columns = {str(column) for column in (predictions_columns or [])}
    meta = dict(metadata or {})
    family = str(model_family or meta.get("model_family") or "unknown")
    return {
        "uncertainty": "prediction_std" in columns,
        "physics_decomposition": family == "tgnn_solv",
        "ood_screening": family == "tgnn_solv",
        "temperature_dependent": bool(meta.get("temperature_dependent") or meta.get("include_temperature")),
        "calibrated": bool(meta.get("calibrated")),
        "native_training": bool(meta.get("native_training")),
        "custom_adapter": bool(meta.get("adapter_ref")),
    }


def build_benchmark_card(
    report_payload: Mapping[str, Any],
    *,
    summary_row: Mapping[str, Any] | None = None,
    predictions_columns: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact benchmark card from a canonical report payload."""
    report = normalize_report_payload(report_payload)
    report_meta = report.get("metadata", {}) if isinstance(report, dict) else {}
    merged_meta = dict(report_meta if isinstance(report_meta, Mapping) else {})
    merged_meta.update(dict(metadata or {}))
    overall = report.get("overall", {}) if isinstance(report, dict) else {}
    split = merged_meta.get("split")
    family = merged_meta.get("model_family")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "card_type": "benchmark_card",
        "model": {
            "name": merged_meta.get("model") or merged_meta.get("model_name"),
            "family": family,
        },
        "split": split,
        "headline_metrics": {
            "mae": overall.get("mae"),
            "rmse": overall.get("rmse"),
            "r2": overall.get("r2"),
            "bias": overall.get("bias"),
            "n_samples": overall.get("n_samples"),
            "n_predictions": overall.get("n_predictions"),
        },
        "summary_row": dict(summary_row or {}),
        "capabilities": _benchmark_capabilities(
            model_family=str(family) if family is not None else None,
            predictions_columns=predictions_columns,
            metadata=merged_meta,
        ),
        "metadata": merged_meta,
        "runtime": runtime_snapshot(),
    }

