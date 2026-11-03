"""Utilities for post-training analysis of long-running TGNN experiments."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch


_VALUE_RE = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf|NA)"
_PHASE_HINT_RE = re.compile(r"Phase\s+(?P<phase>\d+)\s+(?:epochs|train|val)")
_EPOCH_RE = re.compile(
    rf"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+):\s*"
    rf"train=(?P<train>{_VALUE_RE}),\s*"
    rf"val=(?P<val>{_VALUE_RE})"
    rf"(?:,\s*MAE=(?P<mae>{_VALUE_RE}),\s*R²=(?P<r2>{_VALUE_RE}))?,\s*"
    rf"gate=(?P<gate>{_VALUE_RE})"
)
_COMPONENT_RE = re.compile(
    rf"loss/(?P<name>[A-Za-z0-9_]+)_raw=(?P<raw>{_VALUE_RE})\s+"
    rf"loss/[A-Za-z0-9_]+_weighted=(?P<weighted>{_VALUE_RE})\s+"
    rf"weight=(?P<weight>{_VALUE_RE})"
)
_TOTAL_RE = re.compile(
    rf"loss/total=(?P<total>{_VALUE_RE})\s+"
    rf"loss/sol_fraction=(?P<sol_fraction>{_VALUE_RE})\s+"
    rf"loss/sol_fraction_min=(?P<sol_fraction_min>{_VALUE_RE})\s+"
    rf"loss/max_regularizer_ratio=(?P<max_regularizer_ratio>{_VALUE_RE})\s+"
    r"loss/regularizer_domination_count=(?P<regularizer_domination_count>\d+)"
)


def safe_float(value: object) -> float | None:
    """Convert a scalar-ish value to a finite float when possible."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() == "NA":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def checkpoint_status(checkpoint_path: Path) -> dict[str, Any]:
    """Read high-level training status from a TGNN checkpoint."""
    if not checkpoint_path.exists():
        return {
            "exists": False,
            "resume_status": None,
            "resume_phase": None,
            "resume_next_epoch_in_phase": None,
            "trainer_best_val_loss": None,
            "trainer_best_epoch": None,
            "trainer_best_phase": None,
            "trainer_has_best_state": False,
        }

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resume_state = payload.get("resume_state") or {}
    trainer_state = payload.get("trainer_state_dict") or {}
    return {
        "exists": True,
        "resume_status": resume_state.get("status"),
        "resume_phase": resume_state.get("phase"),
        "resume_next_epoch_in_phase": resume_state.get("next_epoch_in_phase"),
        "trainer_best_val_loss": safe_float(trainer_state.get("best_val_loss")),
        "trainer_best_epoch": trainer_state.get("best_epoch"),
        "trainer_best_phase": trainer_state.get("best_phase"),
        "trainer_has_best_state": isinstance(trainer_state.get("best_state"), dict),
    }


def wait_for_checkpoint_completion(
    checkpoint_path: Path,
    *,
    poll_interval_s: float = 600.0,
    timeout_s: float | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Block until the checkpoint marks training as completed."""
    started = time.time()
    last_signature: tuple[Any, ...] | None = None

    while True:
        status = checkpoint_status(checkpoint_path)
        signature = (
            status.get("exists"),
            status.get("resume_status"),
            status.get("resume_phase"),
            status.get("resume_next_epoch_in_phase"),
            status.get("trainer_best_val_loss"),
        )
        if signature != last_signature and log is not None:
            log(
                "Checkpoint status: "
                f"exists={status['exists']} "
                f"resume_status={status['resume_status']} "
                f"phase={status['resume_phase']} "
                f"next_epoch={status['resume_next_epoch_in_phase']} "
                f"best_val={status['trainer_best_val_loss']}"
            )
            last_signature = signature

        if status.get("resume_status") == "completed":
            return status

        if timeout_s is not None and (time.time() - started) > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for checkpoint completion: {checkpoint_path}"
            )

        time.sleep(max(float(poll_interval_s), 1.0))


def state_dicts_equal(
    left: dict[str, torch.Tensor] | None,
    right: dict[str, torch.Tensor] | None,
) -> bool | None:
    """Return whether two model state dicts are exactly equal."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def build_selected_checkpoint_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a checkpoint payload whose model weights are the tracked best state."""
    current_state = None
    if isinstance(payload.get("model_state"), dict):
        current_state = payload["model_state"]
    elif isinstance(payload.get("model_state_dict"), dict):
        current_state = payload["model_state_dict"]

    trainer_state = payload.get("trainer_state_dict") or {}
    best_state = trainer_state.get("best_state")

    if isinstance(best_state, dict) and best_state:
        selected_state = best_state
        selected_source = "trainer_state_dict.best_state"
    elif isinstance(current_state, dict) and current_state:
        selected_state = current_state
        selected_source = "model_state"
    else:
        raise ValueError("Checkpoint does not contain a usable model state dict.")

    selected_payload = dict(payload)
    selected_payload["model_state"] = selected_state
    selected_payload["model_state_dict"] = selected_state

    metadata = {
        "selected_state_source": selected_source,
        "model_state_matches_best_state": state_dicts_equal(current_state, best_state),
        "trainer_best_val_loss": safe_float(trainer_state.get("best_val_loss")),
        "trainer_best_epoch": trainer_state.get("best_epoch"),
        "trainer_best_phase": trainer_state.get("best_phase"),
        "resume_status": (payload.get("resume_state") or {}).get("status"),
    }
    return selected_payload, metadata


def parse_training_log_text(text: str) -> dict[str, Any]:
    """Parse per-epoch TGNN training curves from a raw train.log string."""
    lines = text.replace("\r", "\n").splitlines()
    entries_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    current_key: tuple[int, int] | None = None
    last_phase_hint: int | None = None

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        phase_match = _PHASE_HINT_RE.search(line)
        if phase_match is not None:
            last_phase_hint = int(phase_match.group("phase"))

        epoch_match = _EPOCH_RE.search(line)
        if epoch_match is not None:
            phase = last_phase_hint
            if phase is None:
                continue
            epoch = int(epoch_match.group("epoch"))
            total = int(epoch_match.group("total"))
            entry = {
                "phase": phase,
                "epoch": epoch,
                "epoch_total": total,
                "train_loss": safe_float(epoch_match.group("train")),
                "val_loss": safe_float(epoch_match.group("val")),
                "val_mae": safe_float(epoch_match.group("mae")),
                "r2": safe_float(epoch_match.group("r2")),
                "gate": safe_float(epoch_match.group("gate")),
                "loss_components": {},
                "line_no": line_no,
            }
            entries_by_key[(phase, epoch)] = entry
            current_key = (phase, epoch)
            continue

        if current_key is None:
            continue

        component_match = _COMPONENT_RE.search(line)
        if component_match is not None:
            entry = entries_by_key[current_key]
            entry["loss_components"][component_match.group("name")] = {
                "raw": safe_float(component_match.group("raw")),
                "weighted": safe_float(component_match.group("weighted")),
                "weight": safe_float(component_match.group("weight")),
            }
            continue

        total_match = _TOTAL_RE.search(line)
        if total_match is not None:
            entry = entries_by_key[current_key]
            entry["loss_total"] = safe_float(total_match.group("total"))
            entry["sol_fraction"] = safe_float(total_match.group("sol_fraction"))
            entry["sol_fraction_min"] = safe_float(total_match.group("sol_fraction_min"))
            entry["max_regularizer_ratio"] = safe_float(
                total_match.group("max_regularizer_ratio")
            )
            entry["regularizer_domination_count"] = int(
                total_match.group("regularizer_domination_count")
            )

    entries = sorted(entries_by_key.values(), key=lambda item: (item["phase"], item["epoch"]))
    for entry in entries:
        sol_component = entry.get("loss_components", {}).get("sol", {})
        sol_raw = safe_float(sol_component.get("raw"))
        val_mae = safe_float(entry.get("val_mae"))
        entry["train_sol_raw"] = sol_raw
        entry["train_sol_raw_over_val_mae"] = (
            (sol_raw / val_mae) if sol_raw is not None and val_mae not in (None, 0.0) else None
        )

    phase2_entries = [entry for entry in entries if entry["phase"] == 2 and entry.get("val_mae") is not None]
    best_phase2 = min(phase2_entries, key=lambda item: item["val_mae"]) if phase2_entries else None
    final_entry = entries[-1] if entries else None

    return {
        "entries": entries,
        "n_entries": len(entries),
        "best_phase2_by_val_mae": best_phase2,
        "final_entry": final_entry,
    }


def parse_training_log(log_path: Path) -> dict[str, Any]:
    """Parse a train.log file into machine-readable learning curves."""
    return parse_training_log_text(log_path.read_text(encoding="utf-8", errors="replace"))
