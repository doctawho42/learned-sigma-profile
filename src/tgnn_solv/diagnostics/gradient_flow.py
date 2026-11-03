"""Gradient-flow monitoring utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class GradientRecord:
    """One gradient summary for one parameter after one backward pass."""

    norm: float
    max: float
    mean: float


class GradientFlowMonitor:
    """Attach hooks to trainable parameters and record gradient statistics."""

    def __init__(self, model: nn.Module) -> None:
        self.stats: dict[str, list[GradientRecord]] = defaultdict(list)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.stats.setdefault(name, [])
            self._hooks.append(param.register_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(grad: Tensor | None) -> None:
            self._record(name, grad)

        return hook

    def _record(self, name: str, grad: Tensor | None) -> None:
        if grad is None:
            return
        grad_detached = grad.detach()
        self.stats[name].append(
            GradientRecord(
                norm=float(grad_detached.norm().item()),
                max=float(grad_detached.abs().max().item()),
                mean=float(grad_detached.abs().mean().item()),
            )
        )

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return per-parameter statistics aggregated over all recorded steps."""
        result: dict[str, dict[str, float | int]] = {}
        for name, records in self.stats.items():
            if not records:
                continue
            norms = [record.norm for record in records]
            means = [record.mean for record in records]
            maxes = [record.max for record in records]
            result[name] = {
                "mean_norm": float(sum(norms) / len(norms)),
                "max_norm": float(max(norms)),
                "min_norm": float(min(norms)),
                "mean_abs_grad": float(sum(means) / len(means)),
                "max_abs_grad": float(max(maxes)),
                "n_steps": len(records),
                "n_zero": int(sum(1 for value in norms if value < 1e-10)),
            }
        return result

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
