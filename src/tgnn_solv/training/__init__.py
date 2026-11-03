"""Grouped training namespace with lazy exports.

The heavy trainer/pretraining wrappers import PyG and scipy transitively.
Keeping exports lazy lets lightweight utilities such as Hansen contrastive
losses be imported without constructing the full training stack.
"""

from __future__ import annotations

__all__ = [
    "TGNNSolvTrainer",
    "TGNNSolvLoss",
    "Pretrainer",
    "OptunaTuner",
    "HansenContrastiveLoss",
    "ChannelHansenContrastiveLoss",
    "PairHansenContrastiveLoss",
]


def __getattr__(name: str) -> object:
    if name == "TGNNSolvTrainer":
        from .trainer import TGNNSolvTrainer

        return TGNNSolvTrainer
    if name == "TGNNSolvLoss":
        from .losses import TGNNSolvLoss

        return TGNNSolvLoss
    if name == "Pretrainer":
        from .pretrain import Pretrainer

        return Pretrainer
    if name == "OptunaTuner":
        from .tuner import OptunaTuner

        return OptunaTuner
    if name in {
        "HansenContrastiveLoss",
        "ChannelHansenContrastiveLoss",
        "PairHansenContrastiveLoss",
    }:
        from .hansen_contrastive import (
            ChannelHansenContrastiveLoss,
            HansenContrastiveLoss,
            PairHansenContrastiveLoss,
        )

        exports = {
            "HansenContrastiveLoss": HansenContrastiveLoss,
            "ChannelHansenContrastiveLoss": ChannelHansenContrastiveLoss,
            "PairHansenContrastiveLoss": PairHansenContrastiveLoss,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
