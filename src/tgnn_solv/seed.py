"""Unified entry point for setting random seed across the entire project."""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seed for complete reproducibility.
    
    Sets seed for:
    - random (Python stdlib)
    - numpy
    - torch (CPU + all GPUs)
    - PYTHONHASHSEED
    - cuDNN deterministic mode
    
    Args:
        seed: Random seed value.
        deterministic: If True, enables torch.backends.cudnn.deterministic
            and disables benchmark. Slows down training by ~10%, but guarantees
            reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

