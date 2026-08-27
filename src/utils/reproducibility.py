"""Single reproducibility entry point for data preparation and training."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Seed Python, NumPy and PyTorch (when available) deterministically."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
