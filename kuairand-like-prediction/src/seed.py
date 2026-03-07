"""Centralized seeding utilities for reproducibility."""
import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set seeds for Python, numpy and optionally torch (if installed).

    Also sets PYTHONHASHSEED for deterministic hashing.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # torch might not be installed in some environments; that's fine
        pass
