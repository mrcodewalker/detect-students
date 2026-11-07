"""Reproducibility utilities for deterministic execution across random, numpy, and torch."""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set global random seeds for deterministic execution.

    Args:
        seed: Integer seed value.
        deterministic: If True and PyTorch is available, enforces deterministic algorithms.

    Returns:
        The fixed seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Conditionally seed PyTorch if installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except (AttributeError, RuntimeError):
                # Some operations may not support deterministic algorithms in all PyTorch versions
                pass
    except ImportError:
        # PyTorch not yet installed in this environment
        pass

    return seed

