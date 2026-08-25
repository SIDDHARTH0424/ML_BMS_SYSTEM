"""
Global random seed management for reproducibility.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and (if importable) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator, optionally seeded, for isolated stochastic sampling."""
    return np.random.default_rng(seed)
