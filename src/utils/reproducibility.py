"""Reproducibility utilities for deterministic execution."""

import os
import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from typing import Optional


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # For libraries that use their own RNG
    try:
        import lightgbm as lgb
        lgb_params = {"seed": seed}
    except ImportError:
        pass
    
    try:
        import xgboost as xgb
    except ImportError:
        pass
    
    # PyTorch if available
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, 'cuda'):
                torch.cuda.manual_seed_all(seed)
            if deterministic and hasattr(torch, 'backends'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random generator with optional seed."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little')
    return np.random.default_rng(seed)


class ReproducibleContext:
    """Context manager for reproducible code blocks."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.prev_state = None
    
    def __enter__(self):
        self.prev_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        set_global_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_state:
            random.setstate(self.prev_state['python'])
            np.random.set_state(self.prev_state['numpy'])


def verify_deterministic(func, *args, n_runs: int = 3, **kwargs) -> bool:
    """Verify a function produces deterministic output."""
    results = []
    for _ in range(n_runs):
        with ReproducibleContext(42):
            result = func(*args, **kwargs)
        results.append(result)
    
    # Compare all results
    first = results[0]
    for r in results[1:]:
        if isinstance(first, (np.ndarray, list)):
            if not np.array_equal(np.array(first), np.array(r)):
                return False
        elif first != r:
            return False
    return True