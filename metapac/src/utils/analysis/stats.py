"""Unified statistics module for MetaPAC."""
from typing import Dict, Any, Tuple

import numpy as np


def nan_sparsity(x: np.ndarray) -> float:
    """Calculate the sparsity ratio (proportion of NaN or near-zero values)."""
    x = x.ravel()
    if x.size == 0:
        return np.nan
    zeros = np.sum(~np.isfinite(x) | (np.abs(x) <= 1e-12))
    return zeros / x.size


def quantiles(x: np.ndarray) -> Tuple[float, float, float]:
    """Calculate quartiles (25%, 50%, 75%) of the input array."""
    return tuple(np.nanpercentile(x, [25, 50, 75]) if x.size else (np.nan, np.nan, np.nan))


def compute_stats(vec: np.ndarray, prefix: str = "") -> Dict[str, Any]:
    """Compute comprehensive statistics for a numpy array."""
    if vec is None or vec.size == 0:
        return {
            f"{prefix}mean": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}q25": np.nan,
            f"{prefix}q50": np.nan,
            f"{prefix}q75": np.nan,
            f"{prefix}sparsity": np.nan,
            f"{prefix}l1": np.nan,
            f"{prefix}l2": np.nan
        }

    q25, q50, q75 = quantiles(vec)
    return {
        f"{prefix}mean": np.nanmean(vec),
        f"{prefix}std": np.nanstd(vec),
        f"{prefix}min": np.nanmin(vec),
        f"{prefix}max": np.nanmax(vec),
        f"{prefix}q25": q25,
        f"{prefix}q50": q50,
        f"{prefix}q75": q75,
        f"{prefix}sparsity": nan_sparsity(vec),
        f"{prefix}l1": float(np.nansum(np.abs(vec))),
        f"{prefix}l2": float(np.sqrt(np.nansum(np.square(vec)))),
    }
