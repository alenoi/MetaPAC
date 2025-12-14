"""Dimensionality reduction functions for tensor data.

This module provides various pooling strategies to reduce high-dimensional
activation and gradient tensors to fixed-size feature vectors for meta-learning.
"""

from typing import Optional

import numpy as np


def _to_2d(arr) -> Optional[np.ndarray]:
    """Normalize input into 2D numpy array with shape [sequence, dimension].
    
    Args:
        arr: Input array (can be string representation, list, or numpy array).
        
    Returns:
        2D numpy array or None if input is invalid.
    """
    if arr is None:
        return None
    if isinstance(arr, str):
        try:
            arr = np.array(eval(arr), dtype=float)
        except Exception:
            return None
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def cls_pool(arr2d) -> Optional[np.ndarray]:
    """Extract CLS token (first position) from sequence.
    
    Commonly used for transformer models where the first token contains
    aggregate sequence information.
    
    Args:
        arr2d: Input array to pool.
        
    Returns:
        First row of the 2D array, or None if input is invalid.
    """
    if arr2d is None:
        return None
    arr2d = _to_2d(arr2d)
    if arr2d is None:
        return None
    return arr2d[0]


def safe_mean_pool(arr2d, min_valid_ratio: float = 0.5, fill_value: float = 0.0) -> Optional[np.ndarray]:
    """Mean pooling with robust NaN handling and validity checks.
    
    Computes mean across sequence dimension, but only if sufficient
    valid (non-NaN) values are present. Otherwise returns fill values.
    
    Args:
        arr2d: Input 2D array with shape [sequence, features].
        min_valid_ratio: Minimum ratio of valid (non-NaN) values required.
        fill_value: Value to use when data is insufficient.
        
    Returns:
        Mean-pooled feature vector, or array of fill_values if invalid.
    """
    if arr2d is None:
        return None
    arr2d = _to_2d(arr2d)
    if arr2d is None:
        return None

    valid_mask = ~np.isnan(arr2d)
    valid_ratio = np.mean(valid_mask)

    if valid_ratio >= min_valid_ratio:
        return np.nanmean(arr2d, axis=0)
    return np.full(arr2d.shape[1], fill_value)


def safe_max_pool(arr2d, min_valid_ratio: float = 0.5, fill_value: float = 0.0) -> Optional[np.ndarray]:
    """Max pooling with robust NaN handling and validity checks.
    
    Computes maximum across sequence dimension, but only if sufficient
    valid (non-NaN) values are present. Otherwise returns fill values.
    
    Args:
        arr2d: Input 2D array with shape [sequence, features].
        min_valid_ratio: Minimum ratio of valid (non-NaN) values required.
        fill_value: Value to use when data is insufficient.
        
    Returns:
        Max-pooled feature vector, or array of fill_values if invalid.
    """
    if arr2d is None:
        return None
    arr2d = _to_2d(arr2d)
    if arr2d is None:
        return None

    valid_mask = ~np.isnan(arr2d)
    valid_ratio = np.mean(valid_mask)

    if valid_ratio >= min_valid_ratio:
        return np.nanmax(arr2d, axis=0)
    return np.full(arr2d.shape[1], fill_value)


def mean_pool(arr2d) -> Optional[np.ndarray]:
    """Mean pooling with default parameters (backward compatibility).
    
    Args:
        arr2d: Input array to pool.
        
    Returns:
        Mean-pooled feature vector.
    """
    return safe_mean_pool(arr2d)


def max_pool(arr2d) -> Optional[np.ndarray]:
    """Max pooling with default parameters (backward compatibility).
    
    Args:
        arr2d: Input array to pool.
        
    Returns:
        Max-pooled feature vector.
    """
    return safe_max_pool(arr2d)


# Registry of available reduction functions
REDUCERS = {
    "CLS": cls_pool,
    "mean_pool": mean_pool,
    "max_pool": max_pool,
    "safe_mean_pool": safe_mean_pool,
    "safe_max_pool": safe_max_pool,
}


def apply_reducer(arr, reducer_name: str) -> Optional[np.ndarray]:
    """Apply named reduction function to input array.
    
    Args:
        arr: Input array to reduce.
        reducer_name: Name of reduction function (must be in REDUCERS registry).
        
    Returns:
        Reduced feature vector.
        
    Raises:
        ValueError: If reducer_name is not recognized.
    """
    reducer_fn = REDUCERS.get(reducer_name)
    if reducer_fn is None:
        available = ', '.join(REDUCERS.keys())
        raise ValueError(
            f"Unknown reducer: '{reducer_name}'. "
            f"Available reducers: {available}"
        )
    return reducer_fn(arr)
