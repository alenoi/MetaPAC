"""Utility modules for MetaPAC.

This package provides various utilities for:
- Model analysis and metrics (analysis/)
- Hook management for model inspection (hooks/)
- Statistical computations (analysis/)
"""

from .analysis import (
    mae, rmse, spearman,
    regression_metrics, binary_metrics,
    infer_task_type,
    nan_sparsity, quantiles, compute_stats
)
from .hooks import HookManager, HookHFCallback

__all__ = [
    # Hook utilities
    'HookManager', 'HookHFCallback',

    # Metrics and statistics
    'mae', 'rmse', 'spearman',
    'regression_metrics', 'binary_metrics',
    'infer_task_type',
    'nan_sparsity', 'quantiles', 'compute_stats'
]
