"""Unified statistics module for MetaPAC."""

from .metrics import (
    mae, rmse, spearman_safe as spearman,
    regression_metrics, binary_metrics,
    infer_task_type
)
from .stats import (
    nan_sparsity,
    quantiles,
    compute_stats
)

__all__ = [
    'mae', 'rmse', 'spearman',
    'regression_metrics', 'binary_metrics',
    'infer_task_type',
    'nan_sparsity', 'quantiles', 'compute_stats'
]
