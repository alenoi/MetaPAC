# metapac/src/utils/metrics.py
from __future__ import annotations

"""
Unified metrics module for MetaPAC.

Contents:
- Generic regression errors: mae, rmse
- Rank-based correlation: spearman_safe, grouped_spearman
- Meta-predictor training metrics (optional scikit-learn dependency):
    - infer_task_type
    - regression_metrics (MAE, RMSE, R2)
    - binary_metrics (accuracy, F1, ROC-AUC)
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.stats import spearmanr

# Optional sklearn dependency for model-eval metrics
try:
    from sklearn import metrics as sk_metrics  # type: ignore

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    sk_metrics = None  # type: ignore
    _HAS_SKLEARN = False


# ------------------------------
# Basic regression error metrics
# ------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------
# Rank correlation (Spearman) – safe forms
# ---------------------------------------

def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman's rho with guards:
    - return 0.0 for <3 samples
    - return 0.0 if either array is constant
    - NaNs omitted
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size < 3 or y_pred.size < 3:
        return 0.0
    if np.allclose(y_true, y_true[:1]) or np.allclose(y_pred, y_pred[:1]):
        return 0.0
    res = spearmanr(y_true, y_pred, nan_policy="omit")
    coef = res.correlation if hasattr(res, "correlation") else res[0]
    return float(coef) if (coef is not None and np.isfinite(coef)) else 0.0


def grouped_spearman(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        weighted: bool = True
) -> Tuple[float, List[Tuple[Any, int, float]], int]:
    """
    Compute per-group Spearman, then aggregate by weighted or unweighted mean.

    Returns:
        agg_coef: float                Aggregated coefficient
        per_group: list                [(group_id, n, coef), ...]
        skipped_groups: int            Groups with <3 samples
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    groups = np.asarray(groups).ravel()

    uniq = np.unique(groups)
    vals: List[float] = []
    weights: List[int] = []
    per_group: List[Tuple[Any, int, float]] = []
    skipped = 0

    for g in uniq:
        idx = (groups == g)
        n = int(idx.sum())
        if n < 3:
            skipped += 1
            continue
        coef = spearman_safe(y_true[idx], y_pred[idx])
        vals.append(coef)
        weights.append(n)
        per_group.append((g, n, coef))

    if not vals:
        return 0.0, per_group, skipped

    vals_arr = np.asarray(vals, dtype=float)
    if weighted:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        agg = float((vals_arr * w).sum())
    else:
        agg = float(vals_arr.mean())

    return agg, per_group, skipped


# ---------------------------------------------------------
# Meta-predictor training/evaluation metrics (with sklearn)
# ---------------------------------------------------------

def _require_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn not available. Install with `pip install scikit-learn` "
            "to use regression_metrics / binary_metrics / infer_task_type."
        )


def infer_task_type(y: np.ndarray) -> str:
    """
    Infer task type from target array.
    Returns: 'binary' or 'regression'
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return "regression"
    if np.issubdtype(y.dtype, np.bool_):
        return "binary"
    # Treat small integer sets (<=2 unique) as binary
    uniq = np.unique(y[~np.isnan(y)]) if np.issubdtype(y.dtype, np.floating) else np.unique(y)
    if uniq.size <= 2:
        return "binary"
    return "regression"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R2 for regression tasks.
    """
    _require_sklearn()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }
    return out


def binary_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute accuracy, F1, ROC-AUC for binary tasks given probabilities for class 1.
    """
    _require_sklearn()
    y_true = np.asarray(y_true).ravel().astype(int)
    y_proba = np.asarray(y_proba).ravel()
    y_hat = (y_proba >= 0.5).astype(int)

    acc = float(sk_metrics.accuracy_score(y_true, y_hat))
    f1 = float(sk_metrics.f1_score(y_true, y_hat))
    try:
        auc = float(sk_metrics.roc_auc_score(y_true, y_proba))
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}
