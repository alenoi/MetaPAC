# src/meta/calibration.py
# Per-group affine calibration: ŷ' = a_g * ŷ + b_g
# Forcing a to remain positive preserves Spearman ordering; if a<0 on a small sample,
# clamp it to a small positive value.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class AffineCalib:
    a: float
    b: float


def _fit_affine(y_true: np.ndarray, y_pred: np.ndarray, ridge: float = 1e-6) -> Tuple[float, float]:
    # Solve min ||y - [y_pred, 1] @ [a, b]||^2 + ridge*||[a,b]||^2.
    X = np.stack([y_pred, np.ones_like(y_pred)], axis=1)
    XtX = X.T @ X + ridge * np.eye(2)
    Xty = X.T @ y_true
    a, b = np.linalg.solve(XtX, Xty)
    if a <= 0.0:
        a = max(1e-6, a)  # Keep it positive to preserve ranking.
    return float(a), float(b)


def fit_groupwise_affine(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        min_group_size: int = 100,
        ridge: float = 1e-6,
) -> Dict[str, AffineCalib]:
    out: Dict[str, AffineCalib] = {}
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    groups = np.asarray(groups).astype(str)

    # Global fallback.
    a_glob, b_glob = _fit_affine(y_true, y_pred, ridge=ridge)

    unique = np.unique(groups)
    for g in unique:
        mask = groups == g
        if mask.sum() < min_group_size:
            out[g] = AffineCalib(a=a_glob, b=b_glob)
            continue
        a, b = _fit_affine(y_true[mask], y_pred[mask], ridge=ridge)
        out[g] = AffineCalib(a=a, b=b)
    # Store the "__GLOBAL__" key as well.
    out["__GLOBAL__"] = AffineCalib(a=a_glob, b=b_glob)
    return out


def apply_groupwise_affine(
        y_pred: np.ndarray,
        groups: np.ndarray,
        params: Dict[str, AffineCalib],
) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    y_pred = np.asarray(y_pred).astype(float)
    y_out = np.empty_like(y_pred)
    glob = params.get("__GLOBAL__")
    for i, g in enumerate(groups):
        p = params.get(g, glob)
        y_out[i] = p.a * y_pred[i] + p.b
    return y_out
