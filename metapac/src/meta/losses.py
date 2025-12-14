"""Loss functions for meta-learning training.

This module provides custom loss functions used in meta-predictor training,
particularly the Huber loss for robust regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber loss function with configurable delta parameter.
    
    Huber loss is less sensitive to outliers than MSE. It acts quadratic for
    small errors and linear for large errors, with transition at delta.
    
    Formula:
        loss = 0.5 * error^2                    if |error| <= delta
        loss = delta * (|error| - 0.5 * delta)  if |error| > delta
    
    Attributes:
        delta: Threshold where loss transitions from quadratic to linear.
        reduction: How to reduce losses ('mean', 'sum', or 'none').
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        """Initialize Huber loss.
        
        Args:
            delta: Transition point from quadratic to linear loss.
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.
        
        Args:
            input: Predicted values.
            target: Ground truth values.
            
        Returns:
            Computed loss (scalar if reduction != 'none').
        """
        error = input - target
        abs_error = torch.abs(error)

        # Clamp error magnitude at delta for quadratic part
        quadratic_part = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))

        # Piecewise computation: 0.5 * quadratic_part^2 + delta * (|error| - quadratic_part)
        loss = 0.5 * quadratic_part ** 2 + self.delta * (abs_error - quadratic_part)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
