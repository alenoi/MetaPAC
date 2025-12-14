"""Multi-layer perceptron (MLP) model for regression tasks.

This module provides a simple feedforward neural network with configurable
architecture for regression problems.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    """Create activation function module from name.
    
    Args:
        name: Activation function name (case-insensitive).
              Supported: 'relu', 'gelu'.
    
    Returns:
        PyTorch activation module.
        
    Raises:
        ValueError: If activation name is not supported.
    """
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: '{name}'. Supported: 'relu', 'gelu'.")


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression tasks.
    
    A simple feedforward neural network with configurable hidden layers,
    activation functions, and dropout for regularization.
    
    Attributes:
        net: Sequential network of linear layers with activation and dropout.
    """

    def __init__(self, in_dim: int, hidden_sizes: List[int], dropout: float, activation: str) -> None:
        """Initialize MLP regressor.
        
        Args:
            in_dim: Input feature dimension.
            hidden_sizes: List of hidden layer sizes (e.g., [128, 64, 32]).
            dropout: Dropout probability for regularization.
            activation: Activation function name ('relu' or 'gelu').
        """
        super().__init__()
        layers = []
        current_dim = in_dim
        activation_fn = make_activation(activation)

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(current_dim, hidden_size),
                activation_fn,
                nn.Dropout(dropout)
            ]
            current_dim = hidden_size

        # Output layer (single scalar prediction)
        layers += [nn.Linear(current_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, in_dim].
            
        Returns:
            Predicted values of shape [batch_size].
        """
        # Output shape: (batch_size, 1) -> squeeze to (batch_size,)
        return self.net(x).squeeze(-1)
