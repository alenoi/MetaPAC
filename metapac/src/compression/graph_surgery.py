"""
Graph surgery utilities for physical pruning.

This module provides low-level utilities for reshaping and rebinding
neural network modules during structured pruning.
"""
from __future__ import annotations

import logging
from typing import Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GraphSurgery:
    """Utilities for physical pruning - actual tensor reshaping and module replacement."""

    @staticmethod
    def replace_linear(
            old_linear: nn.Linear,
            new_weight: torch.Tensor,
            new_bias: Optional[torch.Tensor] = None
    ) -> nn.Linear:
        """Create a new Linear layer with given weight and bias.
        
        Args:
            old_linear: Original linear layer (for device/dtype reference)
            new_weight: New weight tensor [out_features, in_features]
            new_bias: New bias tensor [out_features] or None
            
        Returns:
            New nn.Linear with proper shapes and copied tensors
        """
        out_features, in_features = new_weight.shape

        # Create new linear layer
        new_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=(new_bias is not None),
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype
        )

        # Copy weights (no grad during surgery)
        with torch.no_grad():
            new_linear.weight.copy_(new_weight)
            if new_bias is not None and new_linear.bias is not None:
                new_linear.bias.copy_(new_bias)

        # Preserve requires_grad
        new_linear.weight.requires_grad = old_linear.weight.requires_grad
        if new_linear.bias is not None:
            new_linear.bias.requires_grad = old_linear.bias.requires_grad if old_linear.bias is not None else True

        return new_linear

    @staticmethod
    def slice_linear_in_features(
            old_linear: nn.Linear,
            in_indices: List[int]
    ) -> nn.Linear:
        """Slice a linear layer's input features (columns of weight matrix).
        
        Args:
            old_linear: Original linear layer
            in_indices: Indices of input features to keep
            
        Returns:
            New linear layer with sliced input dimension
        """
        in_indices = sorted(in_indices)

        # Slice weight matrix columns (input features)
        with torch.no_grad():
            new_weight = old_linear.weight[:, in_indices].clone()
            new_bias = old_linear.bias.clone() if old_linear.bias is not None else None

        return GraphSurgery.replace_linear(old_linear, new_weight, new_bias)

    @staticmethod
    def slice_linear_out_features(
            old_linear: nn.Linear,
            out_indices: List[int]
    ) -> nn.Linear:
        """Slice a linear layer's output features (rows of weight matrix).
        
        Args:
            old_linear: Original linear layer
            out_indices: Indices of output features to keep
            
        Returns:
            New linear layer with sliced output dimension
        """
        out_indices = sorted(out_indices)

        # Slice weight matrix rows (output features) and bias
        with torch.no_grad():
            new_weight = old_linear.weight[out_indices, :].clone()
            new_bias = old_linear.bias[out_indices].clone() if old_linear.bias is not None else None

        return GraphSurgery.replace_linear(old_linear, new_weight, new_bias)

    @staticmethod
    def slice_attention_head_projections(
            q_lin: nn.Linear,
            k_lin: nn.Linear,
            v_lin: nn.Linear,
            out_lin: nn.Linear,
            kept_head_indices: List[int],
            num_heads: int,
            hidden_size: int
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
        """Slice attention projection matrices to keep only specified heads.
        
        For multi-head attention:
        - Q, K, V projections: [hidden_size, hidden_size]
        - Out projection: [hidden_size, hidden_size]
        - Each head has size: head_dim = hidden_size // num_heads
        
        Args:
            q_lin: Query projection
            k_lin: Key projection
            v_lin: Value projection
            out_lin: Output projection
            kept_head_indices: Indices of heads to keep (0-indexed)
            num_heads: Original number of heads
            hidden_size: Hidden dimension size
            
        Returns:
            Tuple of (new_q_lin, new_k_lin, new_v_lin, new_out_lin)
        """
        head_dim = hidden_size // num_heads
        kept_head_indices = sorted(kept_head_indices)

        logger.info(f"Slicing attention heads: keeping {len(kept_head_indices)}/{num_heads} heads")
        logger.info(f"Head dim: {head_dim}, Hidden size: {hidden_size}")

        # Build index masks for each head's slice in the projection matrices
        # For Q/K/V: output dimension is grouped by heads
        # Shape: [hidden_size, hidden_size] = [num_heads * head_dim, hidden_size]
        kept_output_indices = []
        for head_idx in kept_head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            kept_output_indices.extend(range(start, end))

        with torch.no_grad():
            # Slice Q, K, V projections (output dimension)
            new_q_weight = q_lin.weight[kept_output_indices, :].clone()
            new_k_weight = k_lin.weight[kept_output_indices, :].clone()
            new_v_weight = v_lin.weight[kept_output_indices, :].clone()

            new_q_bias = q_lin.bias[kept_output_indices].clone() if q_lin.bias is not None else None
            new_k_bias = k_lin.bias[kept_output_indices].clone() if k_lin.bias is not None else None
            new_v_bias = v_lin.bias[kept_output_indices].clone() if v_lin.bias is not None else None

            # Slice output projection (input dimension)
            # out_lin takes concatenated head outputs as input
            new_out_weight = out_lin.weight[:, kept_output_indices].clone()
            new_out_bias = out_lin.bias.clone() if out_lin.bias is not None else None

        # Create new linear layers
        new_q_lin = GraphSurgery.replace_linear(q_lin, new_q_weight, new_q_bias)
        new_k_lin = GraphSurgery.replace_linear(k_lin, new_k_weight, new_k_bias)
        new_v_lin = GraphSurgery.replace_linear(v_lin, new_v_weight, new_v_bias)
        new_out_lin = GraphSurgery.replace_linear(out_lin, new_out_weight, new_out_bias)

        logger.info(f"New projection shapes: Q/K/V out={new_q_weight.shape[0]}, Out in={new_out_weight.shape[1]}")

        return new_q_lin, new_k_lin, new_v_lin, new_out_lin

    @staticmethod
    def slice_ffn_neurons(
            lin1: nn.Linear,
            lin2: nn.Linear,
            kept_neuron_indices: List[int],
            hidden_size: int,
            ffn_dim: int
    ) -> tuple[nn.Linear, nn.Linear]:
        """Slice FFN layer to keep only specified neurons.
        
        For FFN:
        - lin1: [hidden_size, ffn_dim] (expand)
        - lin2: [ffn_dim, hidden_size] (project back)
        
        Args:
            lin1: First linear layer (expansion)
            lin2: Second linear layer (projection)
            kept_neuron_indices: Indices of neurons to keep
            hidden_size: Hidden dimension
            ffn_dim: FFN intermediate dimension
            
        Returns:
            Tuple of (new_lin1, new_lin2)
        """
        kept_neuron_indices = sorted(kept_neuron_indices)

        logger.info(f"Slicing FFN neurons: keeping {len(kept_neuron_indices)}/{ffn_dim} neurons")

        with torch.no_grad():
            # Slice lin1 output features (rows)
            new_lin1_weight = lin1.weight[kept_neuron_indices, :].clone()
            new_lin1_bias = lin1.bias[kept_neuron_indices].clone() if lin1.bias is not None else None

            # Slice lin2 input features (columns)
            new_lin2_weight = lin2.weight[:, kept_neuron_indices].clone()
            new_lin2_bias = lin2.bias.clone() if lin2.bias is not None else None

        # Create new linear layers
        new_lin1 = GraphSurgery.replace_linear(lin1, new_lin1_weight, new_lin1_bias)
        new_lin2 = GraphSurgery.replace_linear(lin2, new_lin2_weight, new_lin2_bias)

        logger.info(f"New FFN shapes: lin1 out={new_lin1_weight.shape[0]}, lin2 in={new_lin2_weight.shape[1]}")

        return new_lin1, new_lin2

    @staticmethod
    def validate_forward_pass(
            model: nn.Module,
            input_shape: tuple,
            device: str = 'cpu'
    ) -> bool:
        """Validate that model can perform a forward pass after surgery.
        
        Args:
            model: Model to validate
            input_shape: Input tensor shape (batch_size, seq_len, ...)
            device: Device to test on
            
        Returns:
            True if forward pass succeeds, False otherwise
        """
        try:
            model.eval()
            with torch.no_grad():
                # Create dummy input
                dummy_input = torch.randn(input_shape, device=device)

                # Try forward pass
                _ = model(dummy_input)

            logger.info("✓ Forward pass validation successful")
            return True

        except Exception as e:
            logger.error(f"✗ Forward pass validation failed: {e}")
            return False
