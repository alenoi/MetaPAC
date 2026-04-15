"""Soft (logical) pruning that zeros weights without reshaping."""
from __future__ import annotations

import logging
from typing import Dict, Set

import torch
import torch.nn as nn

from .architecture import AttentionSpec, FFNSpec

logger = logging.getLogger(__name__)


def apply_soft_pruning(
        model: nn.Module,
        attention_specs: list[AttentionSpec],
        ffn_specs: list[FFNSpec],
        heads_to_prune: Dict[str, Set[int]],
        neurons_to_prune: Dict[str, Set[int]]
) -> None:
    """Apply soft pruning by zeroing weights (logical pruning).
    
    This keeps the model architecture unchanged but zeros out pruned weights.
    
    Args:
        model: PyTorch model to prune.
        attention_specs: List of attention layer specifications.
        ffn_specs: List of FFN layer specifications.  
        heads_to_prune: Dict mapping layer_name -> set of head indices to prune.
        neurons_to_prune: Dict mapping layer_name -> set of neuron indices to prune.
    """
    logger.info("Applying soft (logical) pruning")

    # Prune attention heads
    for spec in attention_specs:
        layer_name = spec.module_name
        if layer_name in heads_to_prune:
            heads = heads_to_prune[layer_name]
            prune_attention_heads(spec, heads)
            logger.info(f"  Zeroed {len(heads)} heads in {layer_name}")

    # Prune FFN neurons
    for spec in ffn_specs:
        layer_name = spec.module_name
        if layer_name in neurons_to_prune:
            neurons = neurons_to_prune[layer_name]
            prune_ffn_neurons(spec, neurons)
            logger.info(f"  Zeroed {len(neurons)} neurons in {layer_name}")

    logger.info("Soft pruning complete")


def prune_attention_heads(
        spec: AttentionSpec,
        heads_to_prune: Set[int]
) -> None:
    """Zero out attention head weights (soft pruning).
    
    Args:
        spec: Attention layer specification.
        heads_to_prune: Set of head indices to prune.
    """
    # Get modules
    q_lin = spec.q_lin
    k_lin = spec.k_lin
    v_lin = spec.v_lin
    out_lin = spec.out_lin

    # Get dimensions
    hidden_size = q_lin.weight.shape[0]
    num_heads = spec.num_heads
    head_dim = hidden_size // num_heads

    # Zero weights for each head
    with torch.no_grad():
        for h in heads_to_prune:
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim

            # Zero Q, K, V rows (output dims)
            q_lin.weight.data[start_idx:end_idx, :] = 0
            k_lin.weight.data[start_idx:end_idx, :] = 0
            v_lin.weight.data[start_idx:end_idx, :] = 0

            if q_lin.bias is not None:
                q_lin.bias.data[start_idx:end_idx] = 0
            if k_lin.bias is not None:
                k_lin.bias.data[start_idx:end_idx] = 0
            if v_lin.bias is not None:
                v_lin.bias.data[start_idx:end_idx] = 0

            # Zero Out columns (input dims)
            out_lin.weight.data[:, start_idx:end_idx] = 0


def prune_ffn_neurons(
        spec: FFNSpec,
        neurons_to_prune: Set[int]
) -> None:
    """Zero out FFN neuron weights (soft pruning).
    
    Args:
        spec: FFN layer specification.
        neurons_to_prune: Set of neuron indices to prune.
    """
    # Get modules
    lin1 = spec.lin1
    lin2 = spec.lin2

    # Zero weights for each neuron
    with torch.no_grad():
        for n in neurons_to_prune:
            # Zero lin1 row (output = neuron activation)
            lin1.weight.data[n, :] = 0
            if lin1.bias is not None:
                lin1.bias.data[n] = 0

            # Zero lin2 column (input = neuron activation)
            lin2.weight.data[:, n] = 0
