"""Importance scoring for attention heads and FFN neurons."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch

from .strategies import AttentionSpec, FFNSpec

logger = logging.getLogger(__name__)


def compute_head_importance(
    attention_specs: list[AttentionSpec],
        importance_rankings: Dict[str, float],
        method: str = 'magnitude'
) -> Dict[str, torch.Tensor]:
    """Compute importance scores for attention heads.
    
    For DistilBERT/BERT-style models, attention heads are in:
    - Q (query): [hidden_size, hidden_size]
    - K (key): [hidden_size, hidden_size]
    - V (value): [hidden_size, hidden_size]
    - O (output): [hidden_size, hidden_size]
    
    Each head has size hidden_size/num_heads.
    
    Args:
        model: PyTorch model.
        importance_rankings: Parameter importance scores.
        method: Importance scoring method ('magnitude', 'gradient', 'importance').
    
    Returns:
        Dict mapping layer_name -> head_importance_scores (tensor).
    """
    logger.info(f"Computing attention head importance using method: {method}")
    head_importance = {}

    for spec in attention_specs:
        layer_name = spec.module_name
        if method == 'magnitude':
            scores = _compute_head_magnitude(spec)
        elif method == 'importance':
            scores = _compute_head_importance_from_rankings(layer_name, spec.num_heads, importance_rankings)
        else:
            scores = _compute_head_magnitude(spec)

        head_importance[layer_name] = scores
        logger.info(f"  {layer_name}: {spec.num_heads} heads, importance shape={scores.shape}")

    return head_importance


def _compute_head_magnitude(spec: AttentionSpec) -> torch.Tensor:
    """Compute head importance based on weight magnitude.
    
    Args:
        attention_module: Attention module with q_lin, k_lin, v_lin, out_lin.
        num_heads: Number of attention heads.
        head_dim: Dimension of each head.
    
    Returns:
        Tensor of shape [num_heads] with importance scores.
    """
    scores = torch.zeros(spec.num_heads)

    # Get weight matrices (Q, K, V, O)
    q_weight = spec.q_lin.weight.data  # [hidden, hidden]
    k_weight = spec.k_lin.weight.data
    v_weight = spec.v_lin.weight.data
    out_weight = spec.out_lin.weight.data

    # Compute L2 norm per head
    for h in range(spec.num_heads):
        start_idx = h * spec.head_dim
        end_idx = (h + 1) * spec.head_dim

        # Q, K, V: rows correspond to output dims (organized by head)
        q_norm = torch.norm(q_weight[start_idx:end_idx, :])
        k_norm = torch.norm(k_weight[start_idx:end_idx, :])
        v_norm = torch.norm(v_weight[start_idx:end_idx, :])

        # Output: columns correspond to input dims (organized by head)
        o_norm = torch.norm(out_weight[:, start_idx:end_idx])

        # Average norm across Q, K, V, O
        scores[h] = (q_norm + k_norm + v_norm + o_norm) / 4.0

    return scores


def _compute_head_importance_from_rankings(
        layer_name: str,
        num_heads: int,
        importance_rankings: Dict[str, float]
) -> torch.Tensor:
    """Compute head importance from meta-predictor rankings.
    
    Args:
        layer_name: Name of attention layer.
        num_heads: Number of heads.
        importance_rankings: Parameter importance scores.
    
    Returns:
        Tensor of shape [num_heads] with importance scores.
    """
    # Look for attention parameters in this layer
    q_key = f"{layer_name}.q_lin.weight"
    k_key = f"{layer_name}.k_lin.weight"
    v_key = f"{layer_name}.v_lin.weight"
    out_key = f"{layer_name}.out_lin.weight"

    # Average importance scores
    scores = []
    for key in [q_key, k_key, v_key, out_key]:
        if key in importance_rankings:
            scores.append(importance_rankings[key])

    if scores:
        # All heads in this layer get the same importance (avg of Q, K, V, O)
        avg_importance = np.mean(scores)
        return torch.full((num_heads,), avg_importance)
    else:
        # Fallback to equal importance
        return torch.ones(num_heads)


def compute_ffn_importance(
    ffn_specs: list[FFNSpec],
        importance_rankings: Dict[str, float],
        method: str = 'magnitude'
) -> Dict[str, torch.Tensor]:
    """Compute importance scores for FFN neurons.
    
    For DistilBERT FFN layers:
    - FFN layer 1: [hidden_size, ffn_dim] (expands to 4x typically)
    - FFN layer 2: [ffn_dim, hidden_size] (projects back)
    
    Each neuron in the intermediate layer has incoming and outgoing weights.
    
    Args:
        model: PyTorch model.
        importance_rankings: Parameter importance scores.
        method: Importance scoring method.
    
    Returns:
        Dict mapping layer_name -> neuron_importance_scores (tensor).
    """
    logger.info(f"Computing FFN neuron importance using method: {method}")
    neuron_importance = {}

    for spec in ffn_specs:
        layer_name = spec.module_name
        if method == 'magnitude':
            scores = _compute_ffn_magnitude(spec)
        elif method == 'importance':
            scores = _compute_ffn_importance_from_rankings(layer_name, spec, importance_rankings)
        else:
            scores = _compute_ffn_magnitude(spec)

        neuron_importance[layer_name] = scores
        logger.info(f"  {layer_name}: {len(scores)} neurons, importance shape={scores.shape}")

    return neuron_importance


def _compute_ffn_magnitude(spec: FFNSpec) -> torch.Tensor:
    """Compute FFN neuron importance based on weight magnitude.
    
    Args:
        ffn_module: FFN module with lin1, lin2.
    
    Returns:
        Tensor of shape [ffn_dim] with importance scores.
    """
    # lin1: [hidden -> ffn_dim]
    # lin2: [ffn_dim -> hidden]
    lin1_weight = spec.lin1.weight.data  # [ffn_dim, hidden]
    lin2_weight = spec.lin2.weight.data  # [hidden, ffn_dim]

    ffn_dim = lin1_weight.shape[0]
    scores = torch.zeros(ffn_dim)

    # Compute L2 norm of incoming and outgoing weights for each neuron
    for i in range(ffn_dim):
        # Incoming weights (row i of lin1)
        in_norm = torch.norm(lin1_weight[i, :])
        # Outgoing weights (column i of lin2)
        out_norm = torch.norm(lin2_weight[:, i])
        # Average
        scores[i] = (in_norm + out_norm) / 2.0

    return scores


def _compute_ffn_importance_from_rankings(
        layer_name: str,
        spec: FFNSpec,
        importance_rankings: Dict[str, float]
) -> torch.Tensor:
    """Compute FFN neuron importance from meta-predictor rankings.
    
    Args:
        layer_name: Name of FFN layer.
        ffn_module: FFN module.
        importance_rankings: Parameter importance scores.
    
    Returns:
        Tensor of shape [ffn_dim] with importance scores.
    """
    lin1_key = f"{layer_name}.lin1.weight"
    lin2_key = f"{layer_name}.lin2.weight"

    # Average importance of lin1 and lin2
    scores = []
    for key in [lin1_key, lin2_key]:
        if key in importance_rankings:
            scores.append(importance_rankings[key])

    if scores:
        avg_importance = np.mean(scores)
        ffn_dim = spec.lin1.weight.shape[0]
        return torch.full((ffn_dim,), avg_importance)
    else:
        ffn_dim = spec.lin1.weight.shape[0]
        return torch.ones(ffn_dim)
