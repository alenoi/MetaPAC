"""Physical (hard) pruning that reshapes weight tensors."""
from __future__ import annotations

import logging
from typing import Dict, Set

import torch
import torch.nn as nn

from .architecture import AttentionSpec, FFNSpec
from .strategies import resolve_pruning_strategy

logger = logging.getLogger(__name__)


def apply_physical_pruning(
        model: nn.Module,
        attention_specs: list[AttentionSpec],
        ffn_specs: list[FFNSpec],
        heads_to_prune: Dict[str, Set[int]],
        neurons_to_prune: Dict[str, Set[int]]
) -> None:
    """Apply physical pruning by reshaping and removing weights.
    
    This modifies the model architecture by actually removing weights and
    updating layer dimensions.
    
    Args:
        model: PyTorch model to prune.
        attention_specs: List of attention layer specifications.
        ffn_specs: List of FFN layer specifications.
        heads_to_prune: Dict mapping layer_name -> set of head indices to prune.
        neurons_to_prune: Dict mapping layer_name -> set of neuron indices to prune.
    """
    logger.info("Applying physical (hard) pruning")
    strategy = resolve_pruning_strategy(model)

    # Prune attention heads
    for spec in attention_specs:
        layer_name = spec.module_name
        if layer_name in heads_to_prune:
            heads = heads_to_prune[layer_name]
            strategy.prune_attention_physically(spec, heads)
            logger.info(f"  Pruned {len(heads)} heads from {layer_name}")

    # Prune FFN neurons
    for spec in ffn_specs:
        layer_name = spec.module_name
        if layer_name in neurons_to_prune:
            neurons = neurons_to_prune[layer_name]
            strategy.prune_ffn_physically(spec, neurons)
            logger.info(f"  Pruned {len(neurons)} neurons from {layer_name}")

    logger.info("Physical pruning complete")
