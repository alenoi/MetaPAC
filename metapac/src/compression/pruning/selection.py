"""Compatibility wrappers for pruning selection policies."""
from __future__ import annotations

from typing import Dict, List, Set

import torch

from .policies import resolve_selection_policy
from .strategies import AttentionSpec, FFNSpec


def select_heads_to_prune(
        attention_specs: List[AttentionSpec],
        head_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_heads_per_layer: int = 1
) -> Dict[str, Set[int]]:
    return resolve_selection_policy("global_threshold").select_heads_to_prune(
        attention_specs,
        head_importance,
        pruning_ratio,
        min_heads_per_layer,
    )


def select_neurons_to_prune(
        ffn_specs: List[FFNSpec],
        neuron_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_neurons_per_layer: int = 256
) -> Dict[str, Set[int]]:
    return resolve_selection_policy("global_threshold").select_neurons_to_prune(
        ffn_specs,
        neuron_importance,
        pruning_ratio,
        min_neurons_per_layer,
    )
