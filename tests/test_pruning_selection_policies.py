from __future__ import annotations

import torch

from metapac.src.compression.pruning.policies import resolve_selection_policy
from metapac.src.compression.pruning.strategies import AttentionSpec, FFNSpec


def _attention_spec(name: str, num_heads: int) -> AttentionSpec:
    return AttentionSpec(
        module_name=name,
        module=None,
        num_heads=num_heads,
        hidden_size=num_heads * 4,
        head_dim=4,
        q_lin=None,
        k_lin=None,
        v_lin=None,
        out_lin=None,
    )


def _ffn_spec(name: str, ffn_dim: int) -> FFNSpec:
    return FFNSpec(
        module_name=name,
        module=None,
        hidden_size=8,
        ffn_dim=ffn_dim,
        lin1=None,
        lin2=None,
    )


def test_resolve_selection_policy_supports_per_layer_ratio() -> None:
    policy = resolve_selection_policy("per_layer_ratio")
    assert policy.policy_name == "per_layer_ratio"


def test_per_layer_ratio_policy_prunes_each_layer_independently() -> None:
    policy = resolve_selection_policy("per_layer_ratio")
    specs = [_attention_spec("layer_a", 4), _attention_spec("layer_b", 4)]
    importance = {
        "layer_a": torch.tensor([0.1, 0.2, 0.9, 1.0]),
        "layer_b": torch.tensor([0.3, 0.4, 0.8, 0.7]),
    }

    result = policy.select_heads_to_prune(specs, importance, pruning_ratio=0.5, min_heads_per_layer=2)

    assert result == {"layer_a": {0, 1}, "layer_b": {0, 1}}


def test_global_threshold_policy_preserves_legacy_global_behavior() -> None:
    policy = resolve_selection_policy("global_threshold")
    specs = [_ffn_spec("layer_a", 4), _ffn_spec("layer_b", 4)]
    importance = {
        "layer_a": torch.tensor([0.1, 0.2, 0.9, 1.0]),
        "layer_b": torch.tensor([0.3, 0.4, 0.8, 0.7]),
    }

    result = policy.select_neurons_to_prune(specs, importance, pruning_ratio=0.25, min_neurons_per_layer=2)

    assert result == {"layer_a": {0, 1}}