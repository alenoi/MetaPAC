from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from metapac.src.compression.pruning.physical import apply_physical_pruning
from metapac.src.compression.pruning.strategies import resolve_pruning_strategy


class _DistilAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_lin = nn.Linear(8, 8)
        self.k_lin = nn.Linear(8, 8)
        self.v_lin = nn.Linear(8, 8)
        self.out_lin = nn.Linear(8, 8)


class _DistilFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 8)


class DistilLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="distilbert", num_attention_heads=2, dim=8, hidden_dim=16)
        self.encoder = nn.Module()
        self.encoder.attention = _DistilAttention()
        self.encoder.ffn = _DistilFFN()


class _BertSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)


class _BertAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = _BertSelfAttention()
        self.output = nn.Module()
        self.output.dense = nn.Linear(8, 8)


class _BertIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(8, 16)


class _BertOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(16, 8)


class BertLikeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _BertAttentionBlock()
        self.intermediate = _BertIntermediate()
        self.output = _BertOutput()


class BertLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="bert", num_attention_heads=2, hidden_size=8, intermediate_size=16)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([BertLikeLayer()])


def test_resolves_distilbert_pruning_strategy() -> None:
    model = DistilLikeModel()
    strategy = resolve_pruning_strategy(model)

    attention_specs = strategy.enumerate_attention_modules(model)
    ffn_specs = strategy.enumerate_ffn_modules(model)

    assert strategy.architecture_name == "distilbert"
    assert len(attention_specs) == 1
    assert len(ffn_specs) == 1
    assert attention_specs[0].module_name.endswith("attention")
    assert ffn_specs[0].module_name.endswith("ffn")


def test_resolves_bert_pruning_strategy() -> None:
    model = BertLikeModel()
    strategy = resolve_pruning_strategy(model)

    attention_specs = strategy.enumerate_attention_modules(model)
    ffn_specs = strategy.enumerate_ffn_modules(model)

    assert strategy.architecture_name == "bert"
    assert len(attention_specs) == 1
    assert len(ffn_specs) == 1
    assert attention_specs[0].module_name.endswith("attention.self")
    assert ffn_specs[0].module_name.endswith("encoder.layer.0")


def test_physical_pruning_uses_strategy_tensor_rules() -> None:
    model = DistilLikeModel()
    strategy = resolve_pruning_strategy(model)
    attention_specs = strategy.enumerate_attention_modules(model)
    ffn_specs = strategy.enumerate_ffn_modules(model)

    apply_physical_pruning(
        model,
        attention_specs,
        ffn_specs,
        heads_to_prune={attention_specs[0].module_name: {1}},
        neurons_to_prune={ffn_specs[0].module_name: {0, 1, 2, 3}},
    )

    attention = attention_specs[0]
    ffn = ffn_specs[0]

    assert attention.q_lin.weight.shape == torch.Size([4, 8])
    assert attention.out_lin.weight.shape == torch.Size([8, 4])
    assert attention.q_lin.out_features == 4
    assert attention.out_lin.in_features == 4
    assert ffn.lin1.weight.shape == torch.Size([12, 8])
    assert ffn.lin2.weight.shape == torch.Size([8, 12])
    assert ffn.lin1.out_features == 12
    assert ffn.lin2.in_features == 12