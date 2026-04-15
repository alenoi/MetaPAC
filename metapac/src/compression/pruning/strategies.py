"""Architecture strategies for structured pruning."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Type

import torch
import torch.nn as nn

from metapac.src.model_profiles import resolve_model_profile_from_model

logger = logging.getLogger(__name__)


@dataclass
class AttentionSpec:
    """Specification for an attention layer."""
    module_name: str
    module: nn.Module
    num_heads: int
    hidden_size: int
    head_dim: int
    q_lin: nn.Linear
    k_lin: nn.Linear
    v_lin: nn.Linear
    out_lin: nn.Linear


@dataclass
class FFNSpec:
    """Specification for a feed-forward network layer."""
    module_name: str
    module: nn.Module
    hidden_size: int
    ffn_dim: int
    lin1: nn.Linear
    lin2: nn.Linear


class PruningArchitectureStrategy(ABC):
    architecture_name: str = "generic"

    @abstractmethod
    def matches(self, model: nn.Module) -> bool:
        """Return True if the strategy can enumerate modules for this model."""

    @abstractmethod
    def enumerate_attention_modules(self, model: nn.Module) -> List[AttentionSpec]:
        """Return attention pruning specs for the model."""

    @abstractmethod
    def enumerate_ffn_modules(self, model: nn.Module) -> List[FFNSpec]:
        """Return FFN pruning specs for the model."""

    def prune_attention_physically(self, spec: AttentionSpec, heads_to_prune: Set[int]) -> None:
        """Remove attention head weights from the layer tensors."""
        hidden_size = spec.q_lin.weight.shape[0]
        num_heads = spec.num_heads
        head_dim = hidden_size // num_heads

        all_heads = set(range(num_heads))
        heads_to_keep = sorted(all_heads - heads_to_prune)
        if not heads_to_keep:
            raise ValueError(f"Cannot prune all heads from {spec.module_name}")

        keep_indices: list[int] = []
        for head_index in heads_to_keep:
            start_idx = head_index * head_dim
            end_idx = (head_index + 1) * head_dim
            keep_indices.extend(range(start_idx, end_idx))

        keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
        with torch.no_grad():
            spec.q_lin.weight.data = spec.q_lin.weight.data[keep_tensor, :]
            spec.k_lin.weight.data = spec.k_lin.weight.data[keep_tensor, :]
            spec.v_lin.weight.data = spec.v_lin.weight.data[keep_tensor, :]

            if spec.q_lin.bias is not None:
                spec.q_lin.bias.data = spec.q_lin.bias.data[keep_tensor]
            if spec.k_lin.bias is not None:
                spec.k_lin.bias.data = spec.k_lin.bias.data[keep_tensor]
            if spec.v_lin.bias is not None:
                spec.v_lin.bias.data = spec.v_lin.bias.data[keep_tensor]

            spec.out_lin.weight.data = spec.out_lin.weight.data[:, keep_tensor]

        new_hidden = len(keep_indices)
        spec.q_lin.out_features = new_hidden
        spec.k_lin.out_features = new_hidden
        spec.v_lin.out_features = new_hidden
        spec.out_lin.in_features = new_hidden

    def prune_ffn_physically(self, spec: FFNSpec, neurons_to_prune: Set[int]) -> None:
        """Remove FFN neuron weights from the layer tensors."""
        ffn_dim = spec.lin1.weight.shape[0]
        all_neurons = set(range(ffn_dim))
        neurons_to_keep = sorted(all_neurons - neurons_to_prune)
        if not neurons_to_keep:
            raise ValueError(f"Cannot prune all neurons from {spec.module_name}")

        keep_tensor = torch.tensor(neurons_to_keep, dtype=torch.long)
        with torch.no_grad():
            spec.lin1.weight.data = spec.lin1.weight.data[keep_tensor, :]
            if spec.lin1.bias is not None:
                spec.lin1.bias.data = spec.lin1.bias.data[keep_tensor]
            spec.lin2.weight.data = spec.lin2.weight.data[:, keep_tensor]

        new_ffn_dim = len(neurons_to_keep)
        spec.lin1.out_features = new_ffn_dim
        spec.lin2.in_features = new_ffn_dim

    def _get_model_config(self, model: nn.Module):
        if hasattr(model, "config"):
            return model.config
        return None


class DistilBertPruningStrategy(PruningArchitectureStrategy):
    architecture_name = "distilbert"

    def matches(self, model: nn.Module) -> bool:
        return resolve_model_profile_from_model(model).architecture == self.architecture_name

    def enumerate_attention_modules(self, model: nn.Module) -> List[AttentionSpec]:
        config = self._get_model_config(model)
        num_heads = getattr(config, "num_attention_heads", 12) if config else 12
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768)) if config else 768
        head_dim = hidden_size // num_heads

        specs: List[AttentionSpec] = []
        for name, module in model.named_modules():
            if "attention" in name and hasattr(module, "q_lin"):
                specs.append(
                    AttentionSpec(
                        module_name=name,
                        module=module,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        head_dim=head_dim,
                        q_lin=module.q_lin,
                        k_lin=module.k_lin,
                        v_lin=module.v_lin,
                        out_lin=module.out_lin,
                    )
                )
        return specs

    def enumerate_ffn_modules(self, model: nn.Module) -> List[FFNSpec]:
        config = self._get_model_config(model)
        hidden_size = getattr(config, "dim", getattr(config, "hidden_size", 768)) if config else 768
        ffn_dim = getattr(config, "hidden_dim", getattr(config, "intermediate_size", 3072)) if config else 3072

        specs: List[FFNSpec] = []
        for name, module in model.named_modules():
            if "ffn" in name and hasattr(module, "lin1") and hasattr(module, "lin2"):
                specs.append(
                    FFNSpec(
                        module_name=name,
                        module=module,
                        hidden_size=hidden_size,
                        ffn_dim=ffn_dim,
                        lin1=module.lin1,
                        lin2=module.lin2,
                    )
                )
        return specs


class BertPruningStrategy(PruningArchitectureStrategy):
    architecture_name = "bert"

    def matches(self, model: nn.Module) -> bool:
        return resolve_model_profile_from_model(model).architecture in {"bert", "roberta"}

    def enumerate_attention_modules(self, model: nn.Module) -> List[AttentionSpec]:
        config = self._get_model_config(model)
        num_heads = getattr(config, "num_attention_heads", 12) if config else 12
        hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768)) if config else 768
        head_dim = hidden_size // num_heads

        specs: List[AttentionSpec] = []
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if "attention" in name and hasattr(module, "query"):
                parent_name = ".".join(name.split(".")[:-1])
                out_module = modules.get(f"{parent_name}.output.dense")
                if out_module is None:
                    continue
                specs.append(
                    AttentionSpec(
                        module_name=name,
                        module=module,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        head_dim=head_dim,
                        q_lin=module.query,
                        k_lin=module.key,
                        v_lin=module.value,
                        out_lin=out_module,
                    )
                )
        return specs

    def enumerate_ffn_modules(self, model: nn.Module) -> List[FFNSpec]:
        config = self._get_model_config(model)
        hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768)) if config else 768
        ffn_dim = getattr(config, "intermediate_size", getattr(config, "hidden_dim", 3072)) if config else 3072

        specs: List[FFNSpec] = []
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if "intermediate" in name and hasattr(module, "dense"):
                layer_name = ".".join(name.split(".")[:-1])
                output_module = modules.get(f"{layer_name}.output.dense")
                if output_module is None:
                    continue
                specs.append(
                    FFNSpec(
                        module_name=layer_name,
                        module=module,
                        hidden_size=hidden_size,
                        ffn_dim=ffn_dim,
                        lin1=module.dense,
                        lin2=output_module,
                    )
                )
        return specs


class PruningStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: list[Type[PruningArchitectureStrategy]] = []

    def register(self, strategy_class: Type[PruningArchitectureStrategy]) -> Type[PruningArchitectureStrategy]:
        self._strategies.append(strategy_class)
        return strategy_class

    def resolve(self, model: nn.Module) -> PruningArchitectureStrategy:
        for strategy_class in self._strategies:
            strategy = strategy_class()
            if strategy.matches(model):
                return strategy
        raise ValueError(f"No pruning strategy available for architecture '{resolve_model_profile_from_model(model).architecture}'")


_registry = PruningStrategyRegistry()
_registry.register(DistilBertPruningStrategy)
_registry.register(BertPruningStrategy)


def resolve_pruning_strategy(model: nn.Module) -> PruningArchitectureStrategy:
    strategy = _registry.resolve(model)
    logger.info("Resolved pruning strategy: %s", strategy.architecture_name)
    return strategy