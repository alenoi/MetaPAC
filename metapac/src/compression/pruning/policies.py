"""Selection policies for structured pruning targets."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Set, Type

import torch

from .strategies import AttentionSpec, FFNSpec

logger = logging.getLogger(__name__)


class PruningSelectionPolicy(ABC):
    policy_name: str = "global_threshold"

    @abstractmethod
    def select_heads_to_prune(
        self,
        attention_specs: list[AttentionSpec],
        head_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_heads_per_layer: int = 1,
    ) -> Dict[str, Set[int]]:
        """Select attention heads to prune."""

    @abstractmethod
    def select_neurons_to_prune(
        self,
        ffn_specs: list[FFNSpec],
        neuron_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_neurons_per_layer: int = 256,
    ) -> Dict[str, Set[int]]:
        """Select FFN neurons to prune."""


class GlobalThresholdSelectionPolicy(PruningSelectionPolicy):
    policy_name = "global_threshold"

    def select_heads_to_prune(
        self,
        attention_specs: list[AttentionSpec],
        head_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_heads_per_layer: int = 1,
    ) -> Dict[str, Set[int]]:
        logger.info("Selecting heads to prune (ratio=%.2f%%, min_heads=%d)", pruning_ratio * 100, min_heads_per_layer)

        all_scores: list[float] = []
        for spec in attention_specs:
            layer_name = spec.module_name
            if layer_name not in head_importance:
                logger.warning("No importance scores for %s, skipping", layer_name)
                continue
            all_scores.extend(score.item() for score in head_importance[layer_name])

        if not all_scores:
            logger.warning("No head importance scores available")
            return {}

        all_scores_sorted = sorted(all_scores)
        num_total_heads = len(all_scores)
        num_to_prune = int(num_total_heads * pruning_ratio)
        if num_to_prune == 0:
            logger.info("Pruning ratio results in 0 heads to prune")
            return {}

        threshold = all_scores_sorted[num_to_prune - 1]
        logger.info("Pruning threshold: %.4f (%d/%d heads)", threshold, num_to_prune, num_total_heads)

        heads_to_prune: Dict[str, Set[int]] = {}
        for spec in attention_specs:
            layer_name = spec.module_name
            if layer_name not in head_importance:
                continue

            scores = head_importance[layer_name]
            num_heads = len(scores)
            candidates = [index for index in range(num_heads) if scores[index].item() <= threshold]

            max_prune = num_heads - min_heads_per_layer
            if len(candidates) > max_prune:
                candidates = [
                    index
                    for index, _score in sorted(((idx, scores[idx].item()) for idx in candidates), key=lambda item: item[1])[:max_prune]
                ]

            if candidates:
                heads_to_prune[layer_name] = set(candidates)
                logger.info("  %s: pruning %d/%d heads %s", layer_name, len(candidates), num_heads, candidates)

        total_pruned = sum(len(heads) for heads in heads_to_prune.values())
        logger.info("Selected %d/%d heads to prune", total_pruned, num_total_heads)
        return heads_to_prune

    def select_neurons_to_prune(
        self,
        ffn_specs: list[FFNSpec],
        neuron_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_neurons_per_layer: int = 256,
    ) -> Dict[str, Set[int]]:
        logger.info("Selecting neurons to prune (ratio=%.2f%%, min_neurons=%d)", pruning_ratio * 100, min_neurons_per_layer)

        all_scores: list[float] = []
        for spec in ffn_specs:
            layer_name = spec.module_name
            if layer_name not in neuron_importance:
                logger.warning("No importance scores for %s, skipping", layer_name)
                continue
            all_scores.extend(score.item() for score in neuron_importance[layer_name])

        if not all_scores:
            logger.warning("No neuron importance scores available")
            return {}

        all_scores_sorted = sorted(all_scores)
        num_total_neurons = len(all_scores)
        num_to_prune = int(num_total_neurons * pruning_ratio)
        if num_to_prune == 0:
            logger.info("Pruning ratio results in 0 neurons to prune")
            return {}

        threshold = all_scores_sorted[num_to_prune - 1]
        logger.info("Pruning threshold: %.4f (%d/%d neurons)", threshold, num_to_prune, num_total_neurons)

        neurons_to_prune: Dict[str, Set[int]] = {}
        for spec in ffn_specs:
            layer_name = spec.module_name
            if layer_name not in neuron_importance:
                continue

            scores = neuron_importance[layer_name]
            num_neurons = len(scores)
            candidates = [index for index in range(num_neurons) if scores[index].item() <= threshold]

            max_prune = num_neurons - min_neurons_per_layer
            if max_prune < 0:
                logger.warning("  %s: only %d neurons, skipping (min=%d)", layer_name, num_neurons, min_neurons_per_layer)
                continue

            if len(candidates) > max_prune:
                candidates = [
                    index
                    for index, _score in sorted(((idx, scores[idx].item()) for idx in candidates), key=lambda item: item[1])[:max_prune]
                ]

            if candidates:
                neurons_to_prune[layer_name] = set(candidates)
                logger.info("  %s: pruning %d/%d neurons", layer_name, len(candidates), num_neurons)

        total_pruned = sum(len(neurons) for neurons in neurons_to_prune.values())
        logger.info("Selected %d/%d neurons to prune", total_pruned, num_total_neurons)
        return neurons_to_prune


class PerLayerRatioSelectionPolicy(PruningSelectionPolicy):
    policy_name = "per_layer_ratio"

    def select_heads_to_prune(
        self,
        attention_specs: list[AttentionSpec],
        head_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_heads_per_layer: int = 1,
    ) -> Dict[str, Set[int]]:
        logger.info("Selecting heads per layer (ratio=%.2f%%, min_heads=%d)", pruning_ratio * 100, min_heads_per_layer)
        heads_to_prune: Dict[str, Set[int]] = {}
        total_pruned = 0
        total_heads = 0
        for spec in attention_specs:
            layer_name = spec.module_name
            scores = head_importance.get(layer_name)
            if scores is None:
                logger.warning("No importance scores for %s, skipping", layer_name)
                continue
            num_heads = len(scores)
            total_heads += num_heads
            max_prune = max(0, num_heads - min_heads_per_layer)
            num_to_prune = min(int(num_heads * pruning_ratio), max_prune)
            if num_to_prune <= 0:
                continue
            selected = sorted(range(num_heads), key=lambda index: scores[index].item())[:num_to_prune]
            heads_to_prune[layer_name] = set(selected)
            total_pruned += len(selected)
            logger.info("  %s: pruning %d/%d heads %s", layer_name, len(selected), num_heads, selected)
        logger.info("Selected %d/%d heads to prune", total_pruned, total_heads)
        return heads_to_prune

    def select_neurons_to_prune(
        self,
        ffn_specs: list[FFNSpec],
        neuron_importance: Dict[str, torch.Tensor],
        pruning_ratio: float,
        min_neurons_per_layer: int = 256,
    ) -> Dict[str, Set[int]]:
        logger.info("Selecting neurons per layer (ratio=%.2f%%, min_neurons=%d)", pruning_ratio * 100, min_neurons_per_layer)
        neurons_to_prune: Dict[str, Set[int]] = {}
        total_pruned = 0
        total_neurons = 0
        for spec in ffn_specs:
            layer_name = spec.module_name
            scores = neuron_importance.get(layer_name)
            if scores is None:
                logger.warning("No importance scores for %s, skipping", layer_name)
                continue
            num_neurons = len(scores)
            total_neurons += num_neurons
            max_prune = num_neurons - min_neurons_per_layer
            if max_prune < 0:
                logger.warning("  %s: only %d neurons, skipping (min=%d)", layer_name, num_neurons, min_neurons_per_layer)
                continue
            num_to_prune = min(int(num_neurons * pruning_ratio), max_prune)
            if num_to_prune <= 0:
                continue
            selected = sorted(range(num_neurons), key=lambda index: scores[index].item())[:num_to_prune]
            neurons_to_prune[layer_name] = set(selected)
            total_pruned += len(selected)
            logger.info("  %s: pruning %d/%d neurons", layer_name, len(selected), num_neurons)
        logger.info("Selected %d/%d neurons to prune", total_pruned, total_neurons)
        return neurons_to_prune


class SelectionPolicyRegistry:
    def __init__(self) -> None:
        self._policies: dict[str, Type[PruningSelectionPolicy]] = {}

    def register(self, policy_class: Type[PruningSelectionPolicy]) -> Type[PruningSelectionPolicy]:
        self._policies[policy_class.policy_name] = policy_class
        return policy_class

    def resolve(self, policy_name: str | None) -> PruningSelectionPolicy:
        resolved_name = policy_name or GlobalThresholdSelectionPolicy.policy_name
        policy_class = self._policies.get(resolved_name)
        if policy_class is None:
            raise ValueError(f"Unknown pruning selection policy '{resolved_name}'")
        return policy_class()


_registry = SelectionPolicyRegistry()
_registry.register(GlobalThresholdSelectionPolicy)
_registry.register(PerLayerRatioSelectionPolicy)


def resolve_selection_policy(policy_name: str | None) -> PruningSelectionPolicy:
    policy = _registry.resolve(policy_name)
    logger.info("Resolved pruning selection policy: %s", policy.policy_name)
    return policy