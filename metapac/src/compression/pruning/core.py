"""Main TransformerPruner orchestrator class."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from .architecture import detect_architecture, enumerate_attention_modules, enumerate_ffn_modules
from .config import PruningConfig
from .importance import compute_head_importance, compute_ffn_importance
from .policies import resolve_selection_policy
from .physical import apply_physical_pruning
from .soft import apply_soft_pruning

logger = logging.getLogger(__name__)


class TransformerPruner:
    """Orchestrates structured pruning for transformer models.
    
    Supports two pruning modes:
    - Physical (hard) pruning: Actually removes weights and reshapes tensors
    - Soft (logical) pruning: Zeros out weights but keeps tensor shapes
    
    Example:
        >>> config = PruningConfig({
        ...     'enabled': True,
        ...     'method': 'magnitude',
        ...     'head_pruning_ratio': 0.25,
        ...     'ffn_pruning_ratio': 0.25,
        ...     'physical': False
        ... })
        >>> pruner = TransformerPruner(config)
        >>> metadata = pruner.apply_pruning(model, plan, importance_rankings)
    """

    def __init__(self, config: PruningConfig):
        """Initialize pruner.
        
        Args:
            config: Pruning configuration.
        """
        self.config = config
        self.selection_policy = resolve_selection_policy(config.selection_policy)

    def apply_pruning(
            self,
            model: nn.Module,
            plan: Dict[str, str],
            importance_rankings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply structured pruning to transformer model.
        
        Routes to physical or soft pruning based on config.
        
        Args:
            model: PyTorch model.
            plan: Parameter -> zone mapping.
            importance_rankings: Parameter -> importance score.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        if self.config.physical:
            logger.info("Using PHYSICAL pruning (hard pruning with tensor reshaping)")
            return self._apply_physical_pruning(model, importance_rankings)
        else:
            logger.info("Using SOFT pruning (logical pruning by zeroing)")
            return self._apply_soft_pruning(model, importance_rankings)

    def _apply_physical_pruning(
            self,
            model: nn.Module,
            importance_rankings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply physical structured pruning (actually reshape tensors).
        
        Currently not fully implemented - physical pruning needs GraphSurge
ry
        for safe tensor reshaping.
        
        Args:
            model: PyTorch model.
            importance_rankings: Parameter importance scores.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        logger.info("=" * 60)
        logger.info("PHYSICAL STRUCTURED PRUNING (Hard Pruning)")
        logger.info("=" * 60)
        logger.info(f"Method: {self.config.method}")
        logger.info(f"Head pruning ratio: {self.config.head_pruning_ratio:.1%}")
        logger.info(f"FFN pruning ratio: {self.config.ffn_pruning_ratio:.1%}")

        # Count parameters before
        total_params_before = sum(p.numel() for p in model.parameters())

        # Step 1: Detect architecture and enumerate modules
        architecture = detect_architecture(model)
        logger.info(f"Detected architecture: {architecture}")

        attention_specs = enumerate_attention_modules(model)
        ffn_specs = enumerate_ffn_modules(model)

        logger.info(f"Found {len(attention_specs)} attention modules, {len(ffn_specs)} FFN modules")

        # Step 2: Compute importance scores
        head_importance = compute_head_importance(
            attention_specs, importance_rankings, self.config.method
        )
        neuron_importance = compute_ffn_importance(
            ffn_specs, importance_rankings, self.config.method
        )

        # Step 3: Select what to prune
        heads_to_prune = self.selection_policy.select_heads_to_prune(
            attention_specs,
            head_importance,
            self.config.head_pruning_ratio,
            self.config.min_heads_per_layer
        )

        neurons_to_prune = self.selection_policy.select_neurons_to_prune(
            ffn_specs,
            neuron_importance,
            self.config.ffn_pruning_ratio,
            self.config.min_ffn_neurons
        )

        # Step 4: Apply physical pruning
        apply_physical_pruning(
            model,
            attention_specs,
            ffn_specs,
            heads_to_prune,
            neurons_to_prune
        )

        # Count parameters after
        total_params_after = sum(p.numel() for p in model.parameters())

        # Build metadata
        pruning_meta = {
            'method': self.config.method,
            'head_pruning_ratio': self.config.head_pruning_ratio,
            'ffn_pruning_ratio': self.config.ffn_pruning_ratio,
            'global_pruning': self.config.global_pruning,
            'physical': True,
            'architecture': architecture,
            'total_params_before': total_params_before,
            'total_params_after': total_params_after,
            'params_pruned': total_params_before - total_params_after,
            'compression_ratio': total_params_after / total_params_before if total_params_before > 0 else 1.0,
            'pruned_heads': {k: list(v) for k, v in heads_to_prune.items()},
            'pruned_neurons': {k: list(v) for k, v in neurons_to_prune.items()},
            'heads_pruned': sum(len(v) for v in heads_to_prune.values()),
            'neurons_pruned': sum(len(v) for v in neurons_to_prune.values())
        }

        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params_before:,} -> {total_params_after:,}")
        logger.info(f"Compression ratio: {pruning_meta['compression_ratio']:.2%}")
        logger.info("=" * 60)

        return pruning_meta

    def _apply_soft_pruning(
            self,
            model: nn.Module,
            importance_rankings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply soft structured pruning (zero out weights but keep shapes).
        
        Args:
            model: PyTorch model.
            importance_rankings: Parameter importance scores.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        logger.info("=" * 60)
        logger.info("SOFT STRUCTURED TRANSFORMER PRUNING (Zeroing)")
        logger.info("=" * 60)
        logger.info(f"Method: {self.config.method}")
        logger.info(f"Head pruning ratio: {self.config.head_pruning_ratio:.1%}")
        logger.info(f"FFN pruning ratio: {self.config.ffn_pruning_ratio:.1%}")

        # Count parameters before (actual non-zero values)
        total_params_before = sum(p.numel() for p in model.parameters())
        nonzero_before = sum((p != 0).sum().item() for p in model.parameters())

        # Step 1: Detect architecture and enumerate modules
        architecture = detect_architecture(model)
        logger.info(f"Detected architecture: {architecture}")

        attention_specs = enumerate_attention_modules(model)
        ffn_specs = enumerate_ffn_modules(model)

        logger.info(f"Found {len(attention_specs)} attention modules, {len(ffn_specs)} FFN modules")

        # Step 2: Compute importance scores
        head_importance = compute_head_importance(
            attention_specs, importance_rankings, self.config.method
        )
        neuron_importance = compute_ffn_importance(
            ffn_specs, importance_rankings, self.config.method
        )

        # Step 3: Select what to prune
        heads_to_prune = self.selection_policy.select_heads_to_prune(
            attention_specs,
            head_importance,
            self.config.head_pruning_ratio,
            self.config.min_heads_per_layer
        )

        neurons_to_prune = self.selection_policy.select_neurons_to_prune(
            ffn_specs,
            neuron_importance,
            self.config.ffn_pruning_ratio,
            self.config.min_ffn_neurons
        )

        # Step 4: Apply soft pruning (zero weights)
        apply_soft_pruning(
            model,
            attention_specs,
            ffn_specs,
            heads_to_prune,
            neurons_to_prune
        )

        # Count parameters after (actual non-zero values)
        total_params_after = sum(p.numel() for p in model.parameters())
        nonzero_after = sum((p != 0).sum().item() for p in model.parameters())

        # Build metadata
        pruning_meta = {
            'method': self.config.method,
            'head_pruning_ratio': self.config.head_pruning_ratio,
            'ffn_pruning_ratio': self.config.ffn_pruning_ratio,
            'global_pruning': self.config.global_pruning,
            'physical': False,
            'architecture': architecture,
            'total_params_before': total_params_before,
            'total_params_after': total_params_after,
            'nonzero_params_before': nonzero_before,
            'nonzero_params_after': nonzero_after,
            'params_pruned': nonzero_before - nonzero_after,
            'compression_ratio': nonzero_after / nonzero_before if nonzero_before > 0 else 1.0,
            'pruned_heads': {k: list(v) for k, v in heads_to_prune.items()},
            'pruned_neurons': {k: list(v) for k, v in neurons_to_prune.items()},
            'heads_pruned': sum(len(v) for v in heads_to_prune.values()),
            'neurons_pruned': sum(len(v) for v in neurons_to_prune.values())
        }

        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params_before:,} (unchanged)")
        logger.info(f"Non-zero parameters: {nonzero_before:,} -> {nonzero_after:,}")
        logger.info(f"Compression ratio: {pruning_meta['compression_ratio']:.2%}")
        logger.info("=" * 60)

        return pruning_meta
