"""Pruning configuration."""
from __future__ import annotations

from typing import Dict, Any


class PruningConfig:
    """Configuration for structured pruning."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize pruning configuration.
        
        Args:
            config_dict: Configuration dictionary with pruning settings.
        """
        self.enabled = config_dict.get('enabled', False)
        self.method = config_dict.get('method', 'magnitude')  # magnitude, gradient, taylor
        self.head_pruning_ratio = config_dict.get('head_pruning_ratio', 0.25)  # Prune 25% heads
        self.ffn_pruning_ratio = config_dict.get('ffn_pruning_ratio', 0.25)  # Prune 25% FFN neurons
        self.normalize_scores = config_dict.get('normalize_scores', True)
        self.global_pruning = config_dict.get('global_pruning', True)  # Global vs per-layer
        self.selection_policy = config_dict.get(
            'selection_policy',
            'global_threshold' if self.global_pruning else 'per_layer_ratio'
        )

        # Minimum heads/neurons to keep per layer
        self.min_heads_per_layer = config_dict.get('min_heads_per_layer', 2)
        self.min_ffn_ratio = config_dict.get('min_ffn_ratio', 0.5)  # Keep at least 50% FFN
        self.min_ffn_neurons = config_dict.get('min_ffn_neurons', 256)  # Minimum neurons per layer

        # NEW: Physical (hard) pruning vs soft (zeroing) pruning
        # Physical: actually reshape tensors and reduce parameter count
        # Soft: zero out pruned units but keep tensor shapes
        self.physical = config_dict.get('physical', True)  # Default: ON
