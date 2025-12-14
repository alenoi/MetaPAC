"""
Structured transformer pruning implementation for MetaPAC.

This module implements structured pruning for transformer models:
1. Calculate importance scores for attention heads and FFN neurons
2. Remove least important components (~20-30%)
3. Adjust model architecture accordingly

Key Features:
- Attention head pruning (prune entire heads)
- FFN neuron pruning (structured pruning of intermediate neurons)
- Importance scoring based on weight magnitude and gradient statistics
- Maintains model structural integrity
- Physical (hard) pruning: actual tensor reshaping and parameter reduction
- Soft (logical) pruning: zeroing units but keeping shapes
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from .graph_surgery import GraphSurgery

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

        # Minimum heads/neurons to keep per layer
        self.min_heads_per_layer = config_dict.get('min_heads_per_layer', 2)
        self.min_ffn_ratio = config_dict.get('min_ffn_ratio', 0.5)  # Keep at least 50% FFN

        # NEW: Physical (hard) pruning vs soft (zeroing) pruning
        # Physical: actually reshape tensors and reduce parameter count
        # Soft: zero out pruned units but keep tensor shapes
        self.physical = config_dict.get('physical', True)  # Default: ON


class TransformerPruner:
    """Structured pruning for transformer models."""

    def __init__(self, config: PruningConfig):
        """Initialize pruner.
        
        Args:
            config: Pruning configuration.
        """
        self.config = config
        self._pruned_heads: Dict[str, List[int]] = {}  # layer -> pruned head indices
        self._pruned_neurons: Dict[str, torch.Tensor] = {}  # layer -> pruned neuron mask

    # =========================================================================
    # Architecture Detection (for physical pruning)
    # =========================================================================

    @staticmethod
    def detect_architecture(model: nn.Module) -> str:
        """Detect model architecture (DistilBERT, BERT, RoBERTa, etc.).
        
        Args:
            model: PyTorch model
            
        Returns:
            Architecture name: 'distilbert', 'bert', 'roberta', 'unknown'
        """
        if hasattr(model, 'distilbert'):
            return 'distilbert'
        elif hasattr(model, 'bert'):
            return 'bert'
        elif hasattr(model, 'roberta'):
            return 'roberta'
        else:
            return 'unknown'

    @staticmethod
    def enumerate_attention_modules(model: nn.Module) -> List[AttentionSpec]:
        """Enumerate all attention modules in model.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of AttentionSpec objects
        """
        arch = TransformerPruner.detect_architecture(model)
        logger.info(f"Detected architecture: {arch}")

        attention_modules = []

        # Get model config
        if hasattr(model, 'config'):
            config = model.config
        elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'config'):
            config = model.distilbert.config
        elif hasattr(model, 'bert') and hasattr(model.bert, 'config'):
            config = model.bert.config
        else:
            logger.warning("Could not find model config, using defaults")
            config = None

        if config:
            num_heads = getattr(config, 'num_attention_heads', 12)
            hidden_size = getattr(config, 'dim', getattr(config, 'hidden_size', 768))
        else:
            num_heads = 12
            hidden_size = 768

        head_dim = hidden_size // num_heads

        # Find attention modules based on architecture
        for name, module in model.named_modules():
            if 'attention' in name and hasattr(module, 'q_lin'):
                # DistilBERT style
                spec = AttentionSpec(
                    module_name=name,
                    module=module,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    head_dim=head_dim,
                    q_lin=module.q_lin,
                    k_lin=module.k_lin,
                    v_lin=module.v_lin,
                    out_lin=module.out_lin
                )
                attention_modules.append(spec)

            elif 'attention' in name and hasattr(module, 'query'):
                # BERT style (self.attention.self.query/key/value, self.attention.output.dense)
                # Find the corresponding output layer
                parent_name = '.'.join(name.split('.')[:-1])
                output_name = f"{parent_name}.output.dense"

                # Get output module
                out_module = None
                for n, m in model.named_modules():
                    if n == output_name:
                        out_module = m
                        break

                if out_module is not None:
                    spec = AttentionSpec(
                        module_name=name,
                        module=module,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        head_dim=head_dim,
                        q_lin=module.query,
                        k_lin=module.key,
                        v_lin=module.value,
                        out_lin=out_module
                    )
                    attention_modules.append(spec)

        logger.info(f"Found {len(attention_modules)} attention modules")
        return attention_modules

    @staticmethod
    def enumerate_ffn_modules(model: nn.Module) -> List[FFNSpec]:
        """Enumerate all FFN modules in model.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of FFNSpec objects
        """
        TransformerPruner.detect_architecture(model)
        ffn_modules = []

        # Get model config
        if hasattr(model, 'config'):
            config = model.config
        elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'config'):
            config = model.distilbert.config
        elif hasattr(model, 'bert') and hasattr(model.bert, 'config'):
            config = model.bert.config
        else:
            config = None

        if config:
            hidden_size = getattr(config, 'dim', getattr(config, 'hidden_size', 768))
            ffn_dim = getattr(config, 'hidden_dim', getattr(config, 'intermediate_size', 3072))
        else:
            hidden_size = 768
            ffn_dim = 3072

        # Find FFN modules based on architecture
        for name, module in model.named_modules():
            if 'ffn' in name and hasattr(module, 'lin1') and hasattr(module, 'lin2'):
                # DistilBERT style
                spec = FFNSpec(
                    module_name=name,
                    module=module,
                    hidden_size=hidden_size,
                    ffn_dim=ffn_dim,
                    lin1=module.lin1,
                    lin2=module.lin2
                )
                ffn_modules.append(spec)

            elif 'intermediate' in name and hasattr(module, 'dense'):
                # BERT style (intermediate.dense + output.dense)
                # Find corresponding output layer
                layer_name = '.'.join(name.split('.')[:-1])
                output_name = f"{layer_name}.output.dense"

                output_module = None
                for n, m in model.named_modules():
                    if n == output_name:
                        output_module = m
                        break

                if output_module is not None:
                    spec = FFNSpec(
                        module_name=layer_name,
                        module=module,
                        hidden_size=hidden_size,
                        ffn_dim=ffn_dim,
                        lin1=module.dense,
                        lin2=output_module
                    )
                    ffn_modules.append(spec)

        logger.info(f"Found {len(ffn_modules)} FFN modules")
        return ffn_modules

    # =========================================================================
    # Original Importance Computation
    # =========================================================================

    def compute_head_importance(
            self,
            model: nn.Module,
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

        # Iterate through model to find attention layers
        for name, module in model.named_modules():
            # DistilBERT attention layers: distilbert.transformer.layer.X.attention
            if 'attention' in name and hasattr(module, 'q_lin'):
                layer_name = name

                # Get number of heads from config
                if hasattr(model, 'config'):
                    num_heads = getattr(model.config, 'num_attention_heads', 12)
                    hidden_size = getattr(model.config, 'dim', 768)
                elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'config'):
                    num_heads = model.distilbert.config.num_attention_heads
                    hidden_size = model.distilbert.config.dim
                else:
                    # Fallback: try to infer from weight shapes
                    q_weight = module.q_lin.weight
                    hidden_size = q_weight.shape[0]
                    # Assume standard head configuration
                    num_heads = 12 if hidden_size == 768 else 8

                head_dim = hidden_size // num_heads

                if method == 'magnitude':
                    # Compute average L2 norm of Q, K, V, O weights per head
                    scores = self._compute_head_magnitude(
                        module, num_heads, head_dim
                    )
                elif method == 'importance':
                    # Use meta-predictor importance scores
                    scores = self._compute_head_importance_from_rankings(
                        layer_name, num_heads, importance_rankings
                    )
                else:
                    # Default to magnitude
                    scores = self._compute_head_magnitude(
                        module, num_heads, head_dim
                    )

                head_importance[layer_name] = scores
                logger.info(f"  {layer_name}: {num_heads} heads, importance shape={scores.shape}")

        return head_importance

    def _compute_head_magnitude(
            self,
            attention_module: nn.Module,
            num_heads: int,
            head_dim: int
    ) -> torch.Tensor:
        """Compute head importance based on weight magnitude.
        
        Args:
            attention_module: Attention module with q_lin, k_lin, v_lin, out_lin.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
        
        Returns:
            Tensor of shape [num_heads] with importance scores.
        """
        scores = torch.zeros(num_heads)

        # Get weight matrices (Q, K, V, O)
        q_weight = attention_module.q_lin.weight.data  # [hidden, hidden]
        k_weight = attention_module.k_lin.weight.data
        v_weight = attention_module.v_lin.weight.data
        out_weight = attention_module.out_lin.weight.data

        # Compute L2 norm per head
        for h in range(num_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim

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
            self,
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
            self,
            model: nn.Module,
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

        # Iterate through model to find FFN layers
        for name, module in model.named_modules():
            # DistilBERT FFN layers: distilbert.transformer.layer.X.ffn
            if 'ffn' in name and hasattr(module, 'lin1') and hasattr(module, 'lin2'):
                layer_name = name

                if method == 'magnitude':
                    scores = self._compute_ffn_magnitude(module)
                elif method == 'importance':
                    scores = self._compute_ffn_importance_from_rankings(
                        layer_name, module, importance_rankings
                    )
                else:
                    scores = self._compute_ffn_magnitude(module)

                neuron_importance[layer_name] = scores
                logger.info(f"  {layer_name}: {len(scores)} neurons, importance shape={scores.shape}")

        return neuron_importance

    def _compute_ffn_magnitude(self, ffn_module: nn.Module) -> torch.Tensor:
        """Compute FFN neuron importance based on weight magnitude.
        
        Args:
            ffn_module: FFN module with lin1, lin2.
        
        Returns:
            Tensor of shape [ffn_dim] with importance scores.
        """
        # lin1: [hidden -> ffn_dim]
        # lin2: [ffn_dim -> hidden]
        lin1_weight = ffn_module.lin1.weight.data  # [ffn_dim, hidden]
        lin2_weight = ffn_module.lin2.weight.data  # [hidden, ffn_dim]

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
            self,
            layer_name: str,
            ffn_module: nn.Module,
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
            ffn_dim = ffn_module.lin1.weight.shape[0]
            return torch.full((ffn_dim,), avg_importance)
        else:
            ffn_dim = ffn_module.lin1.weight.shape[0]
            return torch.ones(ffn_dim)

    def prune_attention_heads(
            self,
            model: nn.Module,
            head_importance: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Prune least important attention heads.
        
        Args:
            model: PyTorch model.
            head_importance: Dict of layer_name -> head_scores.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        logger.info("Pruning attention heads...")
        pruning_meta = {
            'pruned_heads': {},
            'total_heads_before': 0,
            'total_heads_after': 0,
            'heads_pruned': 0
        }

        # Collect all head scores for global pruning
        all_scores = []
        layer_heads = []
        for layer_name, scores in head_importance.items():
            for h_idx, score in enumerate(scores):
                all_scores.append(score.item())
                layer_heads.append((layer_name, h_idx))

        pruning_meta['total_heads_before'] = len(all_scores)

        # Determine pruning threshold
        if self.config.global_pruning:
            # Global: prune lowest X% across all layers
            num_to_prune = int(len(all_scores) * self.config.head_pruning_ratio)
            sorted_indices = np.argsort(all_scores)  # ascending order
            heads_to_prune_global = sorted_indices[:num_to_prune]

            # Group by layer
            heads_to_prune_by_layer = {}
            for idx in heads_to_prune_global:
                layer_name, head_idx = layer_heads[idx]
                if layer_name not in heads_to_prune_by_layer:
                    heads_to_prune_by_layer[layer_name] = []
                heads_to_prune_by_layer[layer_name].append(head_idx)
        else:
            # Per-layer: prune lowest X% in each layer
            heads_to_prune_by_layer = {}
            for layer_name, scores in head_importance.items():
                num_heads = len(scores)
                num_to_prune = max(
                    0,
                    min(
                        int(num_heads * self.config.head_pruning_ratio),
                        num_heads - self.config.min_heads_per_layer
                    )
                )
                if num_to_prune > 0:
                    sorted_indices = torch.argsort(scores)  # ascending
                    heads_to_prune_by_layer[layer_name] = sorted_indices[:num_to_prune].tolist()

        # Apply pruning
        for layer_name, head_indices in heads_to_prune_by_layer.items():
            if not head_indices:
                continue

            # Find the attention module
            attention_module = None
            for name, module in model.named_modules():
                if name == layer_name:
                    attention_module = module
                    break

            if attention_module is None:
                logger.warning(f"Could not find attention module: {layer_name}")
                continue

            # Prune the heads
            self._prune_heads_in_layer(attention_module, head_indices)

            # Record
            pruning_meta['pruned_heads'][layer_name] = head_indices
            pruning_meta['heads_pruned'] += len(head_indices)
            self._pruned_heads[layer_name] = head_indices

            logger.info(f"  Pruned {len(head_indices)} heads from {layer_name}: {head_indices}")

        pruning_meta['total_heads_after'] = pruning_meta['total_heads_before'] - pruning_meta['heads_pruned']
        pruning_ratio = pruning_meta['heads_pruned'] / max(1, pruning_meta['total_heads_before'])
        logger.info(f"Pruned {pruning_meta['heads_pruned']} heads ({pruning_ratio:.1%})")

        return pruning_meta

    def _prune_heads_in_layer(self, attention_module: nn.Module, head_indices: List[int]):
        """Prune specific heads from attention layer by zeroing their weights.
        
        Args:
            attention_module: Attention module.
            head_indices: List of head indices to prune.
        """
        # Get head dimension
        q_weight = attention_module.q_lin.weight.data
        hidden_size = q_weight.shape[0]

        # Infer number of heads
        if hasattr(attention_module, 'n_heads'):
            num_heads = attention_module.n_heads
        else:
            # Assume 12 heads for 768 dim, 8 for 512 dim
            num_heads = 12 if hidden_size == 768 else 8

        head_dim = hidden_size // num_heads

        # Zero out weights for pruned heads
        for h in head_indices:
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim

            # Zero Q, K, V rows (output dims)
            attention_module.q_lin.weight.data[start_idx:end_idx, :] = 0
            attention_module.k_lin.weight.data[start_idx:end_idx, :] = 0
            attention_module.v_lin.weight.data[start_idx:end_idx, :] = 0

            # Zero output columns (input dims)
            attention_module.out_lin.weight.data[:, start_idx:end_idx] = 0

            # Zero biases if present
            if attention_module.q_lin.bias is not None:
                attention_module.q_lin.bias.data[start_idx:end_idx] = 0
            if attention_module.k_lin.bias is not None:
                attention_module.k_lin.bias.data[start_idx:end_idx] = 0
            if attention_module.v_lin.bias is not None:
                attention_module.v_lin.bias.data[start_idx:end_idx] = 0

    def prune_ffn_neurons(
            self,
            model: nn.Module,
            neuron_importance: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Prune least important FFN neurons.
        
        Args:
            model: PyTorch model.
            neuron_importance: Dict of layer_name -> neuron_scores.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        logger.info("Pruning FFN neurons...")
        pruning_meta = {
            'pruned_neurons': {},
            'total_neurons_before': 0,
            'total_neurons_after': 0,
            'neurons_pruned': 0
        }

        # Collect all neuron scores for global pruning
        all_scores = []
        layer_neurons = []
        for layer_name, scores in neuron_importance.items():
            for n_idx, score in enumerate(scores):
                all_scores.append(score.item())
                layer_neurons.append((layer_name, n_idx))

        pruning_meta['total_neurons_before'] = len(all_scores)

        # Determine pruning threshold
        if self.config.global_pruning:
            # Global: prune lowest X% across all layers
            num_to_prune = int(len(all_scores) * self.config.ffn_pruning_ratio)
            sorted_indices = np.argsort(all_scores)
            neurons_to_prune_global = sorted_indices[:num_to_prune]

            # Group by layer
            neurons_to_prune_by_layer = {}
            for idx in neurons_to_prune_global:
                layer_name, neuron_idx = layer_neurons[idx]
                if layer_name not in neurons_to_prune_by_layer:
                    neurons_to_prune_by_layer[layer_name] = []
                neurons_to_prune_by_layer[layer_name].append(neuron_idx)
        else:
            # Per-layer: prune lowest X% in each layer
            neurons_to_prune_by_layer = {}
            for layer_name, scores in neuron_importance.items():
                num_neurons = len(scores)
                min_neurons = int(num_neurons * self.config.min_ffn_ratio)
                num_to_prune = max(
                    0,
                    min(
                        int(num_neurons * self.config.ffn_pruning_ratio),
                        num_neurons - min_neurons
                    )
                )
                if num_to_prune > 0:
                    sorted_indices = torch.argsort(scores)
                    neurons_to_prune_by_layer[layer_name] = sorted_indices[:num_to_prune].tolist()

        # Apply pruning
        for layer_name, neuron_indices in neurons_to_prune_by_layer.items():
            if not neuron_indices:
                continue

            # Find the FFN module
            ffn_module = None
            for name, module in model.named_modules():
                if name == layer_name:
                    ffn_module = module
                    break

            if ffn_module is None:
                logger.warning(f"Could not find FFN module: {layer_name}")
                continue

            # Prune the neurons
            self._prune_neurons_in_layer(ffn_module, neuron_indices)

            # Record
            pruning_meta['pruned_neurons'][layer_name] = neuron_indices
            pruning_meta['neurons_pruned'] += len(neuron_indices)
            self._pruned_neurons[layer_name] = torch.zeros(
                ffn_module.lin1.weight.shape[0], dtype=torch.bool
            )
            self._pruned_neurons[layer_name][neuron_indices] = True

            logger.info(f"  Pruned {len(neuron_indices)} neurons from {layer_name}")

        pruning_meta['total_neurons_after'] = pruning_meta['total_neurons_before'] - pruning_meta['neurons_pruned']
        pruning_ratio = pruning_meta['neurons_pruned'] / max(1, pruning_meta['total_neurons_before'])
        logger.info(f"Pruned {pruning_meta['neurons_pruned']} neurons ({pruning_ratio:.1%})")

        return pruning_meta

    def _prune_neurons_in_layer(self, ffn_module: nn.Module, neuron_indices: List[int]):
        """Prune specific neurons from FFN layer by zeroing their weights.
        
        Args:
            ffn_module: FFN module with lin1, lin2.
            neuron_indices: List of neuron indices to prune.
        """
        # Zero out weights for pruned neurons
        for n_idx in neuron_indices:
            # Zero lin1 row (output of neuron)
            ffn_module.lin1.weight.data[n_idx, :] = 0

            # Zero lin2 column (input from neuron)
            ffn_module.lin2.weight.data[:, n_idx] = 0

            # Zero biases if present
            if ffn_module.lin1.bias is not None:
                ffn_module.lin1.bias.data[n_idx] = 0

    # =========================================================================
    # Helper Methods for Physical Pruning
    # =========================================================================

    def _select_heads_to_prune(
            self,
            head_importance: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """Determine which heads to prune based on importance scores.
        
        Args:
            head_importance: Dict of layer_name -> head_scores
            
        Returns:
            Dict of layer_name -> list of head indices to PRUNE (not keep)
        """
        heads_to_prune_by_layer = {}

        # Collect all head scores for global pruning
        all_scores = []
        layer_heads = []
        for layer_name, scores in head_importance.items():
            for h_idx, score in enumerate(scores):
                all_scores.append(score.item())
                layer_heads.append((layer_name, h_idx))

        # Determine pruning threshold
        if self.config.global_pruning:
            # Global: prune lowest X% across all layers
            num_to_prune = int(len(all_scores) * self.config.head_pruning_ratio)
            sorted_indices = np.argsort(all_scores)  # ascending order
            heads_to_prune_global = sorted_indices[:num_to_prune]

            # Group by layer
            for idx in heads_to_prune_global:
                layer_name, head_idx = layer_heads[idx]
                if layer_name not in heads_to_prune_by_layer:
                    heads_to_prune_by_layer[layer_name] = []
                heads_to_prune_by_layer[layer_name].append(head_idx)
        else:
            # Per-layer: prune lowest X% in each layer
            for layer_name, scores in head_importance.items():
                num_heads = len(scores)
                num_to_prune = max(
                    0,
                    min(
                        int(num_heads * self.config.head_pruning_ratio),
                        num_heads - self.config.min_heads_per_layer
                    )
                )
                if num_to_prune > 0:
                    sorted_indices = torch.argsort(scores)  # ascending
                    heads_to_prune_by_layer[layer_name] = sorted_indices[:num_to_prune].tolist()

        return heads_to_prune_by_layer

    def _select_neurons_to_prune(
            self,
            neuron_importance: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Determine which FFN neurons to prune based on importance scores.
        
        Args:
            neuron_importance: Dict of layer_name -> neuron_scores
            
        Returns:
            Dict of layer_name -> binary mask (1=prune, 0=keep)
        """
        neurons_to_prune_by_layer = {}

        # Collect all neuron scores for global pruning
        all_scores = []
        layer_neurons = []
        for layer_name, scores in neuron_importance.items():
            for n_idx, score in enumerate(scores):
                all_scores.append(score.item())
                layer_neurons.append((layer_name, n_idx))

        # Determine pruning threshold
        if self.config.global_pruning:
            # Global: prune lowest X% across all layers
            num_to_prune = int(len(all_scores) * self.config.ffn_pruning_ratio)
            sorted_indices = np.argsort(all_scores)  # ascending order
            neurons_to_prune_global = sorted_indices[:num_to_prune]

            # Group by layer and create masks
            for layer_name, scores in neuron_importance.items():
                mask = torch.zeros_like(scores)
                neurons_to_prune_by_layer[layer_name] = mask

            for idx in neurons_to_prune_global:
                layer_name, neuron_idx = layer_neurons[idx]
                neurons_to_prune_by_layer[layer_name][neuron_idx] = 1
        else:
            # Per-layer: prune lowest X% in each layer
            for layer_name, scores in neuron_importance.items():
                num_neurons = len(scores)
                min_neurons = int(num_neurons * self.config.min_ffn_ratio)
                num_to_prune = max(
                    0,
                    min(
                        int(num_neurons * self.config.ffn_pruning_ratio),
                        num_neurons - min_neurons
                    )
                )

                mask = torch.zeros_like(scores)
                if num_to_prune > 0:
                    sorted_indices = torch.argsort(scores)  # ascending
                    indices_to_prune = sorted_indices[:num_to_prune]
                    mask[indices_to_prune] = 1

                neurons_to_prune_by_layer[layer_name] = mask

        return neurons_to_prune_by_layer

    # =========================================================================
    # Physical (Hard) Pruning - Actual Tensor Reshaping
    # =========================================================================

    def apply_physical_pruning(
            self,
            model: nn.Module,
            plan: Dict[str, str],
            importance_rankings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply physical structured pruning (actually reshape tensors).
        
        This method:
        1. Computes importance scores for heads and neurons
        2. Determines which to prune
        3. Uses GraphSurgery to reshape and rebind modules
        4. Returns metadata with actual parameter counts
        
        Args:
            model: PyTorch model
            plan: Parameter -> zone mapping
            importance_rankings: Parameter -> importance score
            
        Returns:
            Metadata dict with pruning statistics
        """
        logger.info("=" * 60)
        logger.info("PHYSICAL STRUCTURED PRUNING (Hard Pruning)")
        logger.info("=" * 60)
        logger.info(f"Method: {self.config.method}")
        logger.info(f"Head pruning ratio: {self.config.head_pruning_ratio:.1%}")
        logger.info(f"FFN pruning ratio: {self.config.ffn_pruning_ratio:.1%}")

        pruning_meta = {
            'method': self.config.method,
            'head_pruning_ratio': self.config.head_pruning_ratio,
            'ffn_pruning_ratio': self.config.ffn_pruning_ratio,
            'global_pruning': self.config.global_pruning,
            'physical': True,
            'pruned_heads': {},  # layer -> list of kept head indices
            'pruned_neurons': {},  # layer -> list of kept neuron indices
            'heads_pruned': 0,  # Total count (for backward compat)
            'neurons_pruned': 0  # Total count (for backward compat)
        }

        # Count parameters before
        total_params_before = sum(p.numel() for p in model.parameters())

        # Step 1: Compute importance scores
        head_importance = self.compute_head_importance(
            model, importance_rankings, self.config.method
        )
        neuron_importance = self.compute_ffn_importance(
            model, importance_rankings, self.config.method
        )

        # Step 2: Determine which heads/neurons to prune
        heads_to_prune = self._select_heads_to_prune(head_importance)
        neurons_to_prune = self._select_neurons_to_prune(neuron_importance)

        # Step 3: Enumerate modules
        attention_modules = self.enumerate_attention_modules(model)
        ffn_modules = self.enumerate_ffn_modules(model)

        # Step 4: Apply physical pruning to attention layers
        logger.info("\nPhysical pruning of attention heads:")
        for attn_spec in attention_modules:
            layer_name = attn_spec.module_name
            if layer_name in heads_to_prune:
                pruned_indices = heads_to_prune[layer_name]
                all_indices = set(range(attn_spec.num_heads))
                kept_indices = sorted(list(all_indices - set(pruned_indices)))

                logger.info(f"  {layer_name}: keeping {len(kept_indices)}/{attn_spec.num_heads} heads")

                if len(kept_indices) < self.config.min_heads_per_layer:
                    logger.warning(f"    Keeping at least {self.config.min_heads_per_layer} heads (min constraint)")
                    # Keep top-k by importance
                    layer_importance = head_importance[layer_name]
                    top_k_indices = torch.topk(layer_importance, self.config.min_heads_per_layer).indices.tolist()
                    kept_indices = sorted(top_k_indices)

                # Perform graph surgery
                self._prune_attention_layer_physically(
                    attn_spec, kept_indices
                )

                pruning_meta['pruned_heads'][layer_name] = kept_indices

        # Step 5: Apply physical pruning to FFN layers
        logger.info("\nPhysical pruning of FFN neurons:")
        for ffn_spec in ffn_modules:
            layer_name = ffn_spec.module_name
            if layer_name in neurons_to_prune:
                pruned_mask = neurons_to_prune[layer_name]
                kept_indices = torch.where(pruned_mask == 0)[0].tolist()  # Non-pruned neurons

                min_neurons = int(ffn_spec.ffn_dim * self.config.min_ffn_ratio)
                if len(kept_indices) < min_neurons:
                    logger.warning(f"    Keeping at least {min_neurons} neurons (min constraint)")
                    # Keep top-k by importance
                    layer_importance = neuron_importance[layer_name]
                    top_k_indices = torch.topk(layer_importance, min_neurons).indices.tolist()
                    kept_indices = sorted(top_k_indices)

                logger.info(f"  {layer_name}: keeping {len(kept_indices)}/{ffn_spec.ffn_dim} neurons")

                # Perform graph surgery
                self._prune_ffn_layer_physically(
                    ffn_spec, kept_indices
                )

                pruning_meta['pruned_neurons'][layer_name] = kept_indices

        # Step 6: Calculate pruning statistics
        total_heads_pruned = 0
        total_neurons_pruned = 0

        # Count heads pruned
        for layer_name, kept_indices in pruning_meta['pruned_heads'].items():
            # Find original num_heads for this layer
            for attn_spec in attention_modules:
                if attn_spec.module_name == layer_name:
                    original_heads = attn_spec.num_heads
                    heads_pruned = original_heads - len(kept_indices)
                    total_heads_pruned += heads_pruned
                    break

        # Count neurons pruned
        for layer_name, kept_indices in pruning_meta['pruned_neurons'].items():
            # Find original ffn_dim for this layer
            for ffn_spec in ffn_modules:
                if ffn_spec.module_name == layer_name:
                    original_neurons = ffn_spec.ffn_dim
                    neurons_pruned = original_neurons - len(kept_indices)
                    total_neurons_pruned += neurons_pruned
                    break

        pruning_meta['heads_pruned'] = total_heads_pruned
        pruning_meta['neurons_pruned'] = total_neurons_pruned

        # Step 7: Count parameters after
        total_params_after = sum(p.numel() for p in model.parameters())
        pruning_meta['total_params_before'] = total_params_before
        pruning_meta['total_params_after'] = total_params_after
        pruning_meta['params_pruned'] = total_params_before - total_params_after
        pruning_meta['compression_ratio'] = total_params_after / total_params_before

        logger.info("=" * 60)
        logger.info(f"Total heads pruned: {total_heads_pruned}")
        logger.info(f"Total neurons pruned: {total_neurons_pruned}")
        logger.info(f"Total parameters: {total_params_before:,} -> {total_params_after:,}")
        logger.info(f"Parameters pruned: {pruning_meta['params_pruned']:,}")
        logger.info(f"Compression ratio: {pruning_meta['compression_ratio']:.2%}")
        logger.info("=" * 60)

        return pruning_meta

    def _prune_attention_layer_physically(
            self,
            attn_spec: AttentionSpec,
            kept_head_indices: List[int]
    ) -> None:
        """Physically prune attention heads in a layer.
        
        Args:
            attn_spec: Attention layer specification
            kept_head_indices: Indices of heads to keep (sorted)
        """
        # Use graph surgery to slice projections
        new_q, new_k, new_v, new_out = GraphSurgery.slice_attention_head_projections(
            q_lin=attn_spec.q_lin,
            k_lin=attn_spec.k_lin,
            v_lin=attn_spec.v_lin,
            out_lin=attn_spec.out_lin,
            kept_head_indices=kept_head_indices,
            num_heads=attn_spec.num_heads,
            hidden_size=attn_spec.hidden_size
        )

        # Rebind modules (in-place replacement)
        attn_spec.module.q_lin = new_q
        attn_spec.module.k_lin = new_k
        attn_spec.module.v_lin = new_v
        attn_spec.module.out_lin = new_out

        # Update module config if it has n_heads attribute
        if hasattr(attn_spec.module, 'n_heads'):
            attn_spec.module.n_heads = len(kept_head_indices)
        if hasattr(attn_spec.module, 'num_heads'):
            attn_spec.module.num_heads = len(kept_head_indices)

    def _prune_ffn_layer_physically(
            self,
            ffn_spec: FFNSpec,
            kept_neuron_indices: List[int]
    ) -> None:
        """Physically prune FFN neurons in a layer.
        
        Args:
            ffn_spec: FFN layer specification
            kept_neuron_indices: Indices of neurons to keep (sorted)
        """
        # Use graph surgery to slice FFN layers
        new_lin1, new_lin2 = GraphSurgery.slice_ffn_neurons(
            lin1=ffn_spec.lin1,
            lin2=ffn_spec.lin2,
            kept_neuron_indices=kept_neuron_indices,
            hidden_size=ffn_spec.hidden_size,
            ffn_dim=ffn_spec.ffn_dim
        )

        # Rebind modules (in-place replacement)
        ffn_spec.module.lin1 = new_lin1
        ffn_spec.module.lin2 = new_lin2

    # =========================================================================
    # Main Pruning Entry Point (routes physical vs soft)
    # =========================================================================

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
            return self.apply_physical_pruning(model, plan, importance_rankings)
        else:
            logger.info("Using SOFT pruning (logical pruning by zeroing)")
            return self._apply_soft_pruning(model, plan, importance_rankings)

    def _apply_soft_pruning(
            self,
            model: nn.Module,
            plan: Dict[str, str],
            importance_rankings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply soft structured pruning (zero out weights but keep shapes).
        
        This is the original pruning method - renamed for clarity.
        
        Args:
            model: PyTorch model.
            plan: Parameter -> zone mapping.
            importance_rankings: Parameter -> importance score.
        
        Returns:
            Metadata dict with pruning statistics.
        """
        logger.info("=" * 60)
        logger.info("SOFT STRUCTURED TRANSFORMER PRUNING (Zeroing)")
        logger.info("=" * 60)
        logger.info(f"Method: {self.config.method}")
        logger.info(f"Head pruning ratio: {self.config.head_pruning_ratio:.1%}")
        logger.info(f"FFN pruning ratio: {self.config.ffn_pruning_ratio:.1%}")

        pruning_meta = {
            'method': self.config.method,
            'head_pruning_ratio': self.config.head_pruning_ratio,
            'ffn_pruning_ratio': self.config.ffn_pruning_ratio,
            'global_pruning': self.config.global_pruning,
            'physical': False
        }

        # Step 1: Compute importance scores
        head_importance = self.compute_head_importance(
            model, importance_rankings, self.config.method
        )
        neuron_importance = self.compute_ffn_importance(
            model, importance_rankings, self.config.method
        )

        # Step 2: Prune attention heads
        if head_importance:
            head_meta = self.prune_attention_heads(model, head_importance)
            pruning_meta.update(head_meta)
        else:
            logger.warning("No attention heads found for pruning")

        # Step 3: Prune FFN neurons
        if neuron_importance:
            neuron_meta = self.prune_ffn_neurons(model, neuron_importance)
            pruning_meta.update(neuron_meta)
        else:
            logger.warning("No FFN neurons found for pruning")

        # Calculate total parameter reduction
        total_params_before = sum(p.numel() for p in model.parameters())
        total_params_after = sum((p != 0).sum().item() for p in model.parameters())
        pruning_meta['total_params_before'] = total_params_before
        pruning_meta['total_params_after'] = total_params_after
        pruning_meta['params_pruned'] = total_params_before - total_params_after
        pruning_meta['compression_ratio'] = total_params_after / total_params_before

        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params_before:,} -> {total_params_after:,}")
        logger.info(f"Compression ratio: {pruning_meta['compression_ratio']:.2%}")
        logger.info("=" * 60)

        return pruning_meta


def save_pruning_metadata(meta: Dict[str, Any], output_dir: Path):
    """Save pruning metadata to JSON.
    
    Args:
        meta: Pruning metadata.
        output_dir: Output directory.
    """
    meta_path = output_dir / "pruning_meta.json"

    # Convert any tensors to lists for JSON serialization
    serializable_meta = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            serializable_meta[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_meta[key] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            serializable_meta[key] = value

    with open(meta_path, 'w') as f:
        json.dump(serializable_meta, f, indent=2)

    logger.info(f"Saved pruning metadata to: {meta_path}")


def load_pruning_metadata(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load pruning metadata from JSON.
    
    Args:
        checkpoint_dir: Checkpoint directory.
    
    Returns:
        Pruning metadata dict.
    """
    meta_path = checkpoint_dir / "pruning_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Pruning metadata not found: {meta_path}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return meta
