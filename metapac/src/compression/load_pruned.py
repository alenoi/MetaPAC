"""
Load physically pruned models with custom architecture reconstruction.

This module provides utilities to load models that have been physically pruned
with layer-wise heterogeneous architectures.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_physically_pruned_model(
        checkpoint_path: str | Path,
        base_model_name: str = "distilbert-base-uncased",
        device: str = "cpu"
) -> tuple[nn.Module, AutoTokenizer, Dict[str, Any]]:
    """Load a physically pruned model by reconstructing its architecture.
    
    This function:
    1. Loads pruning metadata to understand the architecture
    2. Creates a base model
    3. Replaces pruned layers with correctly-sized modules
    4. Loads the pruned weights
    
    Args:
        checkpoint_path: Path to compressed model directory
        base_model_name: Base model name for architecture template
        device: Device to load on
        
    Returns:
        Tuple of (model, tokenizer, pruning_metadata)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load pruning metadata
    pruning_meta_path = checkpoint_path / "pruning_meta.json"
    if not pruning_meta_path.exists():
        raise FileNotFoundError(f"Pruning metadata not found: {pruning_meta_path}")

    with open(pruning_meta_path) as f:
        pruning_meta = json.load(f)

    if not pruning_meta.get('physical', False):
        raise ValueError("Model was not physically pruned, use standard loading")

    logger.info("Physical pruning detected, reconstructing architecture...")

    # Load base model and tokenizer
    from transformers import AutoModelForSequenceClassification

    logger.info(f"Loading base architecture from: {base_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Get architecture info
    pruned_heads = pruning_meta.get('pruned_heads', {})
    pruned_neurons = pruning_meta.get('pruned_neurons', {})

    logger.info(f"Reconstructing {len(pruned_heads)} attention layers...")
    logger.info(f"Reconstructing {len(pruned_neurons)} FFN layers...")

    # Load state dict
    model_state_path = checkpoint_path / "model_state.pt"
    if not model_state_path.exists():
        raise FileNotFoundError(f"Model state not found: {model_state_path}")

    state_dict = torch.load(model_state_path, map_location='cpu')

    # Reconstruct pruned layers
    model = _reconstruct_pruned_layers(
        model,
        state_dict,
        pruned_heads,
        pruned_neurons
    )

    # CRITICAL: Patch attention forward passes for dynamic head counts
    logger.info("Patching attention forward passes for dynamic head handling...")
    model = _patch_attention_forwards(model, pruned_heads)

    model.to(device)
    model.eval()

    logger.info("Model reconstruction complete!")

    return model, tokenizer, pruning_meta


def _reconstruct_pruned_layers(
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        pruned_heads: Dict[str, list],
        pruned_neurons: Dict[str, list]
) -> nn.Module:
    """Reconstruct pruned layers in the model.
    
    Strategy:
    1. For each pruned layer, replace the modules with correctly-sized ones
    2. Load the pruned weights into the new modules
    
    Args:
        model: Base model
        state_dict: Pruned model state dict
        pruned_heads: Dict of layer_name -> kept_head_indices
        pruned_neurons: Dict of layer_name -> kept_neuron_indices
        
    Returns:
        Model with reconstructed architecture
    """
    from metapac.src.compression.graph_surgery import GraphSurgery

    # Get model config
    if hasattr(model, 'config'):
        config = model.config
    else:
        raise ValueError("Model has no config attribute")

    num_heads = getattr(config, 'num_attention_heads', 12)
    hidden_size = getattr(config, 'dim', getattr(config, 'hidden_size', 768))
    head_dim = hidden_size // num_heads

    logger.info(f"Base architecture: {num_heads} heads, {hidden_size} hidden size, {head_dim} head dim")

    # Reconstruct attention layers
    for layer_name, kept_head_indices in pruned_heads.items():
        logger.info(f"  Reconstructing {layer_name}: {len(kept_head_indices)}/{num_heads} heads")

        # Find the attention module in the model
        attention_module = _get_module_by_name(model, layer_name)
        if attention_module is None:
            logger.warning(f"    Could not find module: {layer_name}")
            continue

        # Get pruned weights from state dict
        q_weight = state_dict[f"{layer_name}.q_lin.weight"]
        k_weight = state_dict[f"{layer_name}.k_lin.weight"]
        v_weight = state_dict[f"{layer_name}.v_lin.weight"]
        out_weight = state_dict[f"{layer_name}.out_lin.weight"]

        q_bias = state_dict.get(f"{layer_name}.q_lin.bias")
        k_bias = state_dict.get(f"{layer_name}.k_lin.bias")
        v_bias = state_dict.get(f"{layer_name}.v_lin.bias")
        out_bias = state_dict.get(f"{layer_name}.out_lin.bias")

        # Create new linear layers with pruned dimensions
        new_q = GraphSurgery.replace_linear(attention_module.q_lin, q_weight, q_bias)
        new_k = GraphSurgery.replace_linear(attention_module.k_lin, k_weight, k_bias)
        new_v = GraphSurgery.replace_linear(attention_module.v_lin, v_weight, v_bias)
        new_out = GraphSurgery.replace_linear(attention_module.out_lin, out_weight, out_bias)

        # Replace modules
        attention_module.q_lin = new_q
        attention_module.k_lin = new_k
        attention_module.v_lin = new_v
        attention_module.out_lin = new_out

        # Update head count if attribute exists
        if hasattr(attention_module, 'n_heads'):
            attention_module.n_heads = len(kept_head_indices)
        if hasattr(attention_module, 'num_heads'):
            attention_module.num_heads = len(kept_head_indices)

        logger.info(f"    ✓ Reconstructed with shapes: Q={q_weight.shape}, Out={out_weight.shape}")

    # Reconstruct FFN layers
    for layer_name, kept_neuron_indices in pruned_neurons.items():
        ffn_dim = len(kept_neuron_indices)
        logger.info(f"  Reconstructing {layer_name}: {ffn_dim} neurons")

        # Find the FFN module
        ffn_module = _get_module_by_name(model, layer_name)
        if ffn_module is None:
            logger.warning(f"    Could not find module: {layer_name}")
            continue

        # Get pruned weights
        lin1_weight = state_dict[f"{layer_name}.lin1.weight"]
        lin2_weight = state_dict[f"{layer_name}.lin2.weight"]

        lin1_bias = state_dict.get(f"{layer_name}.lin1.bias")
        lin2_bias = state_dict.get(f"{layer_name}.lin2.bias")

        # Create new linear layers
        new_lin1 = GraphSurgery.replace_linear(ffn_module.lin1, lin1_weight, lin1_bias)
        new_lin2 = GraphSurgery.replace_linear(ffn_module.lin2, lin2_weight, lin2_bias)

        # Replace modules
        ffn_module.lin1 = new_lin1
        ffn_module.lin2 = new_lin2

        logger.info(f"    ✓ Reconstructed with shapes: Lin1={lin1_weight.shape}, Lin2={lin2_weight.shape}")

    # Load remaining weights (embeddings, classifier, etc.)
    logger.info("Loading remaining weights...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Filter out expected mismatches (pruned layers)
    actual_missing = [k for k in missing_keys if
                      not any(layer in k for layer in list(pruned_heads.keys()) + list(pruned_neurons.keys()))]

    if actual_missing:
        logger.warning(f"Missing keys (non-pruned): {actual_missing[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    return model


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get a module by its full name.
    
    Args:
        model: Model
        name: Module name (e.g., 'distilbert.transformer.layer.0.attention')
        
    Returns:
        Module or None if not found
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


def _patch_attention_forwards(
        model: nn.Module,
        pruned_heads: Dict[str, list]
) -> nn.Module:
    """Patch attention forward passes to handle dynamic head counts.
    
    This completely replaces the forward method with a custom implementation
    that correctly handles pruned head dimensions.
    
    Args:
        model: Model with reconstructed layers
        pruned_heads: Dict of layer_name -> kept_head_indices
        
    Returns:
        Model with patched forward passes
    """
    import types

    logger.info(f"Patching {len(pruned_heads)} attention layers...")

    for layer_name, kept_head_indices in pruned_heads.items():
        attention_module = _get_module_by_name(model, layer_name)
        if attention_module is None:
            logger.warning(f"  Could not find module for patching: {layer_name}")
            continue

        num_kept_heads = len(kept_head_indices)
        logger.info(f"  Patching {layer_name} with {num_kept_heads} heads")

        # Create a completely new forward method
        def create_forward(n_heads, module):
            """Create a forward function for this specific layer."""

            def forward_fn(self, query, key, value, mask=None, head_mask=None, output_attentions=False):
                """
                Custom forward pass with correct number of heads.
                Based on DistilBERT MultiHeadSelfAttention.
                """
                bs = query.size(0)
                q_length = query.size(1)
                k_length = key.size(1)

                # Project Q, K, V (these output the ACTUAL pruned dimensions)
                q = self.q_lin(query)
                k = self.k_lin(key)
                v = self.v_lin(value)

                # Get the ACTUAL output dimension from the projection (not self.dim which is 768!)
                actual_dim = q.size(-1)  # This is the pruned dimension (e.g., 384 or 128)
                dim_per_head = actual_dim // n_heads

                # Project Q, K, V (these projections output pruned_total_dim, not 768!)
                q = self.q_lin(query)
                k = self.k_lin(key)
                v = self.v_lin(value)

                # Reshape for multi-head attention
                # q has shape [bs, q_length, pruned_total_dim]
                q = q.view(bs, q_length, n_heads, dim_per_head).transpose(1, 2)
                k = k.view(bs, k_length, n_heads, dim_per_head).transpose(1, 2)
                v = v.view(bs, k_length, n_heads, dim_per_head).transpose(1, 2)

                # Scaled dot-product attention
                q = q / (dim_per_head ** 0.5)
                scores = torch.matmul(q, k.transpose(-2, -1))

                # Apply attention mask
                if mask is not None:
                    # Reshape mask: (bs, k_length) -> (bs, 1, 1, k_length) -> (bs, n_heads, q_length, k_length)
                    mask_reshp = (bs, 1, 1, k_length)
                    mask = (mask == 0).view(mask_reshp).expand(bs, n_heads, q_length, k_length)
                    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

                # Attention weights
                weights = torch.nn.functional.softmax(scores, dim=-1)
                weights = self.dropout(weights)

                # Apply head mask if provided
                if head_mask is not None:
                    weights = weights * head_mask

                # Apply attention to values
                context = torch.matmul(weights, v)
                context = context.transpose(1, 2).contiguous()
                context = context.view(bs, q_length, actual_dim)

                # Output projection
                output = self.out_lin(context)

                if output_attentions:
                    return (output, weights)
                else:
                    return (output,)

            return forward_fn

        # Create and bind the new forward method
        new_forward = create_forward(num_kept_heads, attention_module)
        attention_module.forward = types.MethodType(new_forward, attention_module)

    logger.info("Attention forward passes patched successfully")
    return model


# Example usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_pruned.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    print(f"Loading physically pruned model from: {checkpoint_path}")
    model, tokenizer, pruning_meta = load_physically_pruned_model(
        checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\n[OK] Model loaded successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Pruned heads: {len(pruning_meta['pruned_heads'])} layers")
    print(f"  Pruned FFN: {len(pruning_meta['pruned_neurons'])} layers")

    # Test forward pass
    print("\nTesting forward pass...")
    inputs = tokenizer("This is a test", return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"[OK] Forward pass successful! Output shape: {outputs.logits.shape}")
