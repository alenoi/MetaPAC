"""Quantization registry utilities.

This module provides utilities for building and managing the variable-bit quantization registry:
- Infer bit-width from metadata
- Attach quantization metadata to modules
- Build registry from combined quantization metadata
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import torch.nn as nn

from ..variable_bit_layers import register_quantized_layer, ensure_registry
from .model_loading import get_module_by_name

logger = logging.getLogger(__name__)


def infer_assigned_bits(meta: Dict[str, Any], default_bits: int = 8) -> int:
    """Infer effective bit-width from heterogeneous metadata dicts.

    Tries common keys: 'assigned_bits', 'bits', 'target_bits', 'final_bits'.

    Args:
        meta: Metadata dictionary containing bit-width information
        default_bits: Default bit-width if not found in metadata

    Returns:
        Inferred bit-width as integer.
    """
    for k in ("assigned_bits", "bits", "final_bits", "target_bits"):
        v = meta.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                continue
    return int(default_bits)


def attach_quant_meta_and_register(
    root: nn.Module,
    layer: nn.Module,
    layer_name: str,
    bits: int
) -> None:
    """Attach minimal quantization metadata to a layer and register it on the root model.

    This makes variable-bit export deterministic without scanning the entire tree.

    Args:
        root: Root model (for registry)
        layer: Layer to attach metadata to
        layer_name: Name of the layer
        bits: Bit-width for quantization
    """
    if not hasattr(layer, "weight") or getattr(layer, "weight") is None:
        weight_numel = 0
        shape = None
    else:
        try:
            weight_numel = int(layer.weight.numel())
            shape = tuple(layer.weight.shape)
        except Exception:
            weight_numel = 0
            shape = None

    layer.quant_meta = {
        "name": layer_name,
        "bits": int(bits),
        "weight_numel": weight_numel,
        "shape": shape,
    }
    register_quantized_layer(root, layer)


def build_variable_bit_registry_from_meta(
    model: nn.Module,
    combined_meta: Dict[str, Dict[str, Any]],
    *,
    exclude_layernorm_and_classifier: bool = True,
    fallback_bits: int = 8,
) -> int:
    """Use the union of quantization metadata to attach quant_meta to modules and register them.

    This enables deterministic variable-bit export by explicitly marking which layers are quantized.

    Args:
        model: Root model to register layers on
        combined_meta: Combined quantization metadata (quant_meta + trim_meta)
        exclude_layernorm_and_classifier: Skip these layer types if True
        fallback_bits: Default bit-width for metadata without explicit bits

    Returns:
        Number of successfully registered layers.
    """
    ensure_registry(model)
    name_map = dict(model.named_modules())
    registered = 0

    for name, meta in combined_meta.items():
        # Map parameter names like "...weight" to module names
        candidate_names = [name]
        if name.endswith(".weight") or name.endswith(".bias"):
            candidate_names.append(name.rsplit(".", 1)[0])

        target_module = None
        target_module_name = None
        for cand in candidate_names:
            m = name_map.get(cand)
            if m is None:
                m = get_module_by_name(model, cand)
            if isinstance(m, nn.Module):
                target_module = m
                target_module_name = cand
                break

        if target_module is None:
            # Could not resolve; skip silently but keep going
            continue

        # Optional: exclude LayerNorm and classifier from variable-bit accounting
        if exclude_layernorm_and_classifier:
            clsname = target_module.__class__.__name__
            if "LayerNorm" in clsname or "layernorm" in clsname.lower():
                continue
            if target_module_name and (
                target_module_name.endswith("classifier") 
                or "classifier" in target_module_name
            ):
                continue

        bits = infer_assigned_bits(meta, default_bits=fallback_bits)
        attach_quant_meta_and_register(model, target_module, target_module_name, bits)
        registered += 1

    return registered
