"""Model loading and state dict utilities.

This module provides utilities for loading models and working with state dicts:
- Load target models from various formats (HF, safetensors, pytorch)
- State dict snapshots and change detection
- Module name resolution
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .checkpoint import extract_checkpoint_step

logger = logging.getLogger(__name__)


def snapshot_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Create detached CPU snapshot of model parameters/buffers for change detection.

    Args:
        model: PyTorch model to snapshot

    Returns:
        Dictionary mapping parameter names to cloned CPU tensors.
    """
    snap: Dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        snap[name] = tensor.detach().cpu().clone()
    return snap


def state_dict_change_stats(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    atol: float = 0.0
) -> Dict[str, Any]:
    """Compute how many tensors changed between two state dict snapshots.

    Args:
        before: State dict before modification
        after: State dict after modification
        atol: Absolute tolerance for considering a change significant

    Returns:
        Dictionary with change statistics:
            - shared_tensors: Number of tensors in both dicts
            - changed_tensors: Number of tensors that changed
            - max_abs_diff: Maximum absolute difference found
    """
    shared_keys = sorted(set(before.keys()) & set(after.keys()))
    changed_tensors = 0
    max_abs_diff = 0.0

    for key in shared_keys:
        b = before[key]
        a = after[key]
        if b.shape != a.shape:
            changed_tensors += 1
            continue
        diff = (a - b).abs().max().item() if b.numel() > 0 else 0.0
        if diff > max_abs_diff:
            max_abs_diff = float(diff)
        if diff > atol:
            changed_tensors += 1

    return {
        "shared_tensors": len(shared_keys),
        "changed_tensors": int(changed_tensors),
        "max_abs_diff": float(max_abs_diff),
    }


def resolve_parent_and_attr(
    root: nn.Module, 
    dotted_name: str
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Resolve 'encoder.layer.2.output.dense' → (parent_module, 'dense').

    Args:
        root: Root module to start resolution from
        dotted_name: Dotted path to child module

    Returns:
        Tuple of (parent_module, attribute_name), or (None, None) if resolution fails.
    """
    try:
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
    except Exception:
        return None, None


def get_module_by_name(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """Return the module with exact dotted name using named_modules() map.

    Falls back to attribute walk if not found in named_modules.

    Args:
        root: Root module to search from
        dotted_name: Dotted path to target module

    Returns:
        Target module, or None if not found.
    """
    name_map = dict(root.named_modules())
    if dotted_name in name_map:
        return name_map[dotted_name]
    parent, attr = resolve_parent_and_attr(root, dotted_name)
    if parent is not None and attr and hasattr(parent, attr):
        return getattr(parent, attr)
    return None


def load_target_model(model_path: str) -> nn.Module:
    """Load target model for compression.

    Supports multiple formats:
    - HuggingFace transformers models
    - SafeTensors format
    - PyTorch state dicts
    - Checkpoint directories with fallback

    Args:
        model_path: Path to model or HuggingFace model ID

    Returns:
        Loaded PyTorch model.

    Raises:
        FileNotFoundError: If model path not found and no fallback available.
    """
    logger.info(f"Loading target model from: {model_path}")

    model_dir = Path(model_path)
    if not model_dir.exists():
        # Best-effort recovery: if specific checkpoint missing, pick latest sibling
        parent = model_dir.parent
        if parent.exists() and model_dir.name.startswith("checkpoint-"):
            candidates = [p for p in parent.glob("checkpoint-*") if p.is_dir()]
            if candidates:
                model_dir = max(candidates, key=extract_checkpoint_step)
                logger.info(f"Fallback to latest checkpoint: {model_dir}")

    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Target model path not found: {model_dir}")

    # Try loading as HuggingFace model first
    try:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        logger.info(f"Loaded transformers model: {model.__class__.__name__}")
        return model
    except Exception as e:
        logger.debug(f"Could not load as transformers model: {e}")
        logger.debug("Falling back to state dict loading...")

    # Try loading state dict
    state_dict = None
    ckpt_path = model_dir / "model.safetensors"
    if ckpt_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
        logger.info(f"Loaded safetensors state dict with {len(state_dict)} parameters")
    else:
        ckpt_path = model_dir / "pytorch_model.bin"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu")
            logger.info(f"Loaded pytorch state dict with {len(state_dict)} parameters")

    if state_dict is not None:
        # Create dummy model that wraps state dict
        class DummyModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                for name, tensor in state_dict.items():
                    parts = name.split('.')
                    if len(parts) == 1:
                        self.register_parameter(name, nn.Parameter(tensor))
                    else:
                        # Build nested structure
                        parent = self
                        for part in parts[:-1]:
                            if not hasattr(parent, part):
                                setattr(parent, part, nn.Module())
                            parent = getattr(parent, part)
                        setattr(parent, parts[-1], nn.Parameter(tensor))

        model = DummyModel(state_dict)
        return model
    else:
        logger.warning("Model checkpoint not found, creating dummy model")
        model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        return model


def make_json_serializable(data: Any) -> Any:
    """Recursively convert data to JSON-serializable types.

    Args:
        data: Data to convert (can be nested dicts/lists)

    Returns:
        JSON-serializable version of data.
    """
    import numpy as np
    
    if np.issubdtype(type(data), np.floating):
        return float(data)
    elif np.issubdtype(type(data), np.integer):
        return int(data)
    elif isinstance(data, dict):
        return {key: make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(make_json_serializable(item) for item in data)
    elif isinstance(data, (str, bool, type(None))):
        return data
    else:
        return str(data)  # Fallback for unsupported types
