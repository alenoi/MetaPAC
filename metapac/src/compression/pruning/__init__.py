"""Structured pruning for transformer models.

This package provides functionality for pruning attention heads and FFN neurons
in transformer models. It supports both soft (logical) pruning by zeroing weights
and physical (hard) pruning by actually removing parameters.

Public API:
    - PruningConfig: Configuration for pruning
    - TransformerPruner: Main pruning orchestrator
    - save_pruning_metadata: Save pruning results to JSON
    - load_pruning_metadata: Load pruning results from JSON

Example:
    >>> from metapac.src.compression.pruning import PruningConfig, TransformerPruner
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
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from .config import PruningConfig
from .core import TransformerPruner

logger = logging.getLogger(__name__)


__all__ = [
    'PruningConfig',
    'TransformerPruner',
    'save_pruning_metadata',
    'load_pruning_metadata',
]


def save_pruning_metadata(meta: Dict[str, Any], output_dir: Path) -> None:
    """Save pruning metadata to JSON.
    
    Args:
        meta: Pruning metadata dict.
        output_dir: Output directory path.
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
        checkpoint_dir: Checkpoint directory containing pruning_meta.json.
    
    Returns:
        Pruning metadata dict.
        
    Raises:
        FileNotFoundError: If pruning_meta.json does not exist.
    """
    meta_path = checkpoint_dir / "pruning_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Pruning metadata not found: {meta_path}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return meta
