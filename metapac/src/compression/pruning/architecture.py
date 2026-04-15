"""Compatibility wrappers around pruning architecture strategies."""
from __future__ import annotations

import logging
from typing import List

import torch.nn as nn

from metapac.src.model_profiles import resolve_model_profile_from_model
from .strategies import AttentionSpec, FFNSpec, resolve_pruning_strategy

logger = logging.getLogger(__name__)


def detect_architecture(model: nn.Module) -> str:
    """Detect model architecture (DistilBERT, BERT, RoBERTa, etc.).
    
    Args:
        model: PyTorch model
        
    Returns:
        Architecture name: 'distilbert', 'bert', 'roberta', 'unknown'
    """
    architecture = resolve_model_profile_from_model(model).architecture
    return architecture if architecture != 'generic' else 'unknown'


def enumerate_attention_modules(model: nn.Module) -> List[AttentionSpec]:
    try:
        strategy = resolve_pruning_strategy(model)
    except ValueError:
        logger.warning("No pruning strategy found for model; returning no attention modules")
        return []
    attention_modules = strategy.enumerate_attention_modules(model)
    logger.info("Found %d attention modules", len(attention_modules))
    return attention_modules


def enumerate_ffn_modules(model: nn.Module) -> List[FFNSpec]:
    try:
        strategy = resolve_pruning_strategy(model)
    except ValueError:
        logger.warning("No pruning strategy found for model; returning no FFN modules")
        return []
    ffn_modules = strategy.enumerate_ffn_modules(model)
    logger.info("Found %d FFN modules", len(ffn_modules))
    return ffn_modules
