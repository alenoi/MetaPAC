"""Pruning phase for compression pipeline.

Applies structured and/or unstructured pruning to model parameters
based on the compression plan from the preparation phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from ..pipeline.phase_base import CompressionPhase, PhaseContext
from ..pruning import PruningConfig, TransformerPruner, save_pruning_metadata
from ...utils.logging_utils import get_logger, log_section, log_metric

logger = get_logger(__name__)


def _apply_parameter_zero_pruning(
    model: nn.Module,
    plan: Dict[str, str],
    prune_ratio: float = 0.1,
) -> Dict[str, Any]:
    """Fallback pruning for unsupported architectures.
    
    Instead of zeroing whole prune-zone tensors (too destructive), zero only the
    lowest-magnitude fraction inside each prune-zone tensor.
    
    Args:
        model: Model to prune
        plan: Parameter name -> action mapping ('prune' for pruning targets)
        prune_ratio: Fraction of parameters to zero within each pruned tensor
        
    Returns:
        Metadata dictionary with pruning statistics
    """
    touched_tensors = 0
    zeroed_elements = 0
    total_elements = 0

    # Keep ratio in a sane range
    ratio = float(max(0.0, min(0.95, prune_ratio)))

    if ratio <= 0.0:
        return {
            "parameter_zero_pruning": True,
            "parameters_zeroed": 0,
            "elements_zeroed": 0,
            "elements_total": 0,
            "effective_sparsity": 0.0,
            "fallback_prune_ratio": ratio,
        }

    for param_name, param_tensor in model.named_parameters():
        action = plan.get(param_name, "keep")
        if action != "prune":
            continue

        flat = param_tensor.data.view(-1)
        n = flat.numel()
        if n == 0:
           continue

        total_elements += n

        # Compute how many lowest-magnitude elements to zero
        k = max(1, int(ratio * n))
        abs_vals = flat.abs()
        threshold = abs_vals.kthvalue(k).values.item()

        # Zero out elements below threshold
        mask = abs_vals <= threshold
        flat[mask] = 0.0
        touched_tensors += 1
        zeroed_elements += int(mask.sum().item())

    effective_sparsity = (zeroed_elements / total_elements * 100.0) if total_elements > 0 else 0.0

    return {
        "parameter_zero_pruning": True,
        "parameters_zeroed": touched_tensors,
        "elements_zeroed": zeroed_elements,
        "elements_total": total_elements,
        "effective_sparsity": effective_sparsity,
        "fallback_prune_ratio": ratio,
    }


class PruningPhase(CompressionPhase):
    """Pruning phase: Remove or zero unimportant parameters/structures.
    
    Applies:
    1. Structured pruning (heads, neurons) for supported architectures
    2. Unstructured (magnitude-based) fallback for others
    """
    
    def __init__(self, phase_config: Dict[str, Any]):
        super().__init__(phase_config)
        self.prune_config = PruningConfig(phase_config) if phase_config.get('enabled') else None
    
    def get_phase_name(self) -> str:
        return "Pruning"
    
    def validate(self, context: PhaseContext) -> None:
        """Validate pruning preconditions."""
        if context.model is None:
            raise ValueError("Model not loaded in context")
        if context.plan is None:
            raise ValueError("Compression plan not available")
        if context.importance_rankings is None:
            raise ValueError("Importance rankings not available")
    
    def execute(self, context: PhaseContext) -> PhaseContext:
        """Execute pruning phase."""
        if not self.enabled:
            logger.info("Pruning disabled, skipping")
            context.metadata['pruning'] = {'enabled': False}
            return context
        
        model = context.model
        plan = context.plan
        importance_rankings = context.importance_rankings
        
        pruning_meta = {}
        
        try:
            pruner = TransformerPruner(self.prune_config)
            
            logger.info(f"Applying structured pruning to prune zone...")
            logger.info(f"Method: {self.prune_config.method}")
            logger.info(f"Head pruning ratio: {self.prune_config.head_pruning_ratio:.1%}")
            logger.info(f"FFN pruning ratio: {self.prune_config.ffn_pruning_ratio:.1%}")
            
            pruning_meta = pruner.apply_pruning(model, plan, importance_rankings)
            
            # Fallback for architectures without structured components
            if pruning_meta.get('heads_pruned', 0) == 0 and pruning_meta.get('neurons_pruned', 0) == 0:
                logger.info("No structured components found, applying parameter-level fallback")
                fallback_ratio = float(max(
                    self.prune_config.head_pruning_ratio,
                    self.prune_config.ffn_pruning_ratio,
                ))
                fallback_meta = _apply_parameter_zero_pruning(model, plan, prune_ratio=fallback_ratio)
                
                if fallback_meta.get("parameters_zeroed", 0) > 0:
                    pruning_meta.update(fallback_meta)
                    logger.info(
                        f"Applied parameter-level fallback pruning: "
                        f"sparsified {fallback_meta['parameters_zeroed']} prune-zone parameters "
                        f"(ratio={fallback_meta.get('fallback_prune_ratio', 0.0):.1%}, "
                        f"effective={fallback_meta.get('effective_sparsity', 0.0):.2%})"
                    )
            
            # Save metadata
            if pruning_meta:
                prune_meta_dir = context.output_path / "compressed"
                prune_meta_dir.mkdir(parents=True, exist_ok=True)
                save_pruning_metadata(pruning_meta, prune_meta_dir)
            
            logger.info(f"Pruned {pruning_meta.get('heads_pruned', 0)} heads and "
                       f"{pruning_meta.get('neurons_pruned', 0)} neurons")
            
            # Log statistics
            log_section(logger, "After Pruning")
            after_pruning_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            log_metric(logger, "Size", f"{after_pruning_size_mb:.2f}", "MB")
            non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
            log_metric(logger, "Non-zero parameters", non_zero_params)
            sparsity = 1.0 - (non_zero_params / sum(p.numel() for p in model.parameters()))
            log_metric(logger, "Sparsity", f"{sparsity:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to apply pruning: {e}")
            raise
        
        # Update context
        context.metadata['pruning'] = pruning_meta
        
        return context
