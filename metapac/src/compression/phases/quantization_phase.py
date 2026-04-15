"""Quantization phase for compression pipeline.

Applies rank-aware quantization to model parameters based on importance scores.
"""
from __future__ import annotations

from typing import Dict, Any

import torch.nn as nn

from ..pipeline.phase_base import CompressionPhase, PhaseContext
from ..quantization import Quantizer, QuantizationConfig
from ...utils.logging_utils import get_logger, log_section, log_metric

logger = get_logger(__name__)


class QuantizationPhase(CompressionPhase):
    """Quantization phase: Apply rank-aware quantization to parameters.
    
    Features:
    - Rank-aware bit allocation (higher importance = more bits)
    - Per-channel or per-tensor quantization
    - Headroom utilization for dynamic precision
    - Variable-bit registry for export
    """
    
    def __init__(self, phase_config: Dict[str, Any]):
        super().__init__(phase_config)
        self.quant_config = QuantizationConfig(phase_config) if phase_config.get('enabled') else None
        self.quantizer = Quantizer(self.quant_config) if self.quant_config else None
    
    def get_phase_name(self) -> str:
        return "Quantization"
    
    def validate(self, context: PhaseContext) -> None:
        """Validate quantization preconditions."""
        if context.model is None:
            raise ValueError("Model not loaded in context")
        if context.plan is None:
            raise ValueError("Compression plan not available")
        if context.importance_rankings is None:
            raise ValueError("Importance rankings not available")
    
    def execute(self, context: PhaseContext) -> PhaseContext:
        """Execute quantization phase."""
        if not self.enabled or self.quantizer is None:
            logger.info("Quantization disabled, skipping")
            context.metadata['quantization'] = {'enabled': False}
            return context
        
        model = context.model
        plan = context.plan
        importance_rankings = context.importance_rankings
        target_bits_map = context.target_bits_map or {}
        
        try:
            logger.info(f"Applying rank-aware quantization to quantize zone...")
            logger.info(f"Mode: {self.quant_config.mode}")
            logger.info(f"Bits range: [{self.quant_config.bits_lower}, {self.quant_config.bits_upper}]")
            logger.info(f"Utilization target: {self.quant_config.util_target}")
            logger.info(f"Per-channel: {self.quant_config.per_channel}")
            
            quant_meta = self.quantizer.apply_quantization(
                model, 
                plan, 
                importance_rankings, 
                target_bits_map
            )
            
            logger.info(f"Quantized {len(quant_meta)} parameters in quantize zone")
            
            # Log statistics
            log_section(logger, "After Quantization")
            after_quant_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            log_metric(logger, "Size (FP32)", f"{after_quant_size_mb:.2f}", "MB")
            logger.info("Variable-bit size will be computed during export")
            
            # Update context with flag for downstream phases
            context.metadata['quantization'] = {
                'enabled': True,
                'num_quantized': len(quant_meta),
                'parameters': quant_meta
            }
            
        except Exception as e:
            logger.error(f"Failed to apply quantization: {e}")
            raise
        
        return context
