"""Abstract base class for compression pipeline phases.

Each phase encapsulates a distinct step in the compression pipeline with:
- Clear input/output contracts
- Independent execution logic
- Structured metadata collection
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

import torch.nn as nn

from ...utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PhaseContext:
    """Shared context passed between pipeline phases.
    
    Attributes:
        model: The PyTorch model being compressed
        config: Full compression configuration
        output_path: Base output directory for artifacts
        metadata: Accumulated metadata from all phases
        ranked_df: Parameter rankings from preparation phase (if available)
        plan: Action plan mapping parameter names to actions (keep/quantize/prune)
        importance_rankings: Normalized importance scores (0-1) per parameter
    """
    model: nn.Module
    config: Dict[str, Any]
    output_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    ranked_df: Optional[Any] = None
    plan: Optional[Dict[str, str]] = None
    importance_rankings: Optional[Dict[str, float]] = None
    target_bits_map: Optional[Dict[str, int]] = None


class CompressionPhase(ABC):
    """Abstract base class for compression pipeline phases.
    
    Each phase implements:
    - execute(): Main phase logic
    - validate(): Pre-execution validation
    - get_phase_name(): Human-readable name for logging
    
    Phases should be stateless and idempotent where possible.
    """
    
    def __init__(self, phase_config: Dict[str, Any]):
        """Initialize phase with configuration.
        
        Args:
            phase_config: Phase-specific configuration dictionary
        """
        self.config = phase_config
        self.enabled = phase_config.get('enabled', True)
    
    @abstractmethod
    def execute(self, context: PhaseContext) -> PhaseContext:
        """Execute the phase logic.
        
        Args:
            context: Shared pipeline context
            
        Returns:
            Updated context with phase results
            
        Raises:
            Exception: If phase execution fails
        """
        pass
    
    def validate(self, context: PhaseContext) -> None:
        """Validate preconditions before execution.
        
        Args:
            context: Shared pipeline context
            
        Raises:
            ValueError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_phase_name(self) -> str:
        """Get human-readable phase name for logging.
        
        Returns:
            Phase name string
        """
        pass
    
    def run(self, context: PhaseContext) -> PhaseContext:
        """Run phase with validation and error handling.
        
        Args:
            context: Shared pipeline context
            
        Returns:
            Updated context
            
        Raises:
            Exception: If phase fails
        """
        if not self.enabled:
            logger.info(f"Phase '{self.get_phase_name()}' is disabled, skipping")
            return context
        
        logger.info(f"Running phase: {self.get_phase_name()}")
        
        # Validation
        try:
            self.validate(context)
        except Exception as e:
            logger.error(f"Phase validation failed: {e}")
            raise
        
        # Execution
        try:
            context = self.execute(context)
            logger.info(f"Phase '{self.get_phase_name()}' completed successfully")
            return context
        except Exception as e:
            logger.error(f"Phase '{self.get_phase_name()}' failed: {e}")
            raise
