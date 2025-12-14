"""
Architecture Adapter Base Class

This module defines the interface for architecture-specific pruning loaders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn


@dataclass
class PruningSpec:
    """Specification for a pruned module.
    
    Attributes:
        target_type: Type of pruning target ("attention", "ffn", "layer", "embedding")
        layer_path: Full module path (e.g., "bert.encoder.layer.4.attention")
        layer_index: Layer index in the model
        original_dim: Original dimension before pruning
        pruned_dim: Dimension after pruning
        kept_indices: Indices of kept elements (for structured pruning)
        pruning_mask: Mask tensor (for unstructured pruning)
        metadata: Additional metadata
    """
    target_type: str
    layer_path: str
    layer_index: int
    original_dim: Optional[int] = None
    pruned_dim: Optional[int] = None
    kept_indices: Optional[List[int]] = None
    pruning_mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class ArchitectureAdapter(ABC):
    """Base class for architecture-specific pruning operations.
    
    Each adapter handles:
    1. Creating base models
    2. Identifying prunable modules
    3. Reconstructing pruned modules
    4. Patching forward passes
    5. Validating compatibility
    
    To add support for a new architecture:
    1. Subclass ArchitectureAdapter
    2. Implement all abstract methods
    3. Register with @registry.register_adapter
    """

    # Architecture metadata (override in subclasses)
    architecture_name: str = "base"
    supported_models: List[str] = []

    @abstractmethod
    def create_base_model(
            self,
            model_name: str,
            config: Optional[dict] = None,
            **kwargs
    ) -> nn.Module:
        """Create an unpruned base model.
        
        Args:
            model_name: Name or path of the model
            config: Optional config overrides
            **kwargs: Additional model creation arguments
            
        Returns:
            Base model instance
        """

    @abstractmethod
    def get_prunable_modules(
            self,
            model: nn.Module
    ) -> Dict[str, nn.Module]:
        """Get all modules that can be pruned.
        
        Args:
            model: The model to inspect
            
        Returns:
            Dict mapping module_path -> module
            Example: {"bert.encoder.layer.0.attention": <module>}
        """

    @abstractmethod
    def reconstruct_module(
            self,
            module: nn.Module,
            pruning_spec: PruningSpec,
            state_dict: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """Reconstruct a pruned module with correct dimensions.
        
        Args:
            module: Original unpruned module
            pruning_spec: Specification of how it was pruned
            state_dict: State dict containing pruned weights
            
        Returns:
            Reconstructed module with pruned dimensions
        """

    @abstractmethod
    def create_forward_patch(
            self,
            module: nn.Module,
            pruning_spec: PruningSpec
    ) -> callable:
        """Create a patched forward method for a pruned module.
        
        Args:
            module: The reconstructed module
            pruning_spec: Specification of the pruning
            
        Returns:
            Patched forward function
        """

    @abstractmethod
    def validate_compatibility(
            self,
            metadata: dict
    ) -> bool:
        """Check if this adapter can handle the pruned model.
        
        Args:
            metadata: Pruning metadata from pruning_meta.json
            
        Returns:
            True if compatible, False otherwise
        """

    # Optional methods (can be overridden)

    def parse_metadata(
            self,
            metadata: dict
    ) -> List[PruningSpec]:
        """Parse pruning metadata into PruningSpec objects.
        
        Default implementation handles common formats.
        Override for architecture-specific parsing.
        
        Args:
            metadata: Raw metadata dict
            
        Returns:
            List of PruningSpec objects
        """
        specs = []

        # Handle v1.0 format (pruned_heads, pruned_neurons)
        if 'pruned_heads' in metadata:
            for layer_path, kept_indices in metadata['pruned_heads'].items():
                specs.append(PruningSpec(
                    target_type="attention",
                    layer_path=layer_path,
                    layer_index=self._extract_layer_index(layer_path),
                    kept_indices=kept_indices,
                    metadata={"original_format": "v1.0"}
                ))

        if 'pruned_neurons' in metadata:
            for layer_path, kept_indices in metadata['pruned_neurons'].items():
                specs.append(PruningSpec(
                    target_type="ffn",
                    layer_path=layer_path,
                    layer_index=self._extract_layer_index(layer_path),
                    kept_indices=kept_indices,
                    metadata={"original_format": "v1.0"}
                ))

        # Handle v2.0 format (pruning_specs)
        if 'pruning_specs' in metadata:
            for spec_dict in metadata['pruning_specs']:
                specs.append(PruningSpec(**spec_dict))

        return specs

    def _extract_layer_index(self, layer_path: str) -> int:
        """Extract layer index from path like 'model.layer.4.attention'."""
        import re
        match = re.search(r'\.(\d+)\.', layer_path)
        if match:
            return int(match.group(1))
        return -1

    def get_module_by_path(
            self,
            model: nn.Module,
            path: str
    ) -> Optional[nn.Module]:
        """Get a module by its full path.
        
        Args:
            model: The model
            path: Dot-separated path (e.g., "bert.encoder.layer.0.attention")
            
        Returns:
            The module, or None if not found
        """
        parts = path.split('.')
        current = model

        for part in parts:
            if not hasattr(current, part):
                return None
            current = getattr(current, part)

        return current

    def set_module_by_path(
            self,
            model: nn.Module,
            path: str,
            new_module: nn.Module
    ) -> None:
        """Set a module by its full path.
        
        Args:
            model: The model
            path: Dot-separated path
            new_module: Module to set
        """
        parts = path.split('.')
        parent_path = '.'.join(parts[:-1])
        attr_name = parts[-1]

        parent = self.get_module_by_path(model, parent_path)
        if parent is not None:
            setattr(parent, attr_name, new_module)
