"""
Adapter Registry

Manages registration and retrieval of architecture adapters.
"""
from typing import Dict, Type, Optional

from metapac.src.model_profiles import resolve_architecture_name, resolve_model_profile_from_name

from .base import ArchitectureAdapter


class AdapterRegistry:
    """Registry for architecture adapters."""

    def __init__(self):
        self._adapters: Dict[str, Type[ArchitectureAdapter]] = {}

    def register(
            self,
            adapter_class: Type[ArchitectureAdapter]
    ) -> Type[ArchitectureAdapter]:
        """Register an adapter class.
        
        Args:
            adapter_class: Adapter class to register
            
        Returns:
            The adapter class (for use as decorator)
        """
        name = adapter_class.architecture_name
        if name in self._adapters:
            raise ValueError(f"Adapter for '{name}' already registered")

        self._adapters[name] = adapter_class
        return adapter_class

    def get(
            self,
            architecture: str
    ) -> Type[ArchitectureAdapter]:
        """Get an adapter class by architecture name.
        
        Args:
            architecture: Architecture name (e.g., "distilbert", "bert")
            
        Returns:
            Adapter class
            
        Raises:
            KeyError: If architecture not found
        """
        if architecture not in self._adapters:
            available = ', '.join(self._adapters.keys())
            raise KeyError(
                f"No adapter found for '{architecture}'. "
                f"Available: {available}"
            )

        return self._adapters[architecture]

    def get_for_model(
            self,
            model_name: str
    ) -> Optional[Type[ArchitectureAdapter]]:
        """Find an adapter for a model name.
        
        Args:
            model_name: Model name (e.g., "bert-base-uncased")
            
        Returns:
            Adapter class, or None if no match found
        """
        profile = resolve_model_profile_from_name(model_name)
        if profile.architecture in self._adapters:
            return self._adapters[profile.architecture]
        for adapter_class in self._adapters.values():
            if any(model_name.startswith(m) for m in adapter_class.supported_models):
                return adapter_class
        return None

    def list_architectures(self) -> list[str]:
        """Get list of supported architectures."""
        return list(self._adapters.keys())

    def list_adapters(self) -> Dict[str, Type[ArchitectureAdapter]]:
        """Get all registered adapters."""
        return self._adapters.copy()


# Global registry instance
_registry = AdapterRegistry()


def register_adapter(
        adapter_class: Type[ArchitectureAdapter]
) -> Type[ArchitectureAdapter]:
    """Decorator to register an adapter.
    
    Usage:
        @register_adapter
        class MyAdapter(ArchitectureAdapter):
            ...
    """
    return _registry.register(adapter_class)


def get_adapter(architecture: str) -> Type[ArchitectureAdapter]:
    """Get an adapter by architecture name."""
    return _registry.get(architecture)


def get_adapter_for_model(model_name: str) -> Optional[Type[ArchitectureAdapter]]:
    """Find an adapter for a model name."""
    return _registry.get_for_model(model_name)


def list_architectures() -> list[str]:
    """Get list of supported architectures."""
    return _registry.list_architectures()


def auto_detect_architecture(metadata: dict) -> Optional[str]:
    """Auto-detect architecture from metadata.
    
    Args:
        metadata: Pruning metadata
        
    Returns:
        Architecture name, or None if cannot detect
    """
    # Check v2.0 format
    if 'architecture' in metadata:
        return metadata['architecture']

    # Check base_model in v2.0 format
    if 'base_model' in metadata:
        model_name = metadata['base_model']
        resolved = resolve_architecture_name(model_name)
        if resolved != 'generic':
            return resolved
        adapter = get_adapter_for_model(model_name)
        if adapter:
            return adapter.architecture_name

    # Try to infer from pruned layer names (v1.0 format)
    all_layers = []
    if 'pruned_heads' in metadata:
        all_layers.extend(metadata['pruned_heads'].keys())
    if 'pruned_neurons' in metadata:
        all_layers.extend(metadata['pruned_neurons'].keys())

    if any('distilbert' in layer.lower() for layer in all_layers):
        return 'distilbert'
    elif any('bert' in layer.lower() for layer in all_layers):
        return 'bert'
    elif any('gpt2' in layer.lower() or 'transformer.h' in layer for layer in all_layers):
        return 'gpt2'
    elif any('t5' in layer.lower() for layer in all_layers):
        return 't5'

    return None
