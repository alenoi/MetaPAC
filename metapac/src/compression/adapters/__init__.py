"""
Architecture adapters for universal pruning loader.

This package provides architecture-specific adapters for loading
physically pruned models.

Example:
    >>> from metapac.src.compression.adapters import get_adapter
    >>> 
    >>> # Get DistilBERT adapter
    >>> adapter_class = get_adapter("distilbert")
    >>> adapter = adapter_class()
    >>> 
    >>> # Use with universal loader
    >>> model = load_physically_pruned_model(checkpoint_path, adapter=adapter)

Adding a new architecture:
    >>> from metapac.src.compression.adapters import ArchitectureAdapter, register_adapter
    >>> 
    >>> @register_adapter
    >>> class MyCustomAdapter(ArchitectureAdapter):
    >>>     architecture_name = "my_custom"
    >>>     supported_models = ["my-org/my-model"]
    >>>     
    >>>     def create_base_model(self, model_name, config=None, **kwargs):
    >>>         from my_package import MyModel
    >>>         return MyModel.from_pretrained(model_name)
    >>>     
    >>>     # ... implement other abstract methods
"""

from .base import ArchitectureAdapter, PruningSpec
from .registry import (
    register_adapter,
    get_adapter,
    get_adapter_for_model,
    list_architectures,
    auto_detect_architecture
)

# Import and register built-in adapters
# (These will be created in future phases)
# from .distilbert import DistilBERTAdapter  # Phase 1
# from .bert import BERTAdapter              # Phase 2
# from .gpt2 import GPT2Adapter              # Phase 3
# from .t5 import T5Adapter                  # Phase 4

__all__ = [
    # Base classes
    'ArchitectureAdapter',
    'PruningSpec',

    # Registry functions
    'register_adapter',
    'get_adapter',
    'get_adapter_for_model',
    'list_architectures',
    'auto_detect_architecture',

    # Built-in adapters (will be added in phases)
    # 'DistilBERTAdapter',
    # 'BERTAdapter',
    # 'GPT2Adapter',
    # 'T5Adapter',
]
