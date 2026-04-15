# metapac/src/compression/__init__.py
"""
Compression subpackage public API.

Only export symbols that are actually available; older, removed APIs are not
imported here to avoid ImportError exceptions.
"""

from .variable_bit_layers import (
    QuantizedLinear,
    QuantizedEmbedding,
    calculate_memory_savings,  # Utility for estimating memory savings.
    replace_linear_with_quantized,  # Deprecated shim for legacy imports.
)

__all__ = [
    "QuantizedLinear",
    "QuantizedEmbedding",
    "calculate_memory_savings",
    "replace_linear_with_quantized",
]
