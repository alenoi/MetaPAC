# metapac/src/compression/__init__.py
"""
Compression subpackage public API.

Csak a ténylegesen elérhető jelképeket exportáljuk; a régi, megszűnt API-kat
nem importáljuk ide, hogy elkerüljük az ImportError-okat.
"""

from .variable_bit_layers import (
    QuantizedLinear,
    QuantizedEmbedding,
    calculate_memory_savings,  # util a memória-megtakarítás becsléséhez
    replace_linear_with_quantized,  # deprecált shim – ha valaki még importálja
)

__all__ = [
    "QuantizedLinear",
    "QuantizedEmbedding",
    "calculate_memory_savings",
    "replace_linear_with_quantized",
]
