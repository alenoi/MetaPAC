"""Feature extraction package for MetaPAC.

This package provides tools for:
- Building meta-datasets from model activations
- Reducing high-dimensional features
- Loading and preprocessing hook data

Main components:
- BuildConfig: Configuration class for dataset building
- build_meta_dataset: Main pipeline for dataset creation
- apply_reducer: Feature dimension reduction utilities
"""

from .builder import BuildConfig, build_meta_dataset, load_config
from .reducers import apply_reducer

__all__ = [
    'BuildConfig',
    'build_meta_dataset',
    'load_config',
    'apply_reducer'
]
