"""Model implementations for MetaPAC.

This package contains:
- TorchMetaPredictor: Neural network for meta-learning
- TorchModelWrapper: Scikit-learn compatible PyTorch model wrapper
- ModelConfig: Configuration for building models
- build_model: Factory function for building models
"""

from .meta_predictor import TorchMetaPredictor
from .model import ModelConfig, build_model
from .wrappers import TorchModelWrapper

__all__ = ['TorchMetaPredictor', 'TorchModelWrapper', 'ModelConfig', 'build_model']
