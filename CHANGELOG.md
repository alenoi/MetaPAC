# Changelog

All notable changes to MetaPAC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-14

### Added
- Initial public release
- Core compression framework with pruning and quantization
- Meta-learning based compression prediction
- Variable-bit quantization support
- Structured and unstructured pruning methods
- Post-compression fine-tuning with knowledge distillation
- CLI interface for all pipeline modes
- Python API for programmatic access
- Example configurations for common use cases
- Comprehensive documentation

### Features
- **Compression Methods:**
  - Magnitude-based pruning
  - Gradient-based pruning
  - Meta-predicted adaptive pruning
  - Variable-bit quantization (2, 4, 6, 8 bits)
  - Uniform INT8 quantization
  
- **Pipeline Modes:**
  - Full automatic pipeline
  - Baseline fine-tuning
  - Feature extraction
  - Meta-predictor training
  - Model compression
  
- **Fine-tuning:**
  - Post-compression recovery
  - Knowledge distillation from original model
  - Configurable training parameters
  
- **Model Support:**
  - DistilBERT
  - BERT (experimental)
  - Extensible adapter system for other architectures

[0.1.0]: https://github.com/alenoi/MetaPAC/releases/tag/v0.1.0
