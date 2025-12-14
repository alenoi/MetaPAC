# MetaPAC: Meta-learning based Predictive Adaptive Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

MetaPAC is a meta-learning based framework for predictive adaptive compression of transformer models. It combines pruning and quantization techniques with meta-learned predictions to achieve optimal compression while maintaining model performance.

## Features

- **Meta-learning based compression**: Predict optimal compression configurations using learned meta-models
- **Hybrid compression**: Combines structured/unstructured pruning with variable-bit quantization
- **Flexible pipeline**: Modular architecture supporting various compression strategies
- **Fine-tuning integration**: Post-compression fine-tuning with knowledge distillation
- **CLI & Python API**: Easy-to-use command-line interface and programmatic access

## Installation

```bash
pip install metapac
```

Or install from source:

```bash
git clone https://github.com/alenoi/MetaPAC.git
cd MetaPAC
pip install -e .
```

## Quick Start

### Using the CLI

**Full automatic pipeline:**
```bash
python -m metapac --mode auto
```

**Compress a model with configuration:**
```bash
python -m metapac --mode compress --config examples/configs/compress_distilbert_sst2.yaml
```

**Extract features for meta-learning:**
```bash
python -m metapac --mode feature_extract --config examples/configs/feature_extraction.yaml
```

**Train meta-predictor:**
```bash
python -m metapac --mode train_meta --config examples/configs/meta_distilbert_sst2.yaml
```

### Using the Python API

```python
from metapac import build_meta_dataset, TorchMetaPredictor, TorchModelWrapper

# Build meta-dataset from a model
config = {
    "model_name": "distilbert-base-uncased",
    "dataset": "glue",
    "dataset_config": "sst2"
}
meta_dataset = build_meta_dataset(config)

# Train meta-predictor
predictor = TorchMetaPredictor()
predictor.train(meta_dataset)

# Use for compression prediction
wrapper = TorchModelWrapper(model)
predictions = predictor.predict(wrapper)
```

## CLI Reference

### Modes

- **`auto`**: Run full pipeline (baseline fine-tuning → feature extraction → meta-training → compression)
- **`auto:feature_extract`**: Run from feature extraction onwards (skip baseline fine-tuning)
- **`baseline_finetune`**: Fine-tune baseline model only
- **`feature_extract`**: Extract features for meta-learning
- **`train_meta`**: Train meta-predictor
- **`compress`**: Compress model with optional fine-tuning

### Command-line Arguments

- `--config PATH`: Path to YAML configuration file
- `--mode MODE`: Pipeline mode to run

### Configuration Files

See `examples/configs/` for example configuration files:

- **`compress_distilbert_sst2.yaml`**: Basic compression configuration
- **`compress_with_finetuning.yaml`**: Compression with post-compression fine-tuning
- **`feature_extraction.yaml`**: Feature extraction configuration
- **`meta_distilbert_sst2.yaml`**: Meta-predictor training configuration

### Scenario Presets

Pre-configured compression scenarios in `examples/configs/scenarios/` (used in ablation studies):

- **`prune_magnitude_logical_30.yaml`**: Pruning-only baseline (30% magnitude-based)
- **`quant_vb_headroom_on.yaml`**: Quantization-only baseline (variable-bit 2-8 bits)
- **`compress_finetune_no_kd.yaml`**: Combined pruning + quantization (no fine-tuning)
- **`compress_finetune_kd.yaml`**: Full pipeline with knowledge distillation (recommended)

## Configuration

### Basic Configuration Structure

```yaml
mode: compress

model:
  name: "distilbert-base-uncased"
  task: "sequence-classification"
  
dataset:
  name: "glue"
  config: "sst2"

compression:
  pruning:
    enabled: true
    ratio: 0.3
    method: "magnitude"
    
  quantization:
    enabled: true
    method: "variable_bit"
    bits: [2, 4, 6, 8]
    
fine_tuning:
  enabled: true
  epochs: 3
  learning_rate: 2e-5
  use_kd: true  # Knowledge distillation
```

### Advanced Options

See example configurations for advanced options including:
- Custom pruning strategies (magnitude, gradient-based, meta-predicted)
- Variable-bit quantization with headroom optimization
- Fine-tuning with knowledge distillation
- Custom meta-predictor architectures

## Pipeline Stages

### 1. Feature Extraction
Extract layer-level and parameter-level features from the model for meta-learning.

### 2. Meta-Predictor Training
Train a meta-model to predict optimal compression configurations based on extracted features.

### 3. Compression
Apply pruning and/or quantization based on meta-predictions or predefined strategies.

### 4. Fine-tuning (Optional)
Fine-tune compressed model with optional knowledge distillation from the original model.

## Output Structure

```
targets/<model_name>/models/experiments/<experiment_name>/
├── pruned_before_quant/      # Model after pruning
├── quantized_before_ft/       # Model after quantization (fake-quant)
├── finetuned/                 # Model after fine-tuning
├── compressed/                # Final compressed model
│   ├── pytorch_model.bin      # Fake-quant FP32 weights
│   ├── model_packed.bin       # Packed variable-bit weights
│   └── compression_config.json
└── logs/                      # Training and compression logs
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.44
- See `requirements.txt` for full dependency list

## Citation

If you use MetaPAC in your research, please cite:

```bibtex
@software{metapac2025,
  title = {MetaPAC: Meta-learning based Predictive Adaptive Compression},
  author = {Panyi, Tamás},
  year = {2025},
  version = {0.1.0},
  institution = {Óbudai Egyetem},
  url = {https://github.com/alenoi/metapac}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This research was conducted at Óbudai Egyetem (Óbuda University).

## Support

For questions, issues, or feature requests, please open an issue on [GitHub](https://github.com/alenoi/MetaPAC/issues).
