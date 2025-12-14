# MetaPAC Example Configurations

This directory contains example configuration files for various MetaPAC use cases.

## Basic Configurations

### compress_distilbert_sst2.yaml
Basic compression configuration for DistilBERT on SST-2 dataset.

**Usage:**
```bash
python -m metapac --mode compress --config examples/configs/compress_distilbert_sst2.yaml
```

### compress_with_finetuning.yaml
Compression with post-compression fine-tuning and knowledge distillation.

**Usage:**
```bash
python -m metapac --mode compress --config examples/configs/compress_with_finetuning.yaml
```

### feature_extraction.yaml
Extract features from a model for meta-learning.

**Usage:**
```bash
python -m metapac --mode feature_extract --config examples/configs/feature_extraction.yaml
```

### meta_distilbert_sst2.yaml
Train a meta-predictor on DistilBERT features.

**Usage:**
```bash
python -m metapac --mode train_meta --config examples/configs/meta_distilbert_sst2.yaml
```

## Scenario Presets

The `scenarios/` directory contains pre-configured compression scenarios used in ablation studies and research.

### prune_magnitude_logical_30.yaml
- **Pruning**: 30% magnitude-based logical pruning
- **Quantization**: Disabled
- **Fine-tuning**: Disabled
- **Use case**: Pruning-only baseline for ablation studies

### quant_vb_headroom_on.yaml
- **Pruning**: Disabled
- **Quantization**: Variable-bit quantization (2-8 bits) with headroom optimization
- **Fine-tuning**: Disabled
- **Use case**: Quantization-only baseline for ablation studies

### compress_finetune_no_kd.yaml
- **Pruning**: 30% magnitude-based logical pruning
- **Quantization**: Variable-bit quantization
- **Fine-tuning**: Enabled (without knowledge distillation)
- **Use case**: Combined compression without knowledge distillation

### compress_finetune_kd.yaml
- **Pruning**: 30% magnitude-based logical pruning
- **Quantization**: Variable-bit quantization
- **Fine-tuning**: Enabled with knowledge distillation
- **Use case**: Full pipeline with all optimizations (recommended)

## Configuration Structure

All configuration files follow this basic structure:

```yaml
mode: compress  # Pipeline mode

model:
  name: "distilbert-base-uncased"
  task: "sequence-classification"
  
dataset:
  name: "glue"
  config: "sst2"

compression:
  pruning:
    enabled: true/false
    ratio: 0.0-1.0
    method: "magnitude" | "gradient" | "meta"
    
  quantization:
    enabled: true/false
    method: "uniform" | "variable_bit"
    bits: [2, 4, 6, 8]  # For variable-bit
    
fine_tuning:
  enabled: true/false
  epochs: 3
  learning_rate: 2e-5
  use_kd: true/false  # Knowledge distillation
```

## Customization

You can customize these configurations by:

1. **Copying an example**: Start with a similar use case
2. **Modifying parameters**: Adjust pruning ratios, quantization bits, etc.
3. **Adding new options**: See full configuration schema in documentation

## More Information

For detailed configuration options and advanced use cases, see the main [README.md](../../README.md).
