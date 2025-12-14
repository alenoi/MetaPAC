"""
Evaluation utilities for compressed models.

This script provides utilities to evaluate compressed models:
1. Measure memory usage reduction
2. Evaluate accuracy on validation/test sets
3. Compare baseline vs compressed model performance
4. Generate compression reports
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
except ImportError:
    print("Warning: transformers and datasets not available")

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dict with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count non-zero parameters (for pruned models)
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'nonzero_params': nonzero_params,
        'zero_params': total_params - nonzero_params,
        'sparsity': 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    }


def measure_memory_usage(model: nn.Module) -> Dict[str, float]:
    """Measure model memory usage.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dict with memory statistics in MB.
    """
    # Parameter memory
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()

    # Non-zero parameter memory (effective for pruned models)
    nonzero_memory = 0
    for param in model.parameters():
        nonzero_memory += (param != 0).sum().item() * param.element_size()

    # Buffer memory
    buffer_memory = 0
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()

    total_memory = param_memory + buffer_memory
    effective_memory = nonzero_memory + buffer_memory

    return {
        'param_memory_mb': param_memory / (1024 ** 2),
        'buffer_memory_mb': buffer_memory / (1024 ** 2),
        'total_memory_mb': total_memory / (1024 ** 2),
        'nonzero_memory_mb': nonzero_memory / (1024 ** 2),
        'effective_memory_mb': effective_memory / (1024 ** 2),
        'memory_reduction_ratio': effective_memory / total_memory if total_memory > 0 else 1.0
    }


def evaluate_model(
        model: nn.Module,
        tokenizer,
        dataset_name: str = 'glue',
        dataset_config: str = 'sst2',
        split: str = 'validation',
        max_samples: Optional[int] = None,
        device: str = 'cpu'
) -> Dict[str, float]:
    """Evaluate model accuracy on a dataset.
    
    Args:
        model: PyTorch model.
        tokenizer: Tokenizer for encoding.
        dataset_name: Dataset name.
        dataset_config: Dataset configuration.
        split: Dataset split to evaluate.
        max_samples: Maximum number of samples to evaluate (None = all).
        device: Device to run on.
    
    Returns:
        Dict with accuracy metrics.
    """
    logger.info(f"Evaluating on {dataset_name}/{dataset_config} ({split})")

    model.to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"Evaluating on {len(dataset)} samples")

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            # Tokenize
            inputs = tokenizer(
                example['sentence'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Prediction
            pred = torch.argmax(logits, dim=-1).item()
            label = example['label']

            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def compare_models(
        baseline_path: str,
        compressed_path: str,
        dataset_name: str = 'glue',
        dataset_config: str = 'sst2',
        device: str = 'cpu'
) -> Dict[str, Any]:
    """Compare baseline and compressed models.
    
    Args:
        baseline_path: Path to baseline model checkpoint.
        compressed_path: Path to compressed model checkpoint.
        dataset_name: Dataset name for evaluation.
        dataset_config: Dataset configuration.
        device: Device to run on.
    
    Returns:
        Dict with comparison results.
    """
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)

    # Load baseline model
    logger.info(f"Loading baseline model from: {baseline_path}")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_path)

    # Load compressed model
    logger.info(f"Loading compressed model from: {compressed_path}")
    compressed_model = AutoModelForSequenceClassification.from_pretrained(baseline_path)

    # Load compressed state if available
    compressed_state_path = Path(compressed_path) / "model_state.pt"
    if compressed_state_path.exists():
        logger.info(f"Loading compressed state from: {compressed_state_path}")
        state_dict = torch.load(compressed_state_path, map_location='cpu')
        compressed_model.load_state_dict(state_dict)

    # Count parameters
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER COUNTS")
    logger.info("=" * 80)

    baseline_params = count_parameters(baseline_model)
    compressed_params = count_parameters(compressed_model)

    logger.info(f"\nBaseline model:")
    logger.info(f"  Total params: {baseline_params['total_params']:,}")
    logger.info(f"  Non-zero params: {baseline_params['nonzero_params']:,}")
    logger.info(f"  Sparsity: {baseline_params['sparsity']:.2%}")

    logger.info(f"\nCompressed model:")
    logger.info(f"  Total params: {compressed_params['total_params']:,}")
    logger.info(f"  Non-zero params: {compressed_params['nonzero_params']:,}")
    logger.info(f"  Sparsity: {compressed_params['sparsity']:.2%}")

    param_reduction = 1.0 - (compressed_params['nonzero_params'] / baseline_params['nonzero_params'])
    logger.info(f"\nParameter reduction: {param_reduction:.2%}")

    # Measure memory
    logger.info("\n" + "=" * 80)
    logger.info("MEMORY USAGE")
    logger.info("=" * 80)

    baseline_memory = measure_memory_usage(baseline_model)
    compressed_memory = measure_memory_usage(compressed_model)

    logger.info(f"\nBaseline model:")
    logger.info(f"  Total memory: {baseline_memory['total_memory_mb']:.2f} MB")
    logger.info(f"  Effective memory: {baseline_memory['effective_memory_mb']:.2f} MB")

    logger.info(f"\nCompressed model:")
    logger.info(f"  Total memory: {compressed_memory['total_memory_mb']:.2f} MB")
    logger.info(f"  Effective memory: {compressed_memory['effective_memory_mb']:.2f} MB")

    memory_reduction = 1.0 - (compressed_memory['effective_memory_mb'] / baseline_memory['effective_memory_mb'])
    logger.info(f"\nMemory reduction: {memory_reduction:.2%}")

    # Evaluate accuracy
    logger.info("\n" + "=" * 80)
    logger.info("ACCURACY EVALUATION")
    logger.info("=" * 80)

    logger.info("\nBaseline model:")
    baseline_metrics = evaluate_model(
        baseline_model, tokenizer, dataset_name, dataset_config, device=device
    )

    logger.info("\nCompressed model:")
    compressed_metrics = evaluate_model(
        compressed_model, tokenizer, dataset_name, dataset_config, device=device
    )

    accuracy_drop = baseline_metrics['accuracy'] - compressed_metrics['accuracy']
    accuracy_retention = compressed_metrics['accuracy'] / baseline_metrics['accuracy']

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nParameter reduction: {param_reduction:.2%}")
    logger.info(f"Memory reduction: {memory_reduction:.2%}")
    logger.info(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"Compressed accuracy: {compressed_metrics['accuracy']:.4f}")
    logger.info(f"Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop / baseline_metrics['accuracy']:.2%})")
    logger.info(f"Accuracy retention: {accuracy_retention:.2%}")
    logger.info("=" * 80)

    return {
        'baseline': {
            'params': baseline_params,
            'memory': baseline_memory,
            'metrics': baseline_metrics
        },
        'compressed': {
            'params': compressed_params,
            'memory': compressed_memory,
            'metrics': compressed_metrics
        },
        'comparison': {
            'param_reduction': param_reduction,
            'memory_reduction': memory_reduction,
            'accuracy_drop': accuracy_drop,
            'accuracy_retention': accuracy_retention
        }
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate compressed model")
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline model checkpoint'
    )
    parser.add_argument(
        '--compressed',
        type=str,
        required=True,
        help='Path to compressed model checkpoint'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='glue',
        help='Dataset name'
    )
    parser.add_argument(
        '--dataset-config',
        type=str,
        default='sst2',
        help='Dataset configuration'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results JSON'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run comparison
    results = compare_models(
        baseline_path=args.baseline,
        compressed_path=args.compressed,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        device=args.device
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nSaved results to: {output_path}")


if __name__ == '__main__':
    main()
