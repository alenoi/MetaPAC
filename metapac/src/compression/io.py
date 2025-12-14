"""
Input/Output utilities for compressed model checkpoints.

Handles loading and saving of compressed models with quantization metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .quantization import load_quantization_metadata


def load_compressed_model(checkpoint_dir: Path, model: nn.Module,
                          device: Optional[torch.device] = None) -> nn.Module:
    """
    Load compressed model from checkpoint directory.  Searches for standard
    filenames (model_state.pt, pytorch_model.bin, model_variable_bit.pt).
    """
    if device is None:
        device = torch.device('cpu')

    # Look for known state file names
    candidate_files = ["model_state.pt", "pytorch_model.bin", "model_variable_bit.pt"]
    model_path = None
    for fname in candidate_files:
        path = checkpoint_dir / fname
        if path.exists():
            model_path = path
            break
    if model_path is None:
        raise FileNotFoundError(f"No model state file found in {checkpoint_dir}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Metadata and summary loading (unchanged)
    quant_meta_path = checkpoint_dir / 'quant_meta.json'
    if quant_meta_path.exists():
        quant_meta = load_quantization_metadata(checkpoint_dir)
        model._quant_meta = quant_meta

    summary_path = checkpoint_dir / 'compression_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        model._compression_summary = summary

    return model


def save_compressed_checkpoint(model: nn.Module, output_dir: Path,
                               quant_meta: Dict[str, Any],
                               summary: Dict[str, Any]):
    """
    Save compressed model checkpoint with metadata.
    
    Args:
        model: Compressed model
        output_dir: Output directory
        quant_meta: Quantization metadata
        summary: Compression summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    model_path = output_dir / 'model_state.pt'
    torch.save(model.state_dict(), model_path)
    print(f"[io] Saved model state to {model_path}")

    # Save quantization metadata
    if quant_meta:
        quant_meta_path = output_dir / 'quant_meta.json'
        with open(quant_meta_path, 'w') as f:
            json.dump(quant_meta, f, indent=2)
        print(f"[io] Saved quantization metadata to {quant_meta_path}")

    # Save compression summary
    summary_path = output_dir / 'compression_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[io] Saved compression summary to {summary_path}")

    print(f"[io] Compressed checkpoint saved to {output_dir}")


def get_compression_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get compression statistics for a model.
    
    Args:
        model: Model (compressed or uncompressed)
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'total_size_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
    }

    # Add quantization stats if available
    if hasattr(model, '_quant_meta'):
        quant_meta = model._quant_meta
        stats['quantized_parameters'] = len(quant_meta)

        # Compute bit distribution
        bit_counts = {}
        for param_meta in quant_meta.values():
            bits = param_meta['bits_final']
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
        stats['bit_distribution'] = bit_counts

        # Average utilization
        avg_util = sum(p['util_final'] for p in quant_meta.values()) / len(quant_meta)
        stats['avg_utilization'] = avg_util

    # Add compression summary if available
    if hasattr(model, '_compression_summary'):
        stats['compression_summary'] = model._compression_summary

    return stats
