# metapac/src/compression/variable_bit_export.py
# Drop-in replacement providing both integrate_variable_bit_export and save_variable_bit_model.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from transformers import PreTrainedModel

from .variable_bit_layers import (
    ensure_registry,
    calculate_memory_savings,
)

__all__ = [
    "integrate_variable_bit_export",
    "save_variable_bit_model",
]


def _to_jsonable(obj):
    """Best-effort conversion to JSON-serializable types."""
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if isinstance(obj, dict):
        # Filter out _q_int tensors (only needed in-memory for packing, not for JSON export)
        return {k: _to_jsonable(v) for k, v in obj.items() if k != '_q_int'}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if torch is not None and isinstance(obj, torch.Tensor):
        try:
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
        except Exception:
            return None
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def save_variable_bit_model(
        model: PreTrainedModel,
        export_dir: str,
        *,
        stats_filename: str = "variable_bit_stats.json",
        meta_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save a HuggingFace-compatible checkpoint together with variable-bit statistics.

    Assumes that quantized layers carry `quant_meta` and that the registry was
    prepared earlier (strategy already calls registry builder). If not, the
    memory accounting still falls back to scanning named_modules().

    Returns the computed stats dictionary.
    """
    ensure_registry(model)
    os.makedirs(export_dir, exist_ok=True)

    # Move model to CPU before saving to avoid GPU/CUDA issues
    # This is especially important for newer GPUs (e.g., Blackwell) that may have
    # compatibility issues with PyTorch's serialization
    original_device = next(model.parameters()).device
    if str(original_device) != 'cpu':
        print(f"[export] Moving model from {original_device} to CPU for safe serialization...")
        model = model.cpu()
    
    # HF-compatible save (writes config.json + model weights)
    # Note: safe_serialization=False is faster and more reliable for large models
    model.save_pretrained(export_dir, safe_serialization=False)
    print(f"[export] Model weights saved to {export_dir}/pytorch_model.bin")

    # Compute and save stats sidecar
    stats = calculate_memory_savings(model)
    stats_path = os.path.join(export_dir, stats_filename)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"[export] Variable-bit stats saved -> {stats_path} | "
        f"fp32={stats['fp32_MiB']:.2f} MiB, quant={stats['quant_MiB']:.2f} MiB, "
        f"ratio={stats['compression_ratio']:.2f}x"
    )

    # Optionally save meta if caller provided a filename via integrate wrapper
    if meta_filename is not None:
        meta_path = os.path.join(export_dir, meta_filename)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"notice": "no meta provided"}, f, ensure_ascii=False, indent=2)

    return stats


def integrate_variable_bit_export(
        model: PreTrainedModel,
        combined_meta: Dict[str, Dict[str, Any]],
        export_dir: str,
        *,
        export_variable_bit: bool = True,
        use_cuda: bool = False,
        source_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Drop-in compatible wrapper expected by strategy.run_compression.

    Behavior:
      1) Ensures registry exists (strategy already registered layers).
      2) Saves the model in HF-compatible format under `export_dir`.
      3) Writes two sidecars:
           - variable_bit_stats.json  (aggregated memory/stats from quant_meta)
           - variable_bit_meta.json   (raw combined_meta converted to JSON)
      4) Returns the stats dict.

    Notes:
      - `export_variable_bit` and `use_cuda` are accepted for signature compatibility;
        here a pure-PyTorch/HF save is performed, runtime kernels are not touched.
      - `source_model_path` is accepted but not required; kept for compatibility.
    """
    del use_cuda, source_model_path  # not used here, kept for API compatibility
    ensure_registry(model)
    os.makedirs(export_dir, exist_ok=True)

    # Save raw combined_meta for auditability and reproducibility
    meta_jsonable = _to_jsonable(combined_meta)
    meta_path = os.path.join(export_dir, "variable_bit_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_jsonable, f, ensure_ascii=False, indent=2)
    print(f"[export] Variable-bit meta saved -> {meta_path}")

    # Proceed with standard HF save + stats
    stats = save_variable_bit_model(
        model,
        export_dir,
        stats_filename="variable_bit_stats.json",
        meta_filename=None,
    )

    return stats
