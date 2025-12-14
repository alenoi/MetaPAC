# metapac/src/compression/load_quantized_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from .variable_bit_layers import QuantizedLinear, QuantizedEmbedding


def _device_of(runtime: str) -> str:
    if runtime == "cpu":
        return "cpu"
    if runtime in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if runtime == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (runtime is None and torch.cuda.is_available()) else (runtime or "cpu")


def _read_state_dict(model_dir: Path) -> Optional[Dict[str, torch.Tensor]]:
    # Prefer pytorch_model.bin (variable-bit quantized), then safetensors
    # model_int8.pt removed - fake-quant INT8 export disabled
    st_paths = [
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
        model_dir / "model.pt",
    ]
    for p in st_paths:
        if p.exists():
            if p.suffix == ".safetensors":
                from safetensors.torch import load_file
                return load_file(str(p))
            return torch.load(str(p), map_location="cpu")
    return None


def _read_variable_bit_meta(model_dir: Path) -> Dict[str, Dict]:
    meta_path = model_dir / "variable_bit_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys to canonical form without trailing '.weight'
    norm = {}
    for k, v in data.items():
        kk = k[:-7] if k.endswith(".weight") else k
        norm[kk] = v
    return norm


def _layer_full_names(module: nn.Module, prefix: str = "") -> Dict[str, nn.Module]:
    """
    Collect full names for Linear and Embedding submodules to enable stable replacement.
    """
    mapping = {}
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) or isinstance(child, nn.Embedding):
            mapping[child_prefix] = child
        # descend
        mapping.update(_layer_full_names(child, child_prefix))
    return mapping


def _bits_and_scale(name: str, meta: Dict[str, Dict], weight: torch.Tensor) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Resolve bits and (optional) scale from metadata; be forgiving with name variations.
    """
    # canonical key options
    candidates = [name, f"{name}.weight"]
    info = None
    for key in candidates:
        if key in meta:
            info = meta[key]
            break

    bits = None
    scale = None
    if isinstance(info, dict):
        b = info.get("bits", None)
        if isinstance(b, int):
            bits = b
        # allow both float or list
        sc = info.get("scale", None)
        if sc is not None:
            try:
                scale = torch.tensor(sc, dtype=torch.float32)
            except Exception:
                scale = None

    # Final guard: if bits is missing, loader will fall back to 8 inside Quantized* class.
    return bits, scale


def replace_with_quantized(model: nn.Module, meta: Dict[str, Dict]) -> Dict[str, int]:
    """
    Replace Linear/Embedding modules with quantized counterparts using variable_bit_meta.
    Returns small stats dict for logging.
    """
    stats = {
        "linear_replaced": 0,
        "embedding_replaced": 0,
        "skipped_no_bits": 0,
        "skipped_shape_mismatch": 0,
    }
    full = _layer_full_names(model)

    @torch.no_grad()
    def _set_module(parent: nn.Module, attr: str, new_mod: nn.Module):
        setattr(parent, attr, new_mod)

    # Walk once, replace via parent handles.
    for full_name, mod in full.items():
        parent_path, attr_name = full_name.rsplit(".", 1) if "." in full_name else ("", full_name)
        parent = model.get_submodule(parent_path) if parent_path else model

        if isinstance(mod, nn.Linear):
            bits, scale = _bits_and_scale(full_name, meta, mod.weight)
            # build quantized version
            qlin = QuantizedLinear(mod.in_features, mod.out_features, bits=bits, scale=(
                scale.item() if isinstance(scale, torch.Tensor) and scale.numel() == 1 else None),
                                   bias=(mod.bias is not None))
            # quantize from current fp32 weight
            qlin.from_fp32_(mod.weight, scale=scale)
            if mod.bias is not None:
                qlin.bias.data.copy_(mod.bias.detach().to(torch.float32))
            _set_module(parent, attr_name, qlin)
            stats["linear_replaced"] += 1

        elif isinstance(mod, nn.Embedding):
            bits, scale = _bits_and_scale(full_name, meta, mod.weight)
            qemb = QuantizedEmbedding(mod.num_embeddings, mod.embedding_dim, bits=bits, scale=(
                scale.item() if isinstance(scale, torch.Tensor) and scale.numel() == 1 else None))
            qemb.from_fp32_(mod.weight, scale=scale)
            _set_module(parent, attr_name, qemb)
            stats["embedding_replaced"] += 1

    return stats


def load_quantized_distilbert(model_dir: str,
                              device: str = "auto",
                              config_path: Optional[str] = None) -> nn.Module:
    """
    Build a DistilBERT model, then replace linear/embedding layers with quantized versions
    according to variable_bit_meta.json in model_dir.
    """
    model_dir_p = Path(model_dir)
    cfg_dir = Path(config_path) if config_path is not None else model_dir_p

    # Determine target device first
    dev = _device_of(device)
    
    # 1) Load config & instantiate architecture ALWAYS ON CPU FIRST
    cfg = AutoConfig.from_pretrained(str(cfg_dir))
    model = AutoModelForSequenceClassification.from_config(cfg)
    model = model.to("cpu")  # Force CPU before loading state dict

    # 2) Load state dict from model_dir ONLY (no fallback to avoid loading baseline weights)
    # Load to CPU explicitly
    state = _read_state_dict(model_dir_p)
    if state is None:
        raise FileNotFoundError(
            f"No model weights found in {model_dir_p}. "
            f"Expected one of: model_int8.pt, pytorch_model.bin, model.safetensors. "
            f"The compression export may have failed or files were moved."
        )
    
    # Ensure all state dict tensors are on CPU
    state = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Not fatal; quantized replacement will dequantize from whatever FP32 is present.
    # print(f"[debug] missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()

    # 3) Load variable-bit metadata
    meta = _read_variable_bit_meta(model_dir_p)

    # 4) Replace modules with quantized variants (on CPU)
    _ = replace_with_quantized(model, meta)
    
    # Force everything to CPU again after replacement
    model = model.to("cpu")

    # 5) Move to target device if requested
    if dev != "cpu":
        model = model.to(dev)
    return model
