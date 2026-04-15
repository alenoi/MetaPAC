from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
from transformers import set_seed

from metapac.src.utils.logging_utils import setup_logger


def setup_baseline_logger(logging_cfg: Dict[str, Any] | None = None, default_log_dir: str | None = None):
    return setup_logger(
        "baseline",
        settings=logging_cfg,
        default_log_dir=default_log_dir,
    )


def set_all_seeds(seed: int) -> None:
    set_seed(seed)


def infer_precision(fp16_flag: str = "auto") -> Tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    capability = torch.cuda.get_device_capability()
    if fp16_flag == "bf16":
        return False, True
    if fp16_flag == "fp16":
        return True, False
    if fp16_flag == "none":
        return False, False
    bf16_ok = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else (capability[0] >= 8)
    if bf16_ok:
        return False, True
    return True, False


def count_parameters(model) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def device_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        return {
            "device": "cuda",
            "name": props.name,
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "capability": f"{props.major}.{props.minor}",
        }
    return {"device": "cpu", "name": "cpu", "total_vram_gb": 0.0, "capability": ""}


def max_memory_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


def save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)