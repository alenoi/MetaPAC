# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import torch
from transformers import AutoConfig

from metapac.src.utils.hf_sources import resolve_model_source, load_sequence_classification_model_from_source


@dataclass
class ModelConfig:
    pretrained_name: str = "distilbert-base-uncased"
    dropout: float = 0.1
    num_labels: int = 2
    labels: Tuple[str, ...] = ("NEGATIVE", "POSITIVE")  # explicit label mapping
    torch_dtype: str | None = None
    source: dict | None = None


def _resolve_torch_dtype(dtype_name: str | None):
    if not dtype_name:
        return None
    key = str(dtype_name).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(key)


def build_model(cfg: ModelConfig):
    id2label: Dict[int, str] = {i: name for i, name in enumerate(cfg.labels)}
    label2id: Dict[str, int] = {name: i for i, name in enumerate(cfg.labels)}

    source_spec = resolve_model_source(cfg.pretrained_name, cfg.source)

    config = AutoConfig.from_pretrained(
        source_spec.reference,
        num_labels=cfg.num_labels,
        classifier_dropout=cfg.dropout,
        id2label=id2label,
        label2id=label2id,
        **source_spec.from_pretrained_kwargs,
    )
    resolved_dtype = _resolve_torch_dtype(cfg.torch_dtype)

    model = load_sequence_classification_model_from_source(
        cfg.pretrained_name,
        cfg.source,
        config=config,
        **({"torch_dtype": resolved_dtype} if resolved_dtype is not None else {}),
    )
    return model
