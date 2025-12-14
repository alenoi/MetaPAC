# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

from transformers import AutoConfig, AutoModelForSequenceClassification


@dataclass
class ModelConfig:
    pretrained_name: str = "distilbert-base-uncased"
    dropout: float = 0.1
    num_labels: int = 2
    labels: Tuple[str, ...] = ("NEGATIVE", "POSITIVE")  # explicit label mapping


def build_model(cfg: ModelConfig):
    id2label: Dict[int, str] = {i: name for i, name in enumerate(cfg.labels)}
    label2id: Dict[str, int] = {name: i for i, name in enumerate(cfg.labels)}

    config = AutoConfig.from_pretrained(
        cfg.pretrained_name,
        num_labels=cfg.num_labels,
        classifier_dropout=cfg.dropout,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_name,
        config=config,
    )
    return model
