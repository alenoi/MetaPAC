from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from metapac.src.model_handlers.common import device_info, save_json, set_all_seeds, setup_baseline_logger
from metapac.src.models import ModelConfig, build_model
from metapac.src.utils import HookHFCallback
from metapac.src.utils.dataset_repository import load_managed_dataset, resolve_dataset_reference
from metapac.src.utils.hf_sources import load_tokenizer_from_source

from .base import ModelHandler
from .registry import register_model_handler


@dataclass
class DataConfig:
    name: str = "sst2"
    max_length: int = 256
    val_split_ratio: Optional[float] = None
    test_split_ratio: Optional[float] = None
    seed: int = 42
    split_strategy: str = "default"
    deduplicate_by_text: bool = False
    enforce_no_text_overlap: bool = False
    source: Optional[Dict[str, object]] = None


def load_tokenizer(model_name: str, source: Optional[Dict[str, object]] = None) -> PreTrainedTokenizerBase:
    tokenizer = load_tokenizer_from_source(model_name, source, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_prepare_datasets(
    cfg: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[DatasetDict, int, Tuple[str, ...]]:
    name = cfg.name.lower()

    if name in ("sst2", "sst-2", "glue/sst2", "glue_sst2"):
        label_names: Tuple[str, ...] = ("NEGATIVE", "POSITIVE")
        text_key = "sentence"
    elif name == "imdb":
        label_names = ("NEGATIVE", "POSITIVE")
        text_key = "text"
    elif name in ("wos", "wos11967", "waashk/wos11967", "web_of_science"):
        label_names = tuple()
        text_key = "text"
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    source_cfg = cfg.source if isinstance(cfg.source, dict) else None
    dataset_name, dataset_config = resolve_dataset_reference(cfg.name)
    raw = load_managed_dataset(
        dataset_name,
        dataset_config,
        source_cfg=source_cfg,
        processing_cfg={
            "split_strategy": cfg.split_strategy,
            "val_split_ratio": cfg.val_split_ratio,
            "test_split_ratio": cfg.test_split_ratio,
            "seed": cfg.seed,
            "deduplicate_by_text": cfg.deduplicate_by_text,
            "enforce_no_text_overlap": cfg.enforce_no_text_overlap,
            "storage": source_cfg.get("storage") if source_cfg else None,
        },
    )

    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_key],
            truncation=True,
            padding=False,
            max_length=cfg.max_length,
        )

    train_cols = raw["train"].column_names if "train" in raw else (
        raw["validation"].column_names if "validation" in raw else []
    )
    keep_cols = [column for column in (text_key, "label") if column in train_cols]

    encoded = raw.map(
        _tokenize,
        batched=True,
        remove_columns=[column for column in train_cols if column not in keep_cols],
    )

    def _to_int_label(example: Dict[str, Any]) -> Dict[str, Any]:
        if "label" in example:
            example["label"] = int(example["label"])
        return example

    encoded = encoded.map(_to_int_label)

    if "test" in encoded:
        drop = [column for column in ("label", "labels") if column in encoded["test"].column_names]
        if drop:
            encoded["test"] = encoded["test"].remove_columns(drop)

    observed_labels = set()
    for split_name in ("train", "validation"):
        if split_name in encoded and "label" in encoded[split_name].column_names:
            observed_labels.update(int(value) for value in encoded[split_name]["label"])

    if not observed_labels:
        num_labels = 2
        if not label_names:
            label_names = ("LABEL_0", "LABEL_1")
    else:
        sorted_labels = sorted(observed_labels)
        mapping = {old: new for new, old in enumerate(sorted_labels)}

        if any(old != new for old, new in mapping.items()):
            def _remap_label(example: Dict[str, Any]) -> Dict[str, Any]:
                if "label" in example:
                    example["label"] = int(mapping[int(example["label"])])
                return example

            for split_name in ("train", "validation"):
                if split_name in encoded and "label" in encoded[split_name].column_names:
                    encoded[split_name] = encoded[split_name].map(_remap_label)

        num_labels = len(sorted_labels)
        if not label_names or len(label_names) != num_labels:
            label_names = tuple(f"LABEL_{index}" for index in range(num_labels))

    return encoded, num_labels, label_names


def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    observed = np.unique(np.concatenate([labels, preds]))
    is_binary = len(observed) <= 2 and set(observed.tolist()).issubset({0, 1})
    average = "binary" if is_binary else "macro"
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average=average, zero_division=0)),
    }


def _align_steps(cfg: Dict[str, Any]) -> Tuple[str, int | None, int | None]:
    eval_strategy = str(cfg.get("eval_strategy", cfg.get("evaluation_strategy", "epoch")))
    logging_steps = int(cfg.get("logging_steps", 50))
    eval_steps = cfg.get("eval_steps")
    save_steps = cfg.get("save_steps")

    if eval_strategy == "steps":
        if eval_steps is None:
            eval_steps = max(logging_steps * 3, 200)
        else:
            eval_steps = int(eval_steps)

        if save_steps is None:
            save_steps = eval_steps
        else:
            save_steps = int(save_steps)

        if save_steps % eval_steps != 0:
            save_steps = math.ceil(save_steps / eval_steps) * eval_steps
    else:
        eval_steps = None
        save_steps = None

    return eval_strategy, eval_steps, save_steps


def train_and_evaluate(
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    model: Any,
    out_dir: str,
    train_cfg: Dict[str, Any],
    report_to: Tuple[str, ...] = ("tensorboard",),
    dataloader_num_workers: int = 2,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(int(train_cfg.get("seed", 42)))
    eval_strategy, eval_steps, save_steps = _align_steps(train_cfg)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 64)),
        eval_strategy=eval_strategy,
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        logging_first_step=True,
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "f1")),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        dataloader_num_workers=dataloader_num_workers,
        report_to=list(report_to),
        **({"eval_steps": eval_steps} if eval_steps is not None else {}),
        **({"save_steps": save_steps} if save_steps is not None else {}),
    )

    callbacks = []
    if bool(train_cfg.get("collect_hooks", True)):
        callbacks.append(
            HookHFCallback(
                model,
                out_dir=str(train_cfg.get("hook_output_dir", "metapac/artifacts/raw")),
                capture_every_n_steps=int(train_cfg.get("collect_hooks_every_n_steps", 1)),
                include_quantiles=bool(train_cfg.get("hook_include_quantiles", True)),
            )
        )
    if bool(train_cfg.get("early_stopping", True)):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 4)),
                early_stopping_threshold=float(train_cfg.get("early_stopping_threshold", 0.002)),
            )
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets.get("train"),
        eval_dataset=datasets.get("validation"),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    try:
        if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    train_metrics: Dict[str, Any] = {}
    if "train" in datasets:
        train_result = trainer.train()
        train_metrics = {key: float(value) for key, value in train_result.metrics.items()}
        try:
            save_json(os.path.join(out_dir, "metrics_train.json"), train_metrics)
        except Exception:
            pass

    eval_metrics: Dict[str, Any] = {}
    if "validation" in datasets:
        eval_metrics = {key: float(value) for key, value in trainer.evaluate().items()}
        trainer.save_metrics("eval", eval_metrics)
        try:
            save_json(os.path.join(out_dir, "metrics_eval.json"), eval_metrics)
        except Exception:
            pass

    test_preds = None
    if "test" in datasets:
        test_ds = datasets["test"]
        for column in ("label", "labels"):
            if column in test_ds.column_names:
                test_ds = test_ds.remove_columns(column)
        pred_output = trainer.predict(test_ds)
        test_preds = np.argmax(pred_output.predictions, axis=-1)
        try:
            np.save(os.path.join(out_dir, "test_preds.npy"), test_preds)
        except Exception:
            pass

    summary = {
        "train": train_metrics,
        "eval": eval_metrics,
        "n_train": int(datasets["train"].num_rows) if "train" in datasets else 0,
        "n_val": int(datasets["validation"].num_rows) if "validation" in datasets else 0,
        "n_test": int(datasets["test"].num_rows) if "test" in datasets else 0,
        "test_preds_shape": None if test_preds is None else tuple(map(int, test_preds.shape)),
        "model_params_million": round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 2),
        "max_cuda_mem_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if torch.cuda.is_available() else None,
    }
    try:
        save_json(os.path.join(out_dir, "summary.json"), summary)
    except Exception:
        pass
    return summary


class HuggingFaceSequenceClassificationHandler(ModelHandler):
    """Common baseline fine-tuning flow for HF sequence classification models."""

    priority = 10

    def run_baseline_finetune(self, config: dict[str, Any]) -> int:
        finetune_cfg = config.get("baseline_finetune", {})

        print("[pipeline] ========================================")
        print(f"[pipeline] Running baseline fine-tuning via {self.handler_name}")
        print("[pipeline] ========================================")

        exp_name = finetune_cfg.get("experiment_name", "baseline")
        out_dir = finetune_cfg.get("output_dir", f"targets/generic/runs/{exp_name}")
        os.makedirs(out_dir, exist_ok=True)

        logger = setup_baseline_logger(finetune_cfg.get("logging"), default_log_dir=os.path.join(out_dir, "logs"))

        train_cfg = finetune_cfg.setdefault("train", {})
        seed = int(train_cfg.get("seed", 42))
        set_all_seeds(seed)
        save_json(os.path.join(out_dir, "device.json"), device_info())
        save_json(os.path.join(out_dir, "config_resolved.json"), finetune_cfg)

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16_ok = torch.cuda.is_available()
        train_cfg.setdefault("bf16", bool(bf16_ok))
        train_cfg.setdefault("fp16", bool(not bf16_ok and fp16_ok))

        model_name = finetune_cfg["model"]["pretrained_name"]
        model_source = finetune_cfg["model"].get("source")

        tokenizer = load_tokenizer(model_name, model_source)
        data_cfg = DataConfig(**finetune_cfg["dataset"])
        datasets, num_labels, label_names = load_and_prepare_datasets(data_cfg, tokenizer)

        model_cfg = ModelConfig(
            pretrained_name=model_name,
            dropout=finetune_cfg["model"].get("dropout", 0.1),
            num_labels=num_labels,
            labels=label_names,
            torch_dtype=finetune_cfg["model"].get("torch_dtype"),
            source=model_source,
        )
        model = build_model(model_cfg)

        if train_cfg.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        summary = train_and_evaluate(
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            out_dir=out_dir,
            train_cfg=train_cfg,
            report_to=tuple(finetune_cfg.get("logging", {}).get("report_to", ["tensorboard"])),
            dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        )
        save_json(os.path.join(out_dir, "summary_main.json"), summary)
        logger.info("[pipeline] Baseline fine-tuning completed successfully")
        logger.info("[pipeline] Results saved to: %s", out_dir)
        return 0


@register_model_handler
class DistilBertSequenceClassificationHandler(HuggingFaceSequenceClassificationHandler):
    handler_name = "distilbert_sequence_classifier"
    model_family = "distilbert"
    priority = 100


@register_model_handler
class GPT2SequenceClassificationHandler(HuggingFaceSequenceClassificationHandler):
    handler_name = "gpt2_sequence_classifier"
    model_family = "gpt2"
    priority = 100


@register_model_handler
class QwenSequenceClassificationHandler(HuggingFaceSequenceClassificationHandler):
    handler_name = "qwen_sequence_classifier"
    model_family = "qwen"
    priority = 100


@register_model_handler
class AutoSequenceClassificationHandler(HuggingFaceSequenceClassificationHandler):
    handler_name = "auto_sequence_classifier"
    supported_model_prefixes = ()
    priority = -100