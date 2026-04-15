from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from metapac.src.compression.pipeline.phase_base import PhaseContext
from metapac.src.models.meta_predictor import TorchMetaPredictor, save_checkpoint_portable


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 8, num_labels: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None):
        if input_ids is None:
            raise ValueError("input_ids is required")

        x = input_ids.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if attention_mask is not None:
            x = x * attention_mask.float()

        hidden = self.encoder(x)
        logits = self.classifier(hidden)
        return SimpleNamespace(logits=logits)


@dataclass
class SmokeWorkspace:
    root: Path
    target_model_dir: Path
    meta_checkpoint_dir: Path
    output_dir: Path
    parameter_names: list[str]

    def build_config(self) -> dict[str, Any]:
        return {
            "mode": "compress",
            "output_dir": str(self.output_dir),
            "compression": {
                "target_model": str(self.target_model_dir),
                "baseline_model_config": str(self.target_model_dir),
                "meta_checkpoint": str(self.meta_checkpoint_dir),
                "output_dir": str(self.output_dir),
                "require_weight_change_for_success": True,
                "checkpoint_selector": {
                    "mode": "exact",
                    "exact_step": None,
                    "apply_to_teacher_and_validation": False,
                },
                "zone_assignment": {"method": "quantile"},
                "zones": {
                    "low": {
                        "quantile_min": 0.0,
                        "quantile_max": 0.34,
                        "action": "prune",
                    },
                    "medium": {
                        "quantile_min": 0.34,
                        "quantile_max": 0.67,
                        "action": "quantize",
                        "bits": 6,
                    },
                    "high": {
                        "quantile_min": 0.67,
                        "quantile_max": 1.01,
                        "action": "keep",
                    },
                },
                "pruning": {
                    "enabled": True,
                    "method": "magnitude",
                    "physical": False,
                    "head_pruning_ratio": 0.3,
                    "ffn_pruning_ratio": 0.3,
                },
                "quantization": {
                    "enabled": True,
                    "mode": "rank_aware_trim",
                    "bits_lower": 4,
                    "bits_upper": 8,
                    "per_channel": False,
                    "symmetric": True,
                    "util_target": 0.98,
                    "export_variable_bit": True,
                    "export_packed": False,
                    "export_int": False,
                },
                "fine_tuning": {
                    "enabled": True,
                    "output_dir": str(self.output_dir / "finetuned"),
                    "data": {
                        "dataset": "synthetic",
                        "dataset_config": None,
                        "max_length": 8,
                        "batch_size": 2,
                    },
                    "training": {
                        "num_epochs": 1,
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                        "warmup_ratio": 0.0,
                        "gradient_clip": 0.0,
                        "device": "cpu",
                        "num_workers": 0,
                    },
                    "distillation": {
                        "enabled": False,
                        "temperature": 2.0,
                        "alpha": 0.5,
                    },
                },
            },
        }


def build_smoke_workspace(tmp_path: Path) -> SmokeWorkspace:
    target_model_dir = tmp_path / "target_model"
    meta_checkpoint_dir = tmp_path / "meta_checkpoint"
    output_dir = tmp_path / "compression_output"

    model = TinyClassifier()
    parameter_names = list(model.state_dict().keys())

    _write_target_model(target_model_dir, model)
    _write_hook_stats(target_model_dir, parameter_names)
    _write_meta_checkpoint(meta_checkpoint_dir)

    return SmokeWorkspace(
        root=tmp_path,
        target_model_dir=target_model_dir,
        meta_checkpoint_dir=meta_checkpoint_dir,
        output_dir=output_dir,
        parameter_names=parameter_names,
    )


def create_model() -> TinyClassifier:
    return TinyClassifier()


def build_plan(model: nn.Module) -> tuple[dict[str, str], dict[str, float], dict[str, int | None]]:
    plan: dict[str, str] = {}
    importance_rankings: dict[str, float] = {}
    target_bits_map: dict[str, int | None] = {}

    parameter_names = [name for name, _ in model.named_parameters()]
    total = max(1, len(parameter_names) - 1)

    for index, name in enumerate(parameter_names):
        bucket = index % 3
        if bucket == 0:
            action = "keep"
        elif bucket == 1:
            action = "quantize"
        else:
            action = "prune"

        plan[name] = action
        importance_rankings[name] = 1.0 - (index / total)
        target_bits_map[name] = 6 if action == "quantize" else None

    return plan, importance_rankings, target_bits_map


def build_phase_context(
    tmp_path: Path,
    *,
    model: nn.Module | None = None,
    plan: dict[str, str] | None = None,
    importance_rankings: dict[str, float] | None = None,
    target_bits_map: dict[str, int | None] | None = None,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    baseline_dir: Path | None = None,
) -> PhaseContext:
    model = model or create_model()
    if plan is None or importance_rankings is None or target_bits_map is None:
        plan, importance_rankings, target_bits_map = build_plan(model)

    baseline_dir = baseline_dir or (tmp_path / "baseline")
    if not baseline_dir.exists():
        _write_target_model(baseline_dir, model)

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_config = config or {
        "output_dir": str(output_dir),
        "compression": {
            "output_dir": str(output_dir),
            "baseline_model_config": str(baseline_dir),
            "target_model": str(baseline_dir),
            "pruning": {
                "enabled": True,
                "method": "magnitude",
                "physical": False,
                "head_pruning_ratio": 0.3,
                "ffn_pruning_ratio": 0.3,
            },
            "quantization": {
                "enabled": True,
                "mode": "rank_aware_trim",
                "bits_lower": 4,
                "bits_upper": 8,
                "per_channel": False,
                "symmetric": True,
                "util_target": 0.98,
                "export_variable_bit": True,
                "export_packed": False,
                "export_int": False,
            },
            "fine_tuning": {
                "enabled": True,
                "output_dir": str(output_dir / "finetuned"),
                "data": {"dataset": "synthetic", "batch_size": 2},
                "training": {
                    "num_epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "gradient_clip": 0.0,
                    "device": "cpu",
                    "num_workers": 0,
                },
                "distillation": {"enabled": False},
            },
        },
    }

    return PhaseContext(
        model=model,
        config=full_config,
        output_path=output_dir,
        metadata=metadata or {},
        plan=plan,
        importance_rankings=importance_rankings,
        target_bits_map=target_bits_map,
    )


def stub_run_fine_tuning(config: dict[str, Any]) -> int:
    checkpoint_path = Path(config["model_checkpoint"]) / "pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    first_key = next(iter(state_dict))
    state_dict[first_key] = state_dict[first_key] + 0.125

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "pytorch_model.bin")

    results = {
        "history": [
            {
                "train_loss": 0.42,
                "train_accuracy": 0.75,
                "val_loss": 0.37,
                "val_accuracy": 0.7,
                "epoch": 1,
            }
        ],
        "best_val_accuracy": 0.7,
        "final_val_accuracy": 0.7,
    }
    with open(output_dir / "fine_tune_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return 0


def stub_integrate_variable_bit_export(
    model: nn.Module,
    combined_meta: dict[str, dict[str, Any]],
    export_dir: str,
    *,
    export_variable_bit: bool = True,
    use_cuda: bool = False,
    source_model_path: str | None = None,
) -> dict[str, Any]:
    del export_variable_bit, use_cuda, source_model_path

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), export_path / "pytorch_model.bin")
    torch.save(model.state_dict(), export_path / "model_state.pt")

    for filename, payload in {
        "config.json": {"architectures": ["TinyClassifier"], "num_labels": 2},
        "tokenizer.json": {"stub": True},
        "tokenizer_config.json": {"stub": True},
        "special_tokens_map.json": {"pad_token": "[PAD]"},
    }.items():
        with open(export_path / filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    with open(export_path / "vocab.txt", "w", encoding="utf-8") as handle:
        handle.write("[PAD]\n[UNK]\n")

    serializable_meta = _make_serializable(combined_meta)
    with open(export_path / "variable_bit_meta.json", "w", encoding="utf-8") as handle:
        json.dump(serializable_meta, handle, indent=2)

    stats = {
        "total_params": int(sum(param.numel() for param in model.parameters())),
        "fp32_MiB": 1.0,
        "quant_MiB": 0.25,
        "compression_ratio": 4.0,
        "layers": [
            {
                "name": name,
                "bits": meta.get("bits_final", meta.get("bits", 8)),
                "params": meta.get("numel", 0),
            }
            for name, meta in serializable_meta.items()
        ],
    }

    with open(export_path / "variable_bit_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return stats


def _write_target_model(target_dir: Path, model: nn.Module) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target_dir / "pytorch_model.bin")

    config = {"architectures": ["TinyClassifier"], "hidden_size": 8, "num_labels": 2}
    for filename, payload in {
        "config.json": config,
        "tokenizer.json": {"stub": True},
        "tokenizer_config.json": {"stub": True},
        "special_tokens_map.json": {"pad_token": "[PAD]"},
    }.items():
        with open(target_dir / filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    with open(target_dir / "vocab.txt", "w", encoding="utf-8") as handle:
        handle.write("[PAD]\n[UNK]\n")


def _write_hook_stats(target_model_dir: Path, parameter_names: list[str]) -> None:
    runs_dir = target_model_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for step in range(3):
        for index, name in enumerate(parameter_names, start=1):
            rows.append(
                {
                    "module": name,
                    "phase": "parameter",
                    "epoch": 0,
                    "step": step,
                    "act_mean": float(index * 0.5 + step * 0.1),
                    "act_std": float(index * 0.2 + step * 0.05),
                    "grad_mean": float(index * 0.1 + step * 0.02),
                }
            )

    pd.DataFrame(rows).to_csv(runs_dir / "parameter_stats_epoch0.csv", index=False)


def _write_meta_checkpoint(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["act_mean", "act_std", "grad_mean"]

    model = TorchMetaPredictor({"model": {"hidden_sizes": [], "dropout": 0.0}}, input_size=len(feature_names))
    with torch.no_grad():
        linear = model.network[0]
        linear.weight.copy_(torch.tensor([[0.8, 0.4, -0.2]], dtype=torch.float32))
        linear.bias.copy_(torch.tensor([0.05], dtype=torch.float32))

    sample = pd.DataFrame(
        [
            [0.5, 0.2, 0.1],
            [1.0, 0.4, 0.2],
            [1.5, 0.6, 0.3],
            [2.0, 0.8, 0.4],
        ],
        columns=feature_names,
        dtype=np.float64,
    )
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    imputer.fit(sample)
    scaler.fit(imputer.transform(sample))

    save_checkpoint_portable(
        checkpoint_dir=checkpoint_dir,
        model=model,
        imputer=imputer,
        scaler=scaler,
        feature_names=feature_names,
        target_name="importance",
        task_type="regression",
        metadata={"smoke": True},
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _make_serializable(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_make_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value