from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from datasets import DownloadConfig, load_dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class ModelSourceSpec:
    reference: str
    from_pretrained_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DatasetSourceSpec:
    mode: str
    path: str | None
    name: str | None
    config_name: str | None
    load_dataset_kwargs: dict[str, Any]


def resolve_model_source(reference: str, source_cfg: Mapping[str, Any] | None = None) -> ModelSourceSpec:
    cfg = _as_dict(source_cfg)
    mode = str(cfg.get("mode", "auto") or "auto").lower()
    path_value = cfg.get("path") or cfg.get("local_path")
    resolved_reference = str(path_value or reference)

    local_files_only = cfg.get("local_files_only")
    if local_files_only is None:
        local_files_only = mode == "local"

    if mode == "local":
        candidate = Path(resolved_reference)
        if not candidate.exists():
            raise FileNotFoundError(f"Local model source not found: {candidate}")
        resolved_reference = str(candidate.resolve())
    elif path_value:
        candidate = Path(resolved_reference)
        if candidate.exists():
            resolved_reference = str(candidate.resolve())

    kwargs: dict[str, Any] = {
        "local_files_only": bool(local_files_only),
    }
    if cfg.get("cache_dir"):
        kwargs["cache_dir"] = str(cfg["cache_dir"])
    if cfg.get("revision"):
        kwargs["revision"] = str(cfg["revision"])
    if cfg.get("trust_remote_code") is not None:
        kwargs["trust_remote_code"] = bool(cfg["trust_remote_code"])

    return ModelSourceSpec(reference=resolved_reference, from_pretrained_kwargs=kwargs)


def load_tokenizer_from_source(
    reference: str,
    source_cfg: Mapping[str, Any] | None = None,
    **tokenizer_kwargs: Any,
):
    spec = resolve_model_source(reference, source_cfg)
    kwargs = dict(spec.from_pretrained_kwargs)
    kwargs.update(tokenizer_kwargs)
    return AutoTokenizer.from_pretrained(spec.reference, **kwargs)


def load_sequence_classification_model_from_source(
    reference: str,
    source_cfg: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
):
    spec = resolve_model_source(reference, source_cfg)
    kwargs = dict(spec.from_pretrained_kwargs)
    kwargs.update(model_kwargs)
    return AutoModelForSequenceClassification.from_pretrained(spec.reference, **kwargs)


def resolve_dataset_source(
    dataset_name: str,
    dataset_config: str | None = None,
    source_cfg: Mapping[str, Any] | None = None,
) -> DatasetSourceSpec:
    cfg = _as_dict(source_cfg)
    mode = str(cfg.get("mode", "hub") or "hub").lower()
    path_value = cfg.get("path") or cfg.get("local_path")
    path = str(path_value) if path_value else None

    local_files_only = cfg.get("local_files_only")
    if local_files_only is None:
        local_files_only = mode in {"disk", "local", "file"}

    if mode in {"disk", "local"}:
        if not path:
            raise ValueError("Dataset source mode 'disk' requires 'path'")
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Local dataset source not found: {dataset_path}")
        return DatasetSourceSpec(
            mode="disk",
            path=str(dataset_path.resolve()),
            name=None,
            config_name=None,
            load_dataset_kwargs={},
        )

    name = str(cfg.get("name") or dataset_name)
    config_name = cfg.get("config")
    if config_name is None:
        config_name = cfg.get("dataset_config", dataset_config)
    if config_name is not None:
        config_name = str(config_name)

    kwargs: dict[str, Any] = {}
    if cfg.get("cache_dir"):
        kwargs["cache_dir"] = str(cfg["cache_dir"])
    if cfg.get("data_dir"):
        kwargs["data_dir"] = str(cfg["data_dir"])
    if cfg.get("data_files") is not None:
        kwargs["data_files"] = cfg["data_files"]
    if cfg.get("revision"):
        kwargs["revision"] = str(cfg["revision"])
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)

    if mode == "file":
        if not path:
            raise ValueError("Dataset source mode 'file' requires 'path'")
        kwargs.setdefault("data_files", path)
        name = str(cfg.get("file_format") or dataset_name)
        config_name = None

    return DatasetSourceSpec(
        mode=mode,
        path=path,
        name=name,
        config_name=config_name,
        load_dataset_kwargs=kwargs,
    )


def load_dataset_from_source(
    dataset_name: str,
    dataset_config: str | None = None,
    *,
    split: str | None = None,
    source_cfg: Mapping[str, Any] | None = None,
):
    spec = resolve_dataset_source(dataset_name, dataset_config, source_cfg)
    if spec.mode == "disk":
        dataset = load_from_disk(spec.path)
        if split is not None:
            return dataset[split]
        return dataset

    kwargs = dict(spec.load_dataset_kwargs)
    if split is not None:
        kwargs["split"] = split
    return load_dataset(spec.name, spec.config_name, **kwargs)