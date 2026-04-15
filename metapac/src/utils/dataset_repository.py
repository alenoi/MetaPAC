from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset, load_from_disk

from .hf_sources import resolve_dataset_source


def _as_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _slugify(value: str) -> str:
    slug = [ch.lower() if ch.isalnum() else "-" for ch in value]
    normalized = "".join(slug).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized or "dataset"


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


@dataclass(frozen=True)
class ManagedDatasetSpec:
    root: Path
    mode: str
    split_strategy: str
    val_split_ratio: Optional[float]
    test_split_ratio: Optional[float]
    seed: int
    deduplicate_by_text: bool
    enforce_no_text_overlap: bool
    allow_online_download: bool
    dataset_name: str
    dataset_config: Optional[str]
    source_cfg: dict[str, Any]


def resolve_dataset_reference(dataset_name: str, dataset_config: str | None = None) -> tuple[str, Optional[str]]:
    name = str(dataset_name or "").lower()
    if name in {"sst2", "sst-2", "glue/sst2", "glue_sst2"}:
        return "glue", "sst2"
    if name == "imdb":
        return "imdb", None
    if name in {"wos", "wos11967", "waashk/wos11967", "web_of_science"}:
        return "waashk/wos11967", None
    return dataset_name, dataset_config


def _infer_storage_mode(processing_cfg: Mapping[str, Any]) -> str:
    cfg = _as_dict(processing_cfg)
    storage_cfg = _as_dict(cfg.get("storage"))
    explicit = storage_cfg.get("mode")
    if explicit:
        return str(explicit).lower()

    split_strategy = str(cfg.get("split_strategy", "default") or "default").lower()
    if split_strategy != "default":
        return "split"
    if cfg.get("val_split_ratio") is not None or cfg.get("test_split_ratio") is not None:
        return "split"
    if bool(cfg.get("deduplicate_by_text", False)) or bool(cfg.get("enforce_no_text_overlap", False)):
        return "split"
    return "raw"


def resolve_managed_dataset_spec(
    dataset_name: str,
    dataset_config: str | None = None,
    *,
    source_cfg: Mapping[str, Any] | None = None,
    processing_cfg: Mapping[str, Any] | None = None,
) -> ManagedDatasetSpec:
    resolved_name, resolved_config = resolve_dataset_reference(dataset_name, dataset_config)
    cfg = _as_dict(processing_cfg)
    storage_cfg = _as_dict(cfg.get("storage"))
    root = Path(storage_cfg.get("root") or "metapac/artifacts/datasets")
    return ManagedDatasetSpec(
        root=root,
        mode=_infer_storage_mode(cfg),
        split_strategy=str(cfg.get("split_strategy", "default") or "default").lower(),
        val_split_ratio=float(cfg["val_split_ratio"]) if cfg.get("val_split_ratio") is not None else None,
        test_split_ratio=float(cfg["test_split_ratio"]) if cfg.get("test_split_ratio") is not None else None,
        seed=int(cfg.get("seed", 42)),
        deduplicate_by_text=bool(cfg.get("deduplicate_by_text", False)),
        enforce_no_text_overlap=bool(cfg.get("enforce_no_text_overlap", False)),
        allow_online_download=bool(storage_cfg.get("allow_online_download", True)),
        dataset_name=resolved_name,
        dataset_config=resolved_config,
        source_cfg=_as_dict(source_cfg),
    )


def _dataset_storage_root(spec: ManagedDatasetSpec) -> Path:
    dataset_slug = _slugify(spec.dataset_name)
    config_slug = _slugify(spec.dataset_config or "default")
    source_fingerprint = _json_hash({
        "dataset_name": spec.dataset_name,
        "dataset_config": spec.dataset_config,
        "source_cfg": spec.source_cfg,
    })
    return spec.root / dataset_slug / config_slug / source_fingerprint


def _raw_dataset_path(spec: ManagedDatasetSpec) -> Path:
    return _dataset_storage_root(spec) / "raw"


def _split_dataset_path(spec: ManagedDatasetSpec) -> Path:
    split_fingerprint = _json_hash({
        "split_strategy": spec.split_strategy,
        "val_split_ratio": spec.val_split_ratio,
        "test_split_ratio": spec.test_split_ratio,
        "seed": spec.seed,
        "deduplicate_by_text": spec.deduplicate_by_text,
        "enforce_no_text_overlap": spec.enforce_no_text_overlap,
    })
    return _dataset_storage_root(spec) / f"split-{spec.split_strategy}-{split_fingerprint}"


def _load_source_dataset(spec: ManagedDatasetSpec):
    resolved = resolve_dataset_source(spec.dataset_name, spec.dataset_config, spec.source_cfg)
    if resolved.mode == "disk":
        return load_from_disk(resolved.path)

    kwargs = dict(resolved.load_dataset_kwargs)
    try:
        return load_dataset(resolved.name, resolved.config_name, **kwargs)
    except Exception:
        if not spec.allow_online_download:
            raise
        download_cfg = kwargs.get("download_config")
        if isinstance(download_cfg, DownloadConfig) and download_cfg.local_files_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=False)
            return load_dataset(resolved.name, resolved.config_name, **kwargs)
        raise


def _ensure_datasetdict(dataset: Any) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(f"Expected DatasetDict, got {type(dataset)!r}")


def _train_val_split(dataset: Dataset, test_size: float, seed: int):
    try:
        return dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    except Exception:
        return dataset.train_test_split(test_size=test_size, seed=seed)


def infer_text_column(raw: DatasetDict) -> str:
    for split_name in ("train", "validation", "test"):
        if split_name not in raw or len(raw[split_name]) == 0:
            continue
        sample = raw[split_name][0]
        for candidate in [
            "sentence",
            "text",
            "sentence1",
            "sentence2",
            "article",
            "review",
            "query",
            "premise",
            "hypothesis",
        ]:
            if candidate in sample and isinstance(sample[candidate], str):
                return candidate
        for key, value in sample.items():
            if isinstance(value, str):
                return key
    raise ValueError("Could not infer text column from dataset")


def _as_labeled_datasetdict(raw: DatasetDict, text_key: str) -> Dataset:
    records: dict[str, list[Any]] = {text_key: [], "label": []}
    for split_name, split_ds in raw.items():
        if text_key not in split_ds.column_names or "label" not in split_ds.column_names:
            continue
        for text, label in zip(split_ds[text_key], split_ds["label"]):
            records[text_key].append(text)
            records["label"].append(int(label))
    if not records[text_key]:
        raise ValueError("No labeled examples found to build split dataset")
    return Dataset.from_dict(records)


def _deduplicate_text_labels(dataset: Dataset, text_key: str) -> Dataset:
    text_to_label: dict[str, int] = {}
    for text, label in zip(dataset[text_key], dataset["label"]):
        label = int(label)
        if text in text_to_label and text_to_label[text] != label:
            continue
        text_to_label[text] = label
    return Dataset.from_dict({text_key: list(text_to_label.keys()), "label": list(text_to_label.values())})


def check_no_text_overlap(dataset_dict: DatasetDict, text_key: str) -> None:
    split_sets: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        if split_name in dataset_dict and text_key in dataset_dict[split_name].column_names:
            split_sets[split_name] = set(dataset_dict[split_name][text_key])
    split_names = list(split_sets.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"Text overlap detected between '{left}' and '{right}': {len(overlap)} samples"
                )


def _build_fixed_ratio_no_overlap_splits(
    raw: DatasetDict,
    text_key: str,
    *,
    seed: int,
    val_split_ratio: float,
    test_split_ratio: float,
    deduplicate_by_text: bool,
) -> DatasetDict:
    if val_split_ratio <= 0 or test_split_ratio <= 0 or (val_split_ratio + test_split_ratio) >= 1.0:
        raise ValueError(
            "For fixed_ratio_no_overlap, val_split_ratio and test_split_ratio must be > 0 and sum to < 1"
        )
    labeled = _as_labeled_datasetdict(raw, text_key)
    if deduplicate_by_text:
        labeled = _deduplicate_text_labels(labeled, text_key)
    test_split = _train_val_split(labeled, test_size=test_split_ratio, seed=seed)
    train_val = test_split["train"]
    test_ds = test_split["test"]
    val_from_remaining = val_split_ratio / (1.0 - test_split_ratio)
    val_split = _train_val_split(train_val, test_size=val_from_remaining, seed=seed)
    return DatasetDict({
        "train": val_split["train"],
        "validation": val_split["test"],
        "test": test_ds,
    })


def _build_split_dataset(raw: DatasetDict, spec: ManagedDatasetSpec) -> DatasetDict:
    text_key = infer_text_column(raw)
    strategy = spec.split_strategy
    if strategy == "fixed_ratio_no_overlap":
        val_ratio = spec.val_split_ratio if spec.val_split_ratio is not None else 0.1
        test_ratio = spec.test_split_ratio if spec.test_split_ratio is not None else 0.1
        dataset = _build_fixed_ratio_no_overlap_splits(
            raw,
            text_key,
            seed=spec.seed,
            val_split_ratio=val_ratio,
            test_split_ratio=test_ratio,
            deduplicate_by_text=spec.deduplicate_by_text,
        )
    else:
        dataset = DatasetDict(raw)
        if "validation" not in dataset:
            if spec.val_split_ratio is not None:
                split = _train_val_split(dataset["train"], test_size=spec.val_split_ratio, seed=spec.seed)
                dataset["train"] = split["train"]
                dataset["validation"] = split["test"]
            else:
                raise ValueError(
                    "Managed split dataset requires a validation split. Configure val_split_ratio or use split_strategy='fixed_ratio_no_overlap'."
                )
    if spec.enforce_no_text_overlap:
        check_no_text_overlap(dataset, text_key)
    return dataset


def materialize_managed_dataset(
    dataset_name: str,
    dataset_config: str | None = None,
    *,
    source_cfg: Mapping[str, Any] | None = None,
    processing_cfg: Mapping[str, Any] | None = None,
) -> Path:
    spec = resolve_managed_dataset_spec(
        dataset_name,
        dataset_config,
        source_cfg=source_cfg,
        processing_cfg=processing_cfg,
    )
    spec.root.mkdir(parents=True, exist_ok=True)

    raw_path = _raw_dataset_path(spec)
    if not raw_path.exists():
        dataset = _ensure_datasetdict(_load_source_dataset(spec))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(raw_path))

    if spec.mode == "raw":
        return raw_path

    split_path = _split_dataset_path(spec)
    if split_path.exists():
        return split_path

    raw_dataset = _ensure_datasetdict(load_from_disk(str(raw_path)))
    split_dataset = _build_split_dataset(raw_dataset, spec)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_dataset.save_to_disk(str(split_path))
    return split_path


def load_managed_dataset(
    dataset_name: str,
    dataset_config: str | None = None,
    *,
    source_cfg: Mapping[str, Any] | None = None,
    processing_cfg: Mapping[str, Any] | None = None,
) -> DatasetDict:
    dataset_path = materialize_managed_dataset(
        dataset_name,
        dataset_config,
        source_cfg=source_cfg,
        processing_cfg=processing_cfg,
    )
    return _ensure_datasetdict(load_from_disk(str(dataset_path)))