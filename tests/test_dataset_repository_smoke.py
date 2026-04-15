from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from metapac.src.utils.dataset_repository import load_managed_dataset, materialize_managed_dataset


def _build_local_source_dataset(source_dir: Path) -> None:
    train = Dataset.from_dict(
        {
            "text": [
                "alpha", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
            ],
            "label": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    test = Dataset.from_dict(
        {
            "text": ["alpha", "beta", "gamma", "delta"],
            "label": [0, 1, 0, 1],
        }
    )
    DatasetDict({"train": train, "test": test}).save_to_disk(str(source_dir))


@pytest.mark.smoke
def test_managed_dataset_repository_materializes_and_reuses_split_dataset(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_dataset"
    storage_root = tmp_path / "dataset_repo"
    _build_local_source_dataset(source_dir)

    source_cfg = {
        "mode": "disk",
        "path": str(source_dir),
        "storage": {
            "root": str(storage_root),
            "mode": "split",
        },
    }
    processing_cfg = {
        "split_strategy": "fixed_ratio_no_overlap",
        "val_split_ratio": 0.2,
        "test_split_ratio": 0.2,
        "seed": 7,
        "deduplicate_by_text": True,
        "enforce_no_text_overlap": True,
        "storage": {
            "root": str(storage_root),
            "mode": "split",
        },
    }

    first_path = materialize_managed_dataset(
        "local_overlap_dataset",
        None,
        source_cfg=source_cfg,
        processing_cfg=processing_cfg,
    )
    second_path = materialize_managed_dataset(
        "local_overlap_dataset",
        None,
        source_cfg=source_cfg,
        processing_cfg=processing_cfg,
    )
    dataset = load_managed_dataset(
        "local_overlap_dataset",
        None,
        source_cfg=source_cfg,
        processing_cfg=processing_cfg,
    )

    assert first_path == second_path
    assert first_path.exists()
    assert set(dataset.keys()) == {"train", "validation", "test"}

    train_texts = set(dataset["train"]["text"])
    validation_texts = set(dataset["validation"]["text"])
    test_texts = set(dataset["test"]["text"])
    assert not (train_texts & validation_texts)
    assert not (train_texts & test_texts)
    assert not (validation_texts & test_texts)


@pytest.mark.smoke
def test_managed_dataset_repository_materializes_raw_dataset(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_dataset_raw"
    storage_root = tmp_path / "dataset_repo_raw"
    _build_local_source_dataset(source_dir)

    dataset = load_managed_dataset(
        "local_raw_dataset",
        None,
        source_cfg={
            "mode": "disk",
            "path": str(source_dir),
            "storage": {
                "root": str(storage_root),
                "mode": "raw",
            },
        },
        processing_cfg={
            "storage": {
                "root": str(storage_root),
                "mode": "raw",
            },
        },
    )

    assert set(dataset.keys()) == {"train", "test"}
    assert (storage_root).exists()