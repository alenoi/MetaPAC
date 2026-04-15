from __future__ import annotations

import pytest

from metapac.src.pipeline import run


@pytest.mark.smoke
@pytest.mark.pipeline
def test_run_assigns_run_specific_hook_and_meta_paths() -> None:
    config = {
        "mode": "none",
        "run_id": "qwen3_06b_wos_fast",
        "baseline_finetune": {
            "experiment_name": "baseline_qwen3_06b_wos_fast",
            "output_dir": "targets/qwen3_06b/runs/baseline_qwen3_06b_wos_fast",
            "train": {},
        },
        "input_dir": "metapac/artifacts/raw",
        "data": {
            "path": "metapac/artifacts/meta_dataset/meta_dataset_qwen3_wos_fast.parquet",
        },
        "outputs": {
            "meta_dataset_path": "metapac/artifacts/meta_dataset/meta_dataset_qwen3_wos_fast.parquet",
        },
    }

    exit_code = run(config)

    assert exit_code == 0
    assert config["baseline_finetune"]["train"]["hook_output_dir"] == (
        "targets/qwen3_06b/runs/baseline_qwen3_06b_wos_fast/artifacts/raw"
    )
    assert config["input_dir"] == "targets/qwen3_06b/runs/baseline_qwen3_06b_wos_fast/artifacts/raw"
    assert config["meta_dataset_path"] == "metapac/artifacts/meta_dataset/qwen3_06b_wos_fast/meta_dataset.parquet"
    assert config["outputs"]["meta_dataset_path"] == config["meta_dataset_path"]
    assert config["data"]["path"] == config["meta_dataset_path"]