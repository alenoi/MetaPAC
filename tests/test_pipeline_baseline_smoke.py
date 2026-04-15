from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from metapac.src.pipeline import run


@pytest.mark.smoke
@pytest.mark.pipeline
def test_run_baseline_mode_executes_registered_handler(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class StubHandler:
        handler_name = "stub"

        def run_baseline_finetune(self, config: dict) -> int:
            observed["hook_output_dir"] = config["baseline_finetune"]["train"]["hook_output_dir"]
            observed["meta_dataset_path"] = config["meta_dataset_path"]
            observed["output_dir"] = config["baseline_finetune"]["output_dir"]
            Path(config["baseline_finetune"]["output_dir"]).mkdir(parents=True, exist_ok=True)
            return 0

    config = {
        "mode": "baseline_finetune",
        "run_id": "distilgpt2_imdb_fast",
        "baseline_finetune": {
            "experiment_name": "baseline_distilgpt2_imdb_fast",
            "output_dir": str(tmp_path / "targets" / "distilgpt2" / "runs" / "baseline_distilgpt2_imdb_fast"),
            "model": {"pretrained_name": "distilgpt2"},
            "dataset": {"name": "imdb"},
            "train": {},
        },
    }

    monkeypatch.setattr("metapac.src.pipeline.create_handler_for_config", lambda cfg: StubHandler())

    exit_code = run(config)

    assert exit_code == 0
    assert observed["output_dir"] == config["baseline_finetune"]["output_dir"]
    assert observed["hook_output_dir"] == str(
        Path(config["baseline_finetune"]["output_dir"]) / "artifacts" / "raw"
    )
    assert str(observed["meta_dataset_path"]).endswith("distilgpt2_imdb_fast/meta_dataset.parquet")


@pytest.mark.smoke
@pytest.mark.pipeline
def test_auto_mode_preserves_nested_baseline_defaults(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class StubHandler:
        handler_name = "stub"

        def run_baseline_finetune(self, config: dict) -> int:
            observed["model_name"] = config["baseline_finetune"]["model"]["pretrained_name"]
            observed["output_dir"] = config["baseline_finetune"]["output_dir"]
            Path(config["baseline_finetune"]["output_dir"]).mkdir(parents=True, exist_ok=True)
            return 0

    def fake_load_default_config(mode: str, repo_root: Path) -> dict:
        if mode == "baseline_finetune":
            with open(repo_root / "metapac/configs/baseline_finetune.yaml", "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        if mode == "feature_extract":
            return {"mode": mode}
        if mode == "train_meta":
            return {"mode": mode}
        if mode == "compress":
            return {"mode": mode, "compression": {"meta_checkpoint": str(tmp_path / "meta_ckpt")}}
        return {"mode": mode}

    monkeypatch.setattr("metapac.src.pipeline.create_handler_for_config", lambda cfg: StubHandler())
    monkeypatch.setattr("metapac.src.pipeline._load_default_config", fake_load_default_config)
    monkeypatch.setattr("metapac.src.pipeline.run_feature_extraction", lambda cfg: 0)
    monkeypatch.setattr("metapac.src.pipeline.train_and_eval", lambda cfg: 0)
    monkeypatch.setattr("metapac.src.pipeline.run_compression", lambda cfg: 0)

    exit_code = run({"mode": "auto"})

    assert exit_code == 0
    assert observed["model_name"] == "distilbert-base-uncased"
    assert str(observed["output_dir"]).endswith("targets/distilbert/runs/baseline_distilbert_sst2")