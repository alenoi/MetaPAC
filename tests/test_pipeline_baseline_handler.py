from __future__ import annotations

from metapac.src.pipeline import run_baseline_finetune


def test_pipeline_delegates_baseline_to_model_handler(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubHandler:
        def run_baseline_finetune(self, config: dict) -> int:
            captured["config"] = config
            return 7

    config = {
        "baseline_finetune": {
            "model": {
                "pretrained_name": "distilbert-base-uncased",
            }
        }
    }

    monkeypatch.setattr("metapac.src.pipeline.create_handler_for_config", lambda cfg: StubHandler())

    exit_code = run_baseline_finetune(config)

    assert exit_code == 7
    assert captured["config"] is config