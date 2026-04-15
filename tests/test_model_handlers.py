from __future__ import annotations

from metapac.src.model_handlers import create_handler_for_config


def _config_for(model_name: str) -> dict:
    return {
        "baseline_finetune": {
            "model": {
                "pretrained_name": model_name,
            }
        }
    }


def test_selects_distilbert_handler() -> None:
    handler = create_handler_for_config(_config_for("distilbert-base-uncased"))
    assert handler.handler_name == "distilbert_sequence_classifier"


def test_selects_gpt2_handler() -> None:
    handler = create_handler_for_config(_config_for("distilgpt2"))
    assert handler.handler_name == "gpt2_sequence_classifier"


def test_selects_qwen_handler() -> None:
    handler = create_handler_for_config(_config_for("Qwen/Qwen3-0.6B"))
    assert handler.handler_name == "qwen_sequence_classifier"


def test_falls_back_to_auto_handler() -> None:
    handler = create_handler_for_config(_config_for("my-org/custom-sequence-model"))
    assert handler.handler_name == "auto_sequence_classifier"