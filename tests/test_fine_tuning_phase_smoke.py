from __future__ import annotations

import pytest
import torch

from metapac.src.compression.phases.fine_tuning import FineTuningPhase
from tests._smoke_helpers import build_phase_context, create_model, stub_run_fine_tuning


@pytest.mark.smoke
@pytest.mark.finetuning
def test_fine_tuning_phase_creates_recovery_artifacts_and_loads_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("metapac.src.compression.fine_tune.run_fine_tuning", stub_run_fine_tuning)

    model = create_model()
    context = build_phase_context(
        tmp_path,
        model=model,
        metadata={
            "quantization": {
                "enabled": True,
                "num_quantized": 2,
                "parameters": {"classifier.weight": {"bits_final": 6}},
            }
        },
    )
    phase = FineTuningPhase(context.config["compression"]["fine_tuning"])

    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    result = phase.run(context)

    assert (result.output_path / "quantized_before_ft" / "pytorch_model.bin").exists()
    assert (result.output_path / "quantized_before_ft" / "config.json").exists()
    assert (result.output_path / "quantized_before_ft" / "tokenizer.json").exists()
    assert result.metadata["fine_tuning"]["success"] is True
    assert result.metadata["fine_tuning"]["metrics"]["best_val_accuracy"] == 0.7
    assert any(not torch.equal(before[name], model.state_dict()[name]) for name in before)