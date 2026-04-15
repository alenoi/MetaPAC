from __future__ import annotations

import pytest
import torch

from metapac.src.compression.phases.quantization_phase import QuantizationPhase
from tests._smoke_helpers import build_phase_context, create_model


@pytest.mark.smoke
@pytest.mark.quantization
def test_quantization_phase_updates_quantize_zone_and_metadata(tmp_path) -> None:
    model = create_model()
    context = build_phase_context(tmp_path, model=model)
    phase = QuantizationPhase(context.config["compression"]["quantization"])

    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    result = phase.run(context)

    quantized_names = [name for name, action in context.plan.items() if action == "quantize"]
    assert quantized_names
    assert result.metadata["quantization"]["enabled"] is True
    assert result.metadata["quantization"]["num_quantized"] == len(quantized_names)
    assert any(not torch.equal(before[name], model.state_dict()[name]) for name in quantized_names)