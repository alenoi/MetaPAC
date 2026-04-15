from __future__ import annotations

import json

import pytest
import torch

from metapac.src.compression.phases.pruning_phase import PruningPhase
from tests._smoke_helpers import build_phase_context, create_model


@pytest.mark.smoke
@pytest.mark.pruning
def test_pruning_phase_applies_fallback_zero_pruning(tmp_path) -> None:
    model = create_model()
    context = build_phase_context(tmp_path, model=model)
    phase = PruningPhase(context.config["compression"]["pruning"])

    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    result = phase.run(context)

    pruned_names = [name for name, action in context.plan.items() if action == "prune"]
    assert pruned_names
    assert any(torch.count_nonzero(model.state_dict()[name] == 0).item() > 0 for name in pruned_names)
    assert (result.output_path / "compressed" / "pruning_meta.json").exists()

    with open(result.output_path / "compressed" / "pruning_meta.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["parameter_zero_pruning"] is True
    assert metadata["parameters_zeroed"] >= 1
    assert any(not torch.equal(before[name], model.state_dict()[name]) for name in pruned_names)