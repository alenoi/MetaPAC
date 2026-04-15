from __future__ import annotations

import json

import pytest

from metapac.src.compression.phases.export import ExportPhase
from tests._smoke_helpers import build_phase_context, create_model, stub_integrate_variable_bit_export


@pytest.mark.smoke
@pytest.mark.export
def test_export_phase_finalizes_artifacts_and_cleans_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "metapac.src.compression.variable_bit_export.integrate_variable_bit_export",
        stub_integrate_variable_bit_export,
    )

    model = create_model()
    context = build_phase_context(
        tmp_path,
        model=model,
        metadata={
            "pruning": {"enabled": True, "parameters_zeroed": 2},
            "quantization": {
                "enabled": True,
                "num_quantized": 2,
                "parameters": {
                    "encoder.0.bias": {"bits_final": 6, "numel": 8},
                    "encoder.2.bias": {"bits_final": 5, "numel": 8},
                },
            },
            "fine_tuning": {"enabled": True, "success": True},
        },
    )

    compressed_dir = context.output_path / "compressed"
    compressed_dir.mkdir(parents=True, exist_ok=True)
    with open(compressed_dir / "pruning_meta.json", "w", encoding="utf-8") as handle:
        json.dump({"stub": True}, handle)
    with open(compressed_dir / "validation_results.json", "w", encoding="utf-8") as handle:
        json.dump({"accuracy": 0.7}, handle)
    with open(compressed_dir / "orphan.txt", "w", encoding="utf-8") as handle:
        handle.write("orphan artifact")

    phase = ExportPhase(context.config["compression"]["quantization"])
    result = phase.run(context)

    assert result.metadata["export"]["success"] is True
    assert (compressed_dir / "pytorch_model.bin").exists()
    assert (compressed_dir / "variable_bit_meta.json").exists()
    assert (compressed_dir / "manifest.json").exists()
    assert not (compressed_dir / "compression_summary.json").exists()
    assert (context.output_path / "artifacts" / "phase4_export" / "compression_summary.json").exists()
    assert (context.output_path / "artifacts" / "phase2_prune_ft" / "model_state.pt").exists()
    assert (context.output_path / "artifacts" / "phase5_validate" / "validation_results.json").exists()
    assert (context.output_path / "artifacts" / "misc" / "orphan.txt").exists()