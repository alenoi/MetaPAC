from __future__ import annotations

import pytest

from metapac.src.compression.pipeline.orchestrator import CompressionPipeline
from tests._smoke_helpers import stub_integrate_variable_bit_export, stub_run_fine_tuning


@pytest.mark.smoke
@pytest.mark.pipeline
def test_pipeline_runs_end_to_end_with_synthetic_assets(smoke_config, monkeypatch) -> None:
    monkeypatch.setattr("metapac.src.compression.fine_tune.run_fine_tuning", stub_run_fine_tuning)
    monkeypatch.setattr(
        "metapac.src.compression.variable_bit_export.integrate_variable_bit_export",
        stub_integrate_variable_bit_export,
    )

    pipeline = CompressionPipeline(smoke_config)
    exit_code = pipeline.run()

    output_dir = smoke_config["output_dir"]
    assert exit_code == 0
    assert (pipeline.config["output_dir"] if isinstance(pipeline.config.get("output_dir"), str) else output_dir)

    from pathlib import Path

    output_path = Path(output_dir)
    assert (output_path / "compressed" / "pytorch_model.bin").exists()
    assert (output_path / "compressed" / "manifest.json").exists()
    assert (output_path / "compressed" / "variable_bit_meta.json").exists()
    assert (output_path / "artifacts" / "phase4_export" / "compression_summary.json").exists()
    assert (output_path / "artifacts" / "phase4_export" / "pruning_meta.json").exists()