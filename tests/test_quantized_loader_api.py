from __future__ import annotations

import metapac.src.compression.load_quantized_model as quantized_loader


def test_legacy_quantized_loader_alias_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def stub_load_quantized_model(model_dir: str, device: str = "auto", config_path: str | None = None):
        captured["model_dir"] = model_dir
        captured["device"] = device
        captured["config_path"] = config_path
        return "stub-model"

    monkeypatch.setattr(quantized_loader, "load_quantized_model", stub_load_quantized_model)

    result = quantized_loader.load_quantized_distilbert("demo-model", device="cpu", config_path="cfg-dir")

    assert result == "stub-model"
    assert captured == {
        "model_dir": "demo-model",
        "device": "cpu",
        "config_path": "cfg-dir",
    }