from __future__ import annotations

from metapac.src.compression.adapters.registry import auto_detect_architecture
from metapac.src.compression.phases.preparation import _expected_hook_prefixes_for_target_model
from metapac.src.model_profiles import resolve_model_profile_from_name


def test_resolve_model_profile_for_qwen() -> None:
    profile = resolve_model_profile_from_name("Qwen/Qwen3-0.6B")
    assert profile.family == "qwen"
    assert profile.architecture == "qwen"


def test_preparation_hook_prefixes_follow_model_profile() -> None:
    assert _expected_hook_prefixes_for_target_model("targets/distilgpt2/runs/demo") == (
        "transformer.",
        "score.",
        "lm_head.",
    )


def test_auto_detect_architecture_prefers_shared_profile_resolution() -> None:
    metadata = {"base_model": "Qwen/Qwen3-0.6B"}
    assert auto_detect_architecture(metadata) == "qwen"