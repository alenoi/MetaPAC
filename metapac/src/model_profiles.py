from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelProfile:
    family: str
    architecture: str
    supported_prefixes: tuple[str, ...]
    expected_hook_prefixes: tuple[str, ...]
    module_markers: tuple[str, ...]

    def matches_reference(self, value: str) -> bool:
        normalized = normalize_model_reference(value)
        path_parts = tuple(part for part in normalized.split("/") if part)
        return any(
            normalized.startswith(prefix)
            or any(part.startswith(prefix) for part in path_parts)
            for prefix in self.supported_prefixes
        )


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        family="distilbert",
        architecture="distilbert",
        supported_prefixes=("distilbert",),
        expected_hook_prefixes=("distilbert.", "classifier.", "pre_classifier."),
        module_markers=("distilbert",),
    ),
    ModelProfile(
        family="bert",
        architecture="bert",
        supported_prefixes=("bert",),
        expected_hook_prefixes=("bert.", "classifier.", "pooler."),
        module_markers=("bert",),
    ),
    ModelProfile(
        family="roberta",
        architecture="roberta",
        supported_prefixes=("roberta",),
        expected_hook_prefixes=("roberta.", "classifier."),
        module_markers=("roberta",),
    ),
    ModelProfile(
        family="gpt2",
        architecture="gpt2",
        supported_prefixes=("distilgpt2", "gpt2"),
        expected_hook_prefixes=("transformer.", "score.", "lm_head."),
        module_markers=("transformer",),
    ),
    ModelProfile(
        family="qwen",
        architecture="qwen",
        supported_prefixes=("qwen/", "qwen2", "qwen2.5", "qwen3"),
        expected_hook_prefixes=("model.", "score.", "lm_head.", "classifier."),
        module_markers=("model",),
    ),
)

DEFAULT_MODEL_PROFILE = ModelProfile(
    family="generic",
    architecture="generic",
    supported_prefixes=(),
    expected_hook_prefixes=("transformer.", "distilbert.", "model.", "bert.", "roberta."),
    module_markers=(),
)


def normalize_model_reference(value: str | Path | None) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\\", "/").lower()


def iter_model_profiles() -> Iterable[ModelProfile]:
    return MODEL_PROFILES


def resolve_model_profile_from_name(value: str | Path | None) -> ModelProfile:
    normalized = normalize_model_reference(value)
    for profile in MODEL_PROFILES:
        if profile.matches_reference(normalized):
            return profile
    return DEFAULT_MODEL_PROFILE


def resolve_model_profile_from_model(model: object) -> ModelProfile:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type.strip():
        return resolve_model_profile_from_name(model_type)

    for profile in MODEL_PROFILES:
        if any(hasattr(model, marker) for marker in profile.module_markers):
            return profile
    return DEFAULT_MODEL_PROFILE


def resolve_architecture_name(value: str | Path | None) -> str:
    return resolve_model_profile_from_name(value).architecture