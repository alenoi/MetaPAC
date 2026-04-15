from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from metapac.src.model_profiles import resolve_model_profile_from_name


class ModelHandler(ABC):
    """Base class for model-specific pipeline handlers."""

    handler_name: ClassVar[str] = "base"
    model_family: ClassVar[str | None] = None
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ()
    priority: ClassVar[int] = 0

    @classmethod
    def can_handle_model_name(cls, model_name: str) -> bool:
        normalized = str(model_name or "").strip().lower()
        if not normalized:
            return False
        if cls.model_family:
            return resolve_model_profile_from_name(normalized).family == cls.model_family
        if not cls.supported_model_prefixes:
            return True
        return any(normalized.startswith(prefix.lower()) for prefix in cls.supported_model_prefixes)

    @classmethod
    def can_handle_config(cls, config: dict[str, Any]) -> bool:
        model_name = (
            config.get("baseline_finetune", {})
            .get("model", {})
            .get("pretrained_name", "")
        )
        return cls.can_handle_model_name(model_name)

    @abstractmethod
    def run_baseline_finetune(self, config: dict[str, Any]) -> int:
        """Execute baseline fine-tuning for a model family."""