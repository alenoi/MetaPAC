from __future__ import annotations

from typing import Any, Type

from .base import ModelHandler


class ModelHandlerRegistry:
    """Registry for pipeline model handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Type[ModelHandler]] = {}

    def register(self, handler_class: Type[ModelHandler]) -> Type[ModelHandler]:
        name = handler_class.handler_name
        if name in self._handlers:
            raise ValueError(f"Model handler '{name}' is already registered")
        self._handlers[name] = handler_class
        return handler_class

    def list_handlers(self) -> dict[str, Type[ModelHandler]]:
        return dict(self._handlers)

    def create_for_config(self, config: dict[str, Any]) -> ModelHandler:
        model_name = (
            config.get("baseline_finetune", {})
            .get("model", {})
            .get("pretrained_name", "")
        )
        candidates = [
            handler_class
            for handler_class in self._handlers.values()
            if handler_class.can_handle_config(config)
        ]
        if not candidates:
            available = ", ".join(sorted(self._handlers)) or "none"
            raise KeyError(
                f"No model handler registered for '{model_name}'. Available handlers: {available}"
            )
        winner = sorted(candidates, key=lambda handler_class: handler_class.priority, reverse=True)[0]
        return winner()


_registry = ModelHandlerRegistry()


def register_model_handler(handler_class: Type[ModelHandler]) -> Type[ModelHandler]:
    return _registry.register(handler_class)


def create_handler_for_config(config: dict[str, Any]) -> ModelHandler:
    return _registry.create_for_config(config)


def list_model_handlers() -> dict[str, Type[ModelHandler]]:
    return _registry.list_handlers()