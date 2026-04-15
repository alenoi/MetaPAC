"""Configuration management for compression pipeline.

This module handles loading and merging configuration:
- Load default configuration from strategy_defaults.yaml
- Merge user config with defaults (user values take precedence)
- Deep merge nested configuration dictionaries
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


def load_strategy_defaults() -> Dict[str, Any]:
    """Load default configuration from strategy_defaults.yaml.

    Returns:
        Dictionary with default configuration values.
        Returns empty dict if file not found (graceful fallback).
    """
    try:
        # Resolve config path relative to this module
        current_file = Path(__file__).resolve()
        # Go up: utils -> compression -> src -> .. -> configs
        config_path = current_file.parent.parent.parent.parent / "configs" / "strategy_defaults.yaml"

        if not config_path.exists():
            logger.warning(f"Strategy defaults config not found: {config_path}")
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f)

        logger.debug(f"Loaded strategy defaults from: {config_path}")
        return defaults or {}

    except Exception as e:
        logger.warning(f"Failed to load strategy defaults: {e}")
        return {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Args:
        base: Base dictionary (lower priority)
        override: Override dictionary (higher priority)

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_with_defaults(
    user_cfg: Dict[str, Any],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge user config with defaults (user config takes precedence).

    Args:
        user_cfg: User-provided configuration
        defaults: Default configuration values

    Returns:
        Merged configuration dictionary with user values taking precedence.
    """
    return deep_merge(defaults, user_cfg)
