"""
Centralized logging utility for MetaPAC pipeline.

Provides consistent, timestamped logging across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and timestamps."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Add color based on level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format: [HH:MM:SS] [LEVEL] message
        level_colored = f"{color}{record.levelname:8s}{reset}"

        if hasattr(record, 'phase'):
            # Phase-specific formatting: [HH:MM:SS] [phase] message
            return f"[{timestamp}] [{record.phase}] {record.getMessage()}"
        else:
            # Standard formatting: [HH:MM:SS] [LEVEL] message
            return f"[{timestamp}] [{level_colored}] {record.getMessage()}"


_logging_settings: dict = {}
_configured_loggers: dict[str, logging.Logger] = {}


def _resolve_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return int(getattr(logging, level.upper(), logging.INFO))
    return logging.INFO


def _build_handlers(name: str, settings: dict, level: int, default_log_dir: Optional[str]) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    if settings.get("console_enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        handlers.append(console_handler)

    if settings.get("file_enabled", False):
        log_dir = Path(settings.get("dir") or default_log_dir or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = settings.get("filename") or f"{name.split('.')[-1]}.log"
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    return handlers


def setup_logger(
    name: str,
    level: int | str = logging.INFO,
    *,
    settings: Optional[dict] = None,
    default_log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    merged_settings = dict(_logging_settings)
    if settings:
        merged_settings.update(settings)
    resolved_level = _resolve_level(merged_settings.get("level", level))

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers = []

    for handler in _build_handlers(name, merged_settings, resolved_level, default_log_dir):
        logger.addHandler(handler)

    _configured_loggers[name] = logger

    return logger


def configure_logging(settings: Optional[dict] = None, *, default_log_dir: Optional[str] = None) -> None:
    global _logging_settings

    merged = dict(_logging_settings)
    if settings:
        merged.update(settings)
    if default_log_dir and "dir" not in merged:
        merged["dir"] = default_log_dir

    _logging_settings = merged

    for name in list(_configured_loggers):
        setup_logger(name, settings=_logging_settings, default_log_dir=default_log_dir)


def log_phase_header(logger: logging.Logger, phase: str, description: str = ""):
    """Log a phase header with consistent formatting."""
    logger.info("=" * 80)
    if description:
        txt = f"PHASE {phase}: {description}"
        logger.info("||" + txt.center(76) + "||")
    else:
        txt = f"PHASE {phase}"
        logger.info("||" + txt.center(76) + "||")
    logger.info("=" * 80)


def log_section(logger: logging.Logger, title: str):
    """Log a section header."""
    logger.info(title.center(len(title) + 4, ' ').center(60, '-').center(80, ' '))


def log_metric(logger: logging.Logger, name: str, value, unit: str = ""):
    """Log a metric with consistent formatting."""
    if unit:
        logger.info(f" {name}: {value} {unit}")
    else:
        logger.info(f" {name}: {value}")


def log_progress(logger: logging.Logger, current: int, total: int, item: str = "items"):
    """Log progress with percentage."""
    pct = (current / total * 100) if total > 0 else 0
    logger.info(f"Progress: {current}/{total} {item} ({pct:.1f}%)")


# Global logger instance
_default_logger = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create the default logger."""
    global _default_logger
    logger_name = name or 'metapac'
    if logger_name in _configured_loggers:
        return _configured_loggers[logger_name]
    if _default_logger is None or logger_name == 'metapac':
        _default_logger = setup_logger(logger_name, settings=_logging_settings)
        return _default_logger
    return setup_logger(logger_name, settings=_logging_settings)
