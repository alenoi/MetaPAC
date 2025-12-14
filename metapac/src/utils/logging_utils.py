"""
Centralized logging utility for MetaPAC pipeline.

Provides consistent, timestamped logging across all modules.
"""

import logging
import sys
from datetime import datetime
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


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    return logger


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
    if _default_logger is None:
        _default_logger = setup_logger(name or 'metapac')
    return _default_logger
