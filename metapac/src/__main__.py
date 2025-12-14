"""MetaPAC command-line interface entry point.

Usage:
    python -m metapac.src --config path/to/config.yaml
    python -m metapac.src.pipeline --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from metapac.src.pipeline import run


def main() -> int:
    """Parse arguments and run pipeline.
    
    Returns:
        Exit code from pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="MetaPAC: Meta-learning based Predictive Adaptive Compression"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Run pipeline
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
