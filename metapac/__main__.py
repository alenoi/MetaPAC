"""MetaPAC command-line interface entry point.

Usage:
    python -m metapac --config path/to/config.yaml
    python -m metapac --mode auto
    python -m metapac --mode compress --config path/to/config.yaml
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
        description="MetaPAC: Meta-learning based Predictive Adaptive Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults (including baseline fine-tuning)
  python -m metapac --mode auto
  
  # Run from feature extraction onwards (skip baseline fine-tuning)
  python -m metapac --mode auto:feature_extract
  
  # Run baseline fine-tuning only
  python -m metapac --mode baseline_finetune --config metapac/configs/baseline_finetune.yaml
  
  # Run specific mode with config
  python -m metapac --mode compress --config metapac/configs/compress_distilbert_sst2.yaml
  
    # Use config file (mode specified in config)
    python -m metapac --config metapac/configs/auto_distilbert_sst2_fast.yaml
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Pipeline mode (baseline_finetune, feature_extract, train_meta, compress, auto, auto:MODE)"
    )

    args = parser.parse_args()

    # Load config from file if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: Configuration file not found: {config_path}", file=sys.stderr)
            return 1

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Override mode from command line if provided
    if args.mode:
        config["mode"] = args.mode

    # Check if mode is specified
    if "mode" not in config:
        parser.print_help()
        print("\nERROR: No mode specified. Use --mode or provide a config file with 'mode' key.", file=sys.stderr)
        return 1

    # Run pipeline
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
