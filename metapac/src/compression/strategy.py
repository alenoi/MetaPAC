"""Compression strategy - main entry point for compression pipeline.

This module provides a convenient entry point that delegates to the modular
phase-based compression pipeline architecture.

Architecture:
    pipeline/
        orchestrator.py       - Main pipeline coordinator
        phase_base.py         - Abstract base classes (CompressionPhase, PhaseContext)
        config_manager.py     - Configuration management
    
    phases/
        preparation.py        - Feature extraction & importance scoring
        pruning_phase.py      - Structured/unstructured pruning
        quantization_phase.py - Rank-aware quantization
        fine_tuning.py        - Post-compression recovery training
        export.py             - Model export (HF, variable-bit, packed)
    
    utils/
        checkpoint.py         - Checkpoint selection & resolution
        model_loading.py      - Model loading utilities
        registry.py           - Quantization layer registry

Usage:
    From code:
        from metapac.src.compression.strategy import run_compression
        exit_code = run_compression(config_dict)
    
    From CLI:
        python -m metapac.src.compression.strategy --config path/to/config.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

# Import main entry point from orchestrator
from .pipeline.orchestrator import run_compression as _run_compression
from ..utils.logging_utils import configure_logging


def run_compression(cfg: Dict[str, Any]) -> int:
    """Main compression pipeline entry point.
    
    Args:
        cfg: Full compression configuration dictionary
    
    Returns:
        Exit code (0 = success, 1 = failure)
    
    Pipeline Phases:
        1. Preparation: Load meta-predictor, compute importance scores
        2. Pruning: Remove unimportant structures (optional)
        3. Quantization: Reduce bit-width based on importance
        4. Fine-Tuning: Recover accuracy after compression (optional)
        5. Export: Save compressed model in multiple formats
        6. Validation: Evaluate compressed vs baseline model (optional)
    """
    output_dir = cfg.get("compression", {}).get("output_dir") or cfg.get("output_dir")
    default_log_dir = str(Path(output_dir) / "logs") if output_dir else None
    configure_logging(cfg.get("logging", {}), default_log_dir=default_log_dir)
    return _run_compression(cfg)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="MetaPAC Compression Pipeline - Phase-Based Architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory (optional)"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        help="Override target model path (optional)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Apply CLI overrides
    if args.output_dir:
        if 'compression' not in config:
            config['compression'] = {}
        config['compression']['output_dir'] = args.output_dir
    
    if args.target_model:
        if 'compression' not in config:
            config['compression'] = {}
        config['compression']['target_model'] = args.target_model
    
    # Run compression
    exit_code = run_compression(config)
    sys.exit(exit_code)
