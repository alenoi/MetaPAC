"""MetaPAC main pipeline for orchestrating different execution modes.

This module provides the main entry point for running MetaPAC operations:
- feature_extract: Build meta-dataset from model hooks
- train_meta: Train meta-predictor model
- compress: Apply learned compression strategy
- auto: Run pipeline stages sequentially with default configs

The pipeline handles path resolution, configuration defaults, and mode dispatching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

from metapac.src.compression.strategy import run_compression
from metapac.src.feature_extraction.extract import run_feature_extraction
from metapac.src.models.meta_predictor import train_and_eval
from metapac.src.utils.paths import PathRegistry

# Import baseline fine-tuning (lazy import to avoid circular dependencies)
def run_baseline_finetune(config: Dict[str, Any]) -> int:
    """Run baseline model fine-tuning.
    
    Wrapper for baseline fine-tuning that imports and calls the actual training code.
    
    Args:
        config: Configuration dictionary with baseline_finetune settings.
        
    Returns:
        Exit code (0 for success).
    """
    try:
        import os
        import torch
        from metapac.src.models import ModelConfig, build_model
        from targets.distilbert.src.core_utils import setup_logger, set_all_seeds, device_info, save_json
        from targets.distilbert.src.data import DataConfig, load_tokenizer, load_and_prepare_datasets
        from targets.distilbert.src.train import train_and_evaluate
        
        # Extract baseline_finetune config
        finetune_cfg = config.get("baseline_finetune", {})
        
        print("[pipeline] ========================================")
        print("[pipeline] Running baseline fine-tuning")
        print("[pipeline] ========================================")
        
        logger = setup_logger()
        
        # Output directory and experiment name
        exp_name = finetune_cfg.get("experiment_name", "baseline")
        out_dir = finetune_cfg.get("output_dir", f"targets/distilbert/runs/{exp_name}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Seed and device info
        seed = finetune_cfg.get("train", {}).get("seed", 42)
        set_all_seeds(seed)
        devinfo = device_info()
        save_json(os.path.join(out_dir, "device.json"), devinfo)
        save_json(os.path.join(out_dir, "config_resolved.json"), finetune_cfg)
        
        # Auto-detect precision
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16_ok = torch.cuda.is_available()
        
        finetune_cfg.setdefault("train", {})
        finetune_cfg["train"].setdefault("bf16", bool(bf16_ok))
        finetune_cfg["train"].setdefault("fp16", bool(not bf16_ok and fp16_ok))
        
        # Load tokenizer and datasets
        model_name = finetune_cfg["model"]["pretrained_name"]
        tokenizer = load_tokenizer(model_name)
        data_cfg = DataConfig(**finetune_cfg["dataset"])
        datasets, num_labels, label_names = load_and_prepare_datasets(data_cfg, tokenizer)
        
        # Build model
        model_cfg = ModelConfig(
            pretrained_name=model_name,
            dropout=finetune_cfg["model"].get("dropout", 0.1),
            num_labels=num_labels,
            labels=label_names,
        )
        model = build_model(model_cfg)
        
        # Optional: Gradient checkpointing
        if finetune_cfg["train"].get("gradient_checkpointing", False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing ENABLED.")
        
        # Train and evaluate
        summary = train_and_evaluate(
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            out_dir=out_dir,
            train_cfg=finetune_cfg["train"],
            report_to=tuple(finetune_cfg.get("logging", {}).get("report_to", ["tensorboard"])),
            dataloader_num_workers=finetune_cfg["train"].get("dataloader_num_workers", 2),
        )
        
        # Save summary
        save_json(os.path.join(out_dir, "summary_main.json"), summary)
        logger.info("[pipeline] Baseline fine-tuning completed successfully")
        logger.info(f"[pipeline] Results saved to: {out_dir}")
        
        return 0
        
    except ImportError as e:
        print(f"[pipeline] ERROR: Failed to import baseline training modules: {e}")
        print("[pipeline] Make sure targets/distilbert/src/ modules are available")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"[pipeline] ERROR: Baseline fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

# Default configuration files for each mode
DEFAULT_CONFIGS = {
    "baseline_finetune": "metapac/configs/baseline_finetune.yaml",
    "feature_extract": "metapac/configs/feature_extraction.yaml",
    "train_meta": "metapac/configs/meta_distilbert_sst2.yaml",
    "compress": "metapac/configs/compress_distilbert_sst2.yaml",
}

# Pipeline stage ordering (baseline fine-tuning added at the beginning)
PIPELINE_STAGES = ["baseline_finetune", "feature_extract", "train_meta", "compress"]


def _resolve_repo_root() -> Path:
    """Resolve repository root directory.
    
    Walks up the directory tree from the current file to find the
    repository root (parent of 'metapac' directory).
    
    Returns:
        Path to repository root directory.
    """
    current_path = Path(__file__).resolve()

    # Check current path and all parents
    for path in [current_path] + list(current_path.parents):
        metapac_dir = path / "metapac"

        if metapac_dir.exists() and metapac_dir.is_dir():
            # If we're inside metapac/src, repo root is metapac's parent
            if path.name == "src" and path.parent.name == "metapac":
                return path.parent.parent
            # If we found the metapac directory, return its parent
            if path.name == "metapac":
                return path.parent

    # Fallback to current working directory
    return Path.cwd()


def _load_default_config(mode: str, repo_root: Path) -> Dict[str, Any]:
    """Load default configuration for a given mode.
    
    Args:
        mode: Pipeline mode name.
        repo_root: Repository root directory.
        
    Returns:
        Configuration dictionary loaded from default config file.
        
    Raises:
        FileNotFoundError: If default config file doesn't exist.
    """
    if mode not in DEFAULT_CONFIGS:
        raise ValueError(f"No default config defined for mode: {mode}")

    config_path = repo_root / DEFAULT_CONFIGS[mode]

    if not config_path.exists():
        raise FileNotFoundError(f"Default config not found: {config_path}")

    print(f"[pipeline] Loading default config: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _run_auto_pipeline(start_mode: str, config: Dict[str, Any]) -> int:
    """Run pipeline stages automatically from start_mode onwards.
    
    Executes pipeline stages sequentially starting from start_mode:
    - feature_extract → train_meta → compress
    
    Each stage uses its default configuration unless overridden in config.
    
    Args:
        start_mode: Starting mode (e.g., 'train_meta' skips feature_extract).
        config: Base configuration dictionary (can be empty).
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    print("[pipeline] ========================================")
    print(f"[pipeline] AUTO MODE: Running pipeline from '{start_mode}'")
    print("[pipeline] ========================================")

    # Determine which stages to run
    if start_mode not in PIPELINE_STAGES:
        raise ValueError(
            f"Invalid start mode: '{start_mode}'. "
            f"Valid stages: {PIPELINE_STAGES}"
        )

    start_idx = PIPELINE_STAGES.index(start_mode)
    stages_to_run = PIPELINE_STAGES[start_idx:]

    print(f"[pipeline] Stages to run: {' -> '.join(stages_to_run)}")
    print()

    # Run each stage sequentially
    repo_root = _resolve_repo_root()

    for stage in stages_to_run:
        print("[pipeline] ========================================")
        print(f"[pipeline] Stage: {stage}")
        print("[pipeline] ========================================")

        # Load default config for this stage
        stage_config = _load_default_config(stage, repo_root)

        # Merge with user-provided config (user config takes precedence)
        stage_config.update(config)
        stage_config["mode"] = stage

        # Run the stage
        exit_code = run(stage_config)

        if exit_code != 0:
            print(f"[pipeline] ERROR: Stage '{stage}' failed with exit code {exit_code}")
            return exit_code

        print(f"[pipeline] Stage '{stage}' completed successfully")
        print()

    print("[pipeline] ========================================")
    print("[pipeline] AUTO MODE: All stages completed successfully!")
    print("[pipeline] ========================================")

    return 0


def run(config: Dict[str, Any]) -> int:
    """Run MetaPAC pipeline in specified mode.
    
    Main pipeline orchestrator that:
    1. Resolves repository structure and paths
    2. Sets configuration defaults
    3. Dispatches to appropriate mode handler
    
    Supported modes:
    - baseline_finetune: Fine-tune baseline model on target dataset
    - feature_extract: Extract features from model and build meta-dataset
    - train_meta: Train meta-predictor on meta-dataset
    - compress: Apply compression with meta-predictor guidance
    - auto: Run all stages (baseline_finetune → feature_extract → train_meta → compress) with defaults
    - auto:MODE: Run from MODE onwards (e.g., auto:feature_extract skips baseline fine-tuning)
    - noop/none: No-op mode for testing
    
    Auto mode behavior:
    - Each stage uses its default config file from DEFAULT_CONFIGS
    - User-provided config overrides defaults for all stages
    - Pipeline stops on first stage failure
    
    Args:
        config: Configuration dictionary with 'mode' key and mode-specific settings.
        
    Returns:
        Exit code (0 for success).
        
    Raises:
        ValueError: If mode is unknown.
        
    Examples:
        >>> # Run full pipeline with defaults
        >>> run({"mode": "auto"})
        
        >>> # Run from training onwards
        >>> run({"mode": "auto:train_meta"})
        
        >>> # Run compression only
        >>> run({"mode": "compress", "compression": {...}})
    """
    # Extract and normalize mode
    mode = str(config.get("mode", "noop")).lower()

    # Resolve paths
    repo_root = _resolve_repo_root()
    paths = PathRegistry(repo_root=repo_root)
    paths.ensure_dirs()

    # Set configuration defaults for consistent path handling
    config.setdefault("meta_dataset_path", str(paths.meta_dataset_path))
    config.setdefault("outputs", {})
    config["outputs"].setdefault("runs_dir", str(paths.runs_dir))
    config["outputs"].setdefault("results_dir", str(paths.results_dir))
    config["outputs"].setdefault("meta_dataset_path", str(paths.meta_dataset_path))

    # Print pipeline information
    print(f"[pipeline] Mode: {mode}")
    print(f"[pipeline] Repo root: {repo_root}")
    print(f"[pipeline] Meta-dataset path: {config['meta_dataset_path']}")

    # Handle auto mode
    if mode == "auto":
        # Auto mode runs all stages with defaults (starting from baseline fine-tuning)
        return _run_auto_pipeline("baseline_finetune", config)

    # Handle auto mode starting from specific stage
    if mode.startswith("auto:"):
        start_stage = mode.split(":", 1)[1]
        return _run_auto_pipeline(start_stage, config)

    # Dispatch to appropriate mode handler
    if mode == "baseline_finetune":
        return run_baseline_finetune(config)
    if mode == "feature_extract":
        return run_feature_extraction(config)
    if mode == "train_meta":
        return train_and_eval(config)
    if mode == "compress":
        return run_compression(config)
    if mode in {"noop", "none"}:
        print("[pipeline] No-op mode; nothing to do.")
        return 0

    raise ValueError(
        f"Unknown pipeline mode: '{mode}'. "
        f"Valid modes: 'baseline_finetune', 'feature_extract', 'train_meta', 'compress', 'auto', 'auto:MODE', 'noop'"
    )
