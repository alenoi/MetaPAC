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
from metapac.src.model_handlers import create_handler_for_config
from metapac.src.models.meta_predictor import train_and_eval
from metapac.src.utils.logging_utils import configure_logging
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
        handler = create_handler_for_config(config)
        return handler.run_baseline_finetune(config)
    except ImportError as e:
        print(f"[pipeline] ERROR: Failed to import baseline training modules: {e}")
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


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries without discarding nested defaults."""
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _run_slug(config: Dict[str, Any]) -> str:
    candidates = [
        config.get("run_id"),
        config.get("run_tag"),
        config.get("experiment_name"),
        config.get("baseline_finetune", {}).get("experiment_name"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().replace("/", "_")
    return "default_run"


def _is_shared_hook_dir(path_value: str | None) -> bool:
    if not path_value:
        return True
    normalized = str(path_value).replace("\\", "/").rstrip("/")
    return normalized in {
        "artifacts",
        "artifacts/raw",
        "metapac/artifacts/raw",
    }


def _should_rewrite_meta_dataset_path(path_value: str | None, repo_root: Path) -> bool:
    if not path_value:
        return True

    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path

    meta_root = repo_root / "metapac" / "artifacts" / "meta_dataset"
    try:
        rel = path.resolve().relative_to(meta_root.resolve())
    except Exception:
        return False

    return len(rel.parts) <= 1


def _infer_dataset_storage_mode(dataset_cfg: Dict[str, Any]) -> str:
    split_strategy = str(dataset_cfg.get("split_strategy", "default") or "default").lower()
    if split_strategy != "default":
        return "split"
    if dataset_cfg.get("val_split_ratio") is not None or dataset_cfg.get("test_split_ratio") is not None:
        return "split"
    if bool(dataset_cfg.get("deduplicate_by_text", False)) or bool(dataset_cfg.get("enforce_no_text_overlap", False)):
        return "split"
    return "raw"


def _copy_dataset_processing_defaults(source_cfg: Dict[str, Any], target_cfg: Dict[str, Any]) -> None:
    for key in (
        "split_strategy",
        "val_split_ratio",
        "test_split_ratio",
        "seed",
        "deduplicate_by_text",
        "enforce_no_text_overlap",
    ):
        if target_cfg.get(key) is None and source_cfg.get(key) is not None:
            target_cfg[key] = source_cfg[key]


def _ensure_dataset_storage(config: Dict[str, Any], repo_root: Path) -> None:
    dataset_root = _to_repo_relative(repo_root / "metapac" / "artifacts" / "datasets", repo_root)

    baseline_dataset_cfg = config.setdefault("baseline_finetune", {}).setdefault("dataset", {})
    baseline_source_cfg = baseline_dataset_cfg.setdefault("source", {})
    if isinstance(baseline_source_cfg, dict):
        baseline_storage = baseline_source_cfg.setdefault("storage", {})
        baseline_storage.setdefault("root", dataset_root)
        baseline_storage.setdefault("mode", _infer_dataset_storage_mode(baseline_dataset_cfg))

    compression_cfg = config.setdefault("compression", {})

    fine_tuning_cfg = compression_cfg.setdefault("fine_tuning", {})
    fine_tune_data_cfg = fine_tuning_cfg.setdefault("data", {})
    _copy_dataset_processing_defaults(baseline_dataset_cfg, fine_tune_data_cfg)
    ft_source_cfg = fine_tune_data_cfg.setdefault("source", {})
    if isinstance(ft_source_cfg, dict):
        ft_storage = ft_source_cfg.setdefault("storage", {})
        ft_storage.setdefault("root", dataset_root)
        ft_storage.setdefault(
            "mode",
            baseline_source_cfg.get("storage", {}).get("mode", _infer_dataset_storage_mode(fine_tune_data_cfg))
            if isinstance(baseline_source_cfg, dict) else _infer_dataset_storage_mode(fine_tune_data_cfg),
        )

    validation_cfg = compression_cfg.setdefault("validation", {})
    _copy_dataset_processing_defaults(baseline_dataset_cfg, validation_cfg)
    validation_source_cfg = validation_cfg.setdefault("source", {})
    if isinstance(validation_source_cfg, dict):
        validation_storage = validation_source_cfg.setdefault("storage", {})
        validation_storage.setdefault("root", dataset_root)
        validation_storage.setdefault(
            "mode",
            baseline_source_cfg.get("storage", {}).get("mode", _infer_dataset_storage_mode(validation_cfg))
            if isinstance(baseline_source_cfg, dict) else _infer_dataset_storage_mode(validation_cfg),
        )


def _apply_run_artifact_paths(config: Dict[str, Any], repo_root: Path) -> None:
    run_slug = _run_slug(config)

    baseline_cfg = config.setdefault("baseline_finetune", {})
    train_cfg = baseline_cfg.setdefault("train", {})
    baseline_output_dir = baseline_cfg.get("output_dir")

    if baseline_output_dir:
        hook_dir = Path(baseline_output_dir) / "artifacts" / "raw"
    else:
        hook_dir = repo_root / "metapac" / "artifacts" / "raw" / run_slug

    hook_dir_value = _to_repo_relative(hook_dir, repo_root)
    if _is_shared_hook_dir(train_cfg.get("hook_output_dir")):
        train_cfg["hook_output_dir"] = hook_dir_value

    if _is_shared_hook_dir(config.get("input_dir")):
        config["input_dir"] = hook_dir_value

    meta_dataset_path = repo_root / "metapac" / "artifacts" / "meta_dataset" / run_slug / "meta_dataset.parquet"
    meta_dataset_value = _to_repo_relative(meta_dataset_path, repo_root)

    if _should_rewrite_meta_dataset_path(config.get("meta_dataset_path"), repo_root):
        config["meta_dataset_path"] = meta_dataset_value

    outputs_cfg = config.setdefault("outputs", {})
    if _should_rewrite_meta_dataset_path(outputs_cfg.get("meta_dataset_path"), repo_root):
        outputs_cfg["meta_dataset_path"] = meta_dataset_value

    data_cfg = config.setdefault("data", {})
    if _should_rewrite_meta_dataset_path(data_cfg.get("path"), repo_root):
        data_cfg["path"] = meta_dataset_value

    _ensure_dataset_storage(config, repo_root)

    hook_dir.mkdir(parents=True, exist_ok=True)
    meta_dataset_path.parent.mkdir(parents=True, exist_ok=True)


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

    def _latest_meta_checkpoint_dir() -> str | None:
        """Find latest portable meta checkpoint directory for current run_tag/experiment_name."""
        outputs_cfg = config.get("outputs", {})
        runs_dir = Path(outputs_cfg.get("runs_dir", repo_root / "metapac" / "runs"))
        ckpt_root = runs_dir / "checkpoints"
        if not ckpt_root.exists():
            return None

        prefixes = []
        run_tag = config.get("run_tag")
        exp_name = config.get("experiment_name")
        if isinstance(run_tag, str) and run_tag:
            prefixes.append(run_tag)
        if isinstance(exp_name, str) and exp_name:
            prefixes.append(exp_name)
        prefixes.append("metapac_meta")

        candidates = []
        for d in ckpt_root.iterdir():
            if not d.is_dir():
                continue
            if not (d / "model_state.pt").exists() or not (d / "feature_names.json").exists():
                continue
            name = d.name
            if any(name.startswith(pfx) for pfx in prefixes):
                candidates.append(d)

        if not candidates:
            return None

        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)

    for stage in stages_to_run:
        print("[pipeline] ========================================")
        print(f"[pipeline] Stage: {stage}")
        print("[pipeline] ========================================")

        # Load default config for this stage
        stage_config = _load_default_config(stage, repo_root)

        # Merge with user-provided config while preserving nested stage defaults.
        stage_config = _deep_merge_dicts(stage_config, config)
        stage_config["mode"] = stage

        # Run the stage
        exit_code = run(stage_config)

        if exit_code != 0:
            print(f"[pipeline] ERROR: Stage '{stage}' failed with exit code {exit_code}")
            return exit_code

        # Propagate freshly trained meta-checkpoint to downstream compression stage.
        if stage == "train_meta":
            latest_ckpt = _latest_meta_checkpoint_dir()
            if latest_ckpt:
                config.setdefault("compression", {})
                config["compression"]["meta_checkpoint"] = latest_ckpt
                print(f"[pipeline] Auto-selected latest meta checkpoint for compress: {latest_ckpt}")

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
    _apply_run_artifact_paths(config, repo_root)

    logging_cfg = config.get("logging", {})
    default_log_dir = None
    if mode == "baseline_finetune":
        default_log_dir = str(Path(config.get("baseline_finetune", {}).get("output_dir", "logs")) / "logs")
    elif mode in {"compress", "auto"} or mode.startswith("auto:"):
        compression_output = config.get("compression", {}).get("output_dir") or config.get("output_dir")
        if compression_output:
            default_log_dir = str(Path(compression_output) / "logs")
    configure_logging(logging_cfg, default_log_dir=default_log_dir)

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
