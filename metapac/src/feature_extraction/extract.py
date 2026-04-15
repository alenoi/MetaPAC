# metapac/src/feature_extraction/extract.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Any

import yaml

from .builder import BuildConfig, build_meta_dataset


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_feature_extraction(cfg: Dict[str, Any]) -> int:
    out_path = Path(
        cfg.get("outputs", {}).get("meta_dataset_path", "metapac/artifacts/meta_dataset/meta_dataset.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config into BuildConfig.
    build_cfg = BuildConfig(
        reducer=cfg.get("reducer", "mean_pool"),
        token_average=cfg.get("token_average", True),
        phases=cfg.get("phases"),
        write_parquet=True,
        write_csv=True,
        max_files=cfg.get("max_files"),
        chunksize=cfg.get("chunksize"),
        run_id=cfg.get("run_id"),
        input_dir=cfg.get("input_dir", "artifacts"),
        hook_pattern=cfg.get("hook_pattern", "hook_stats_*.csv"),
        # NaN handling config
        min_valid_ratio=cfg.get("min_valid_ratio", 0.5),
        fill_value=cfg.get("fill_value", 0.0),
        min_samples_threshold=cfg.get("min_samples_threshold", 10),
        imputation_strategy=cfg.get("imputation_strategy", "zero")
    )

    # Run feature extraction.
    try:
        print(f"[debug] Using configuration:")
        print(f"  Input dir: {build_cfg.input_dir}")
        print(f"  Output dir: {str(out_path.parent)}")
        print(f"  NaN config: min_valid_ratio={build_cfg.min_valid_ratio}, imputation={build_cfg.imputation_strategy}")

        path = build_meta_dataset(build_cfg.input_dir, str(out_path.parent), build_cfg)

        built_path = Path(path)
        if built_path.resolve() != out_path.resolve():
            shutil.copy2(built_path, out_path)
            print(f"[feature_extraction] Copied meta-dataset to configured path: {out_path}")

        print(f"[feature_extraction] Successfully wrote meta-dataset to: {path}")
        return 0
    except Exception as e:
        import traceback
        print(f"[feature_extraction][error] Failed to build meta-dataset: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print(f"[debug] Loading config from {args.config}")
    config = load_yaml_config(args.config)

    if args.verbose:
        print(f"[debug] Loaded config: {config}")

    exit(run_feature_extraction(config))


def run_feature_extraction(cfg: Dict[str, Any]) -> int:
    out_path = Path(
        cfg.get("outputs", {}).get("meta_dataset_path", "metapac/artifacts/meta_dataset/meta_dataset.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config into BuildConfig.
    build_cfg = BuildConfig(
        reducer=cfg.get("reducer", "mean_pool"),
        token_average=cfg.get("token_average", True),
        phases=cfg.get("phases"),
        write_parquet=True,
        write_csv=True,
        max_files=cfg.get("max_files"),
        chunksize=cfg.get("chunksize"),
        run_id=cfg.get("run_id"),
        input_dir=cfg.get("input_dir", "artifacts"),
        hook_pattern=cfg.get("hook_pattern", "hook_stats_*.csv"),
        # NaN handling config
        min_valid_ratio=cfg.get("min_valid_ratio", 0.5),
        fill_value=cfg.get("fill_value", 0.0),
        min_samples_threshold=cfg.get("min_samples_threshold", 10),
        imputation_strategy=cfg.get("imputation_strategy", "zero")
    )

    # Run feature extraction.
    try:
        print(f"[debug] Using configuration:")
        print(f"  Input dir: {build_cfg.input_dir}")
        print(f"  Output dir: {str(out_path.parent)}")
        print(f"  NaN config: min_valid_ratio={build_cfg.min_valid_ratio}, imputation={build_cfg.imputation_strategy}")

        path = build_meta_dataset(build_cfg.input_dir, str(out_path.parent), build_cfg)

        built_path = Path(path)
        if built_path.resolve() != out_path.resolve():
            shutil.copy2(built_path, out_path)
            print(f"[feature_extraction] Copied meta-dataset to configured path: {out_path}")

        print(f"[feature_extraction] Successfully wrote meta-dataset to: {path}")
        return 0
    except Exception as e:
        import traceback
        print(f"[feature_extraction][error] Failed to build meta-dataset: {str(e)}")
        print(traceback.format_exc())
        return 1
