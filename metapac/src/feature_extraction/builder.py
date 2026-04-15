"""Meta-dataset builder for feature extraction from model hooks.

This module provides functionality to build meta-datasets from captured model
activation and gradient statistics. It aggregates hook data across training steps
and epochs, computing comprehensive statistics for meta-learning.
"""

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import yaml

from .io import load_hook_csvs
from .reducers import apply_reducer
from ..utils.analysis import compute_stats


@dataclass
class BuildConfig:
    """Configuration for meta-dataset construction.
    
    Attributes:
        reducer: Method for reducing tensor dimensions ('mean_pool', 'max_pool', 'cls_pool').
        token_average: Whether to average across token dimensions.
        phases: List of training phases to include (e.g., ['forward', 'backward']).
        write_parquet: Whether to save output as Parquet format.
        write_csv: Whether to save output as CSV format.
        max_files: Maximum number of hook files to process (None for unlimited).
        chunksize: Number of rows to process per chunk for large datasets.
        run_id: Default run identifier if not present in CSV files.
        input_dir: Directory containing hook statistics files.
        hook_pattern: Glob pattern for hook statistics files.
        min_valid_ratio: Minimum ratio of valid (non-NaN) values required.
        fill_value: Default value for missing data after imputation.
        min_samples_threshold: Minimum number of samples required for aggregation.
        imputation_strategy: Strategy for handling missing values ('zero', 'mean', 'median').
    """
    reducer: str = "mean_pool"
    token_average: bool = True
    phases: Optional[List[str]] = None
    write_parquet: bool = True
    write_csv: bool = True
    max_files: Optional[int] = None
    chunksize: Optional[int] = None
    run_id: Optional[str] = None
    input_dir: str = "artifacts"
    hook_pattern: str = "hook_stats_*.csv"

    # NaN handling configuration
    min_valid_ratio: float = 0.5
    fill_value: float = 0.0
    min_samples_threshold: int = 10
    imputation_strategy: str = "zero"


def load_config(path: str) -> BuildConfig:
    """Load build configuration from YAML file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        BuildConfig instance with loaded parameters.
        
    Note:
        Only fields defined in BuildConfig dataclass are loaded;
        unknown keys in YAML are silently ignored.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Filter out keys that are not BuildConfig fields
    valid_keys = {f.name for f in BuildConfig.__dataclass_fields__.values()}
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    return BuildConfig(**filtered_cfg)


def _row_stats(vec: np.ndarray, prefix: str) -> Dict[str, Any]:
    """Compute statistical features for a vector of values.
    
    Args:
        vec: NumPy array containing feature values.
        prefix: Prefix string for statistic column names.
        
    Returns:
        Dictionary mapping statistic names to values.
    """
    stats = compute_stats(vec, prefix=f"{prefix}_")
    return stats


def _safe_git_hash() -> Optional[str]:
    """Retrieve current git commit hash safely.
    
    Returns:
        Git commit hash string, or None if git is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _collect_meta() -> Dict[str, Any]:
    """Collect metadata about the build environment.
    
    Returns:
        Dictionary containing build timestamp, hostname, and git commit.
    """
    return {
        "build_time": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "git_commit": _safe_git_hash(),
    }


def build_feature_rows_from_dataframe(df: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Build training-style feature rows from normalized hook statistics.

    This is the shared tabular feature pipeline used by both meta-dataset
    construction and compression-time inference. It preserves per-step rows and
    enriches them with epoch-level aggregate statistics, matching the feature
    schema used during meta-model training.
    """
    # Apply phase filtering with robust fallback
    if config.phases is not None and len(config.phases) > 0 and "phase" in df.columns:
        df_filtered = df[df["phase"].isin(config.phases)].copy()
        if df_filtered.empty:
            available_phases = sorted(df['phase'].dropna().unique().tolist())
            print(
                f"[warn] Phase filter {config.phases} removed all rows; "
                f"falling back to all phases present: {available_phases}"
            )
        else:
            df = df_filtered

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty after loading/phase filtering. "
            "Check input CSV files and 'phases' configuration."
        )

    has_listlike = ("activation_values" in df.columns) or ("grad_values" in df.columns)

    candidate_group_keys = ["run_id", "epoch", "step", "module", "phase"]
    group_keys = [k for k in candidate_group_keys if k in df.columns]
    if not group_keys:
        raise ValueError(
            f"No valid grouping keys found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    grouped_data = df.groupby(group_keys, sort=False, dropna=False)

    for keys, group_df in grouped_data:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_row = dict(zip(group_keys, keys))

        if has_listlike:
            activation_vectors, gradient_vectors = [], []

            if "activation_values" in group_df.columns:
                for arr in group_df["activation_values"].tolist():
                    if arr is not None:
                        reduced = apply_reducer(arr, config.reducer)
                        if reduced is not None:
                            activation_vectors.append(reduced)

            if "grad_values" in group_df.columns:
                for arr in group_df["grad_values"].tolist():
                    if arr is not None:
                        reduced = apply_reducer(arr, config.reducer)
                        if reduced is not None:
                            gradient_vectors.append(reduced)

            activation_vectors = np.array(activation_vectors) if activation_vectors else np.array([])
            gradient_vectors = np.array(gradient_vectors) if gradient_vectors else np.array([])

            if config.token_average and activation_vectors.size:
                activation_agg = np.nanmean(activation_vectors, axis=0)
            elif activation_vectors.size:
                activation_agg = activation_vectors.mean(axis=0)
            else:
                activation_agg = np.array([])

            if config.token_average and gradient_vectors.size:
                gradient_agg = np.nanmean(gradient_vectors, axis=0)
            elif gradient_vectors.size:
                gradient_agg = gradient_vectors.mean(axis=0)
            else:
                gradient_agg = np.array([])

            row = {**key_row, "reducer": config.reducer}
            row.update(_row_stats(activation_agg, "act"))
            row.update(_row_stats(gradient_agg, "grad"))
            records.append(row)
        else:
            activation_cols = [c for c in group_df.columns if c.startswith("act_")]
            gradient_cols = [c for c in group_df.columns if c.startswith("grad_")]
            param_cols = [c for c in group_df.columns if c.startswith("param_")]

            row = {**key_row, "reducer": config.reducer}
            if activation_cols:
                row.update({c: group_df[c].astype(float).mean() for c in activation_cols})
            if gradient_cols:
                row.update({c: group_df[c].astype(float).mean() for c in gradient_cols})
            if param_cols:
                row.update({c: group_df[c].astype(float).mean() for c in param_cols})
            records.append(row)

    batch_df = pd.DataFrame.from_records(records)
    if batch_df.empty:
        raise ValueError(
            f"Empty batch_df created. Available input columns: {list(df.columns)}. "
            f"Check for 'activation_values'/'grad_values' or pre-computed 'act_*'/'grad_*' columns."
        )

    candidate_epoch_keys = ["run_id", "epoch", "module", "phase", "reducer"]
    epoch_keys = [k for k in candidate_epoch_keys if k in batch_df.columns]
    if not epoch_keys:
        raise ValueError(
            f"No keys available for epoch aggregation. "
            f"Available batch_df columns: {list(batch_df.columns)}"
        )

    metric_cols = [
        c for c in batch_df.columns
        if c.startswith("act_") or c.startswith("grad_") or c.startswith("param_")
    ]

    print(f"[builder] Aggregating {len(metric_cols)} metric columns to epoch level...")
    print(
        f"[builder] Processing {len(batch_df):,} batch rows into ~{batch_df.groupby(epoch_keys).ngroups} epoch groups"
    )

    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = [
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('min', 'min'),
            ('max', 'max'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('valid_samples', lambda x: x.notna().sum()),
            ('valid_ratio', lambda x: x.notna().sum() / len(x))
        ]

    print("[builder] Running groupby aggregation (this may take 2-5 minutes)...")
    epoch_agg = batch_df.groupby(epoch_keys, dropna=False, sort=False).agg(agg_dict)
    epoch_agg.columns = [f'{col}_{agg}' if agg != '' else col for col, agg in epoch_agg.columns]
    epoch_agg = epoch_agg.reset_index()

    rename_map = {}
    for col in metric_cols:
        rename_map[f'{col}_mean'] = f'{col}_epoch_mean'
        rename_map[f'{col}_std'] = f'{col}_epoch_std'
        rename_map[f'{col}_median'] = f'{col}_epoch_median'
        rename_map[f'{col}_min'] = f'{col}_epoch_min'
        rename_map[f'{col}_max'] = f'{col}_epoch_max'
        rename_map[f'{col}_q25'] = f'{col}_epoch_q25'
        rename_map[f'{col}_q75'] = f'{col}_epoch_q75'
        rename_map[f'{col}_valid_samples'] = f'{col}_epoch_valid_samples'
        rename_map[f'{col}_valid_ratio'] = f'{col}_epoch_valid_ratio'

    epoch_df = epoch_agg.rename(columns=rename_map)
    print(f"[builder] ✓ Epoch aggregation complete: {len(epoch_df)} epoch groups created")

    print(f"[builder] Merging epoch aggregations back into {len(batch_df):,} batch rows...")
    merged = batch_df.merge(
        epoch_df,
        on=epoch_keys,
        how="left",
        suffixes=("", "_epoch_dup")
    )
    print(f"[builder] ✓ Merge complete: {len(merged):,} rows, {len(merged.columns)} columns")

    dup_cols = [c for c in merged.columns if c.endswith("_epoch_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
        print(f"[builder] Removed {len(dup_cols)} duplicate columns")

    return merged


def build_meta_dataset(input_dir: str, out_dir: str, config: BuildConfig) -> str:
    """Build aggregated meta-dataset from hook statistics.
    
    This function loads hook statistics from CSV files, applies dimensionality reduction,
    aggregates statistics across training steps and epochs, and saves the final dataset.
    
    Args:
        input_dir: Directory containing hook statistics CSV files.
        out_dir: Output directory for the meta-dataset.
        config: BuildConfig instance with processing parameters.
        
    Returns:
        Path to the generated Parquet file.
        
    Raises:
        ValueError: If input data is invalid or processing fails.
    """
    # Initialize NaN tracking statistics
    nan_stats = {
        "total_features": 0,
        "nan_features": 0,
        "nan_ratios": {},
        "dropped_groups": 0,
        "imputed_values": 0,
        "samples_below_threshold": 0
    }

    # Load hook CSV files with specified pattern
    print(f"[debug] Loading hook CSVs from {input_dir} with pattern {config.hook_pattern}")
    try:
        df = load_hook_csvs(
            input_dir,
            pattern=config.hook_pattern,
            max_files=config.max_files,
            chunksize=config.chunksize,
            default_run_id=config.run_id
        )
        print(f"[debug] Loaded DataFrame shape: {df.shape}")

        # Calculate initial NaN statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_stats["total_features"] = len(numeric_cols)
        nan_stats["nan_features"] = df[numeric_cols].isna().any().sum()

        for col in numeric_cols:
            nan_ratio = df[col].isna().mean()
            if nan_ratio > 0:
                nan_stats["nan_ratios"][col] = float(nan_ratio)

    except Exception as e:
        print(f"[error] Failed to load hook CSVs: {str(e)}")
        raise

    merged = build_feature_rows_from_dataframe(df, config)

    # Prepare output directory and metadata
    os.makedirs(out_dir, exist_ok=True)
    meta_info = _collect_meta()

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj: Any) -> Any:
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_native(x) for x in obj]
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    # Update metadata with NaN statistics
    meta_info["nan_statistics"] = convert_to_native(nan_stats)

    # Write metadata file
    meta_info_path = os.path.join(out_dir, "meta_info.json")
    with open(meta_info_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    # Write detailed NaN statistics separately
    nan_stats_path = os.path.join(out_dir, "nan_stats.json")
    with open(nan_stats_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_native(nan_stats), f, ensure_ascii=False, indent=2)

    # Write output files in configured formats
    out_parquet = os.path.join(out_dir, "meta_dataset.parquet")
    out_csv = os.path.join(out_dir, "meta_dataset.csv")
    out_columns = os.path.join(out_dir, "columns.json")

    # Always write Parquet (guaranteed output for train_meta)
    if not config.write_parquet:
        print("[builder] Warning: write_parquet=False, but Parquet is required for train_meta. Forcing write.")

    print(f"[builder] Writing parquet to {out_parquet}...")
    merged.to_parquet(out_parquet, index=False)
    print(f"[builder] ✓ Parquet file written ({os.path.getsize(out_parquet) / (1024 ** 2):.1f} MB)")

    # Export list of columns for consistency checks in downstream pipeline
    # Separate metadata columns from feature columns
    metadata_cols = ["run_id", "epoch", "step", "module", "layer", "param", "param_name",
                     "name", "phase", "reducer", "group", "id", "index", "idx"]
    feature_cols = [c for c in merged.columns if c not in metadata_cols]

    columns_manifest = {
        "total_columns": len(merged.columns),
        "feature_columns": sorted(feature_cols),
        "metadata_columns": sorted([c for c in metadata_cols if c in merged.columns]),
        "nan_threshold_used": config.min_valid_ratio,
        "generated_at": datetime.now().isoformat(),
    }

    with open(out_columns, "w", encoding="utf-8") as f:
        json.dump(columns_manifest, f, ensure_ascii=False, indent=2)
    print(f"[builder] ✓ Column manifest written: {out_columns} ({len(feature_cols)} features)")

    # Optional CSV export
    if config.write_csv:
        print(f"[builder] Writing CSV to {out_csv}...")
        merged.to_csv(out_csv, index=False)
        print(f"[builder] ✓ CSV file written ({os.path.getsize(out_csv) / (1024 ** 2):.1f} MB)")

    # Print summary statistics
    print(f"[info] NaN Statistics Summary:")
    print(f"  - Total features: {nan_stats['total_features']}")
    print(f"  - Features with NaNs: {nan_stats['nan_features']}")
    print(f"  - Dropped groups: {nan_stats['dropped_groups']}")
    print(f"  - Imputed values: {nan_stats['imputed_values']}")
    print(f"  - Samples below threshold: {nan_stats['samples_below_threshold']}")

    return out_parquet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[Feature Extraction] Starting with config: {args.config}")
    result = build_meta_dataset(
        input_dir=cfg.input_dir,
        out_dir="metapac/artifacts/meta_dataset",
        config=cfg
    )
    print(f"[Feature Extraction] Complete! Dataset saved to: {result}")
