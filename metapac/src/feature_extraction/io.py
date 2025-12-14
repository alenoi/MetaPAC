"""Input/output utilities for loading and processing hook statistics.

This module provides functions for:
- Loading hook statistics from CSV files
- Parsing and normalizing column data
- Handling missing values and data inconsistencies
- Schema validation and column aliasing
"""

import glob
import json
import os
import re
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_PATTERN = "hook_stats_epoch*.csv"


def _parse_listlike(col, fill_value=0.0, expected_dim=None) -> Optional[np.ndarray]:
    """Parse a list-like column value with improved NaN handling.
    
    Args:
        col: The input column value
        fill_value: Value to use for missing/invalid data
        expected_dim: Expected dimension of the output array
    
    Returns:
        np.ndarray: Parsed array with appropriate handling of missing values
    """
    if pd.isna(col):
        return np.full(expected_dim or 1, fill_value) if expected_dim else None

    if isinstance(col, (list, np.ndarray)):
        arr = np.array(col, dtype=float)
        if expected_dim and arr.size != expected_dim:
            # Pad or truncate to match expected dimension
            if arr.size < expected_dim:
                new_arr = np.full(expected_dim, fill_value)
                new_arr[:arr.size] = arr
                arr = new_arr
            else:
                arr = arr[:expected_dim]
        return arr

    if isinstance(col, (int, float)):
        return np.array([col], dtype=float)

    s = str(col).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = np.array(json.loads(s), dtype=float)
        except Exception:
            try:
                arr = np.array(eval(s), dtype=float)
            except Exception:
                return np.full(expected_dim or 1, fill_value) if expected_dim else None

        if expected_dim and arr.size != expected_dim:
            if arr.size < expected_dim:
                new_arr = np.full(expected_dim, fill_value)
                new_arr[:arr.size] = arr
                arr = new_arr
            else:
                arr = arr[:expected_dim]
        return arr

    try:
        parts = [float(x) for x in s.split(",")]
        arr = np.array(parts, dtype=float)
        if expected_dim and arr.size != expected_dim:
            if arr.size < expected_dim:
                new_arr = np.full(expected_dim, fill_value)
                new_arr[:arr.size] = arr
                arr = new_arr
            else:
                arr = arr[:expected_dim]
        return arr
    except Exception:
        return np.full(expected_dim or 1, fill_value) if expected_dim else None


def _filename_epoch(path: str) -> Optional[int]:
    """Extract epoch number from filename.
    
    Parses filenames like 'hook_stats_epoch12.csv' to extract the epoch number (12).
    
    Args:
        path: Path to the file.
        
    Returns:
        Epoch number if found, None otherwise.
    """
    match = re.search(r"epoch(\d+)", os.path.basename(path))
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _ensure_columns(df: pd.DataFrame,
                    path: str,
                    default_run_id: Optional[str] = None) -> pd.DataFrame:
    """Ensure required columns exist with proper aliasing and defaults.
    
    This function normalizes column names using aliases and provides sensible
    defaults for missing required columns (run_id, epoch, step, module, phase).
    
    Args:
        df: Input DataFrame from CSV file.
        path: Path to the CSV file (used for inferring defaults).
        default_run_id: Default run identifier if not present in data.
        
    Returns:
        DataFrame with standardized column names.
        
    Raises:
        ValueError: If required 'module' column cannot be found.
    """
    # Column alias mapping for flexible input formats
    aliases: Dict[str, List[str]] = {
        "run_id": ["run_id", "run", "runid", "runID"],
        "epoch": ["epoch", "ep", "epoch_idx"],
        "step": ["step", "global_step", "iter", "iteration", "step_idx", "batch"],
        "module": ["module", "layer", "name", "module_name"],
        "phase": ["phase", "split", "mode", "stage"],
    }

    out = df.copy()

    # Ensure run_id column exists
    if not any(alias in out.columns for alias in aliases["run_id"]):
        # Infer from path or use default
        run_id = (default_run_id or
                  os.path.basename(os.path.abspath(os.path.dirname(path))) or
                  "run_unknown")
        out["run_id"] = run_id
    else:
        # Rename from first matching alias
        for alias in aliases["run_id"]:
            if alias in out.columns:
                out.rename(columns={alias: "run_id"}, inplace=True)
                break

    # Ensure epoch column exists
    if not any(alias in out.columns for alias in aliases["epoch"]):
        # Try to infer from filename, default to 0
        filename_epoch = _filename_epoch(path)
        out["epoch"] = filename_epoch if filename_epoch is not None else 0
    else:
        for alias in aliases["epoch"]:
            if alias in out.columns:
                out.rename(columns={alias: "epoch"}, inplace=True)
                break

    # Ensure step column exists
    if not any(alias in out.columns for alias in aliases["step"]):
        # Generate sequential index if step is missing
        out["step"] = np.arange(len(out), dtype=int)
    else:
        for alias in aliases["step"]:
            if alias in out.columns:
                out.rename(columns={alias: "step"}, inplace=True)
                break

    # Ensure module column exists (required, no default)
    if not any(alias in out.columns for alias in aliases["module"]):
        raise ValueError(
            "Missing required column 'module' and no valid alias found "
            "(tried: 'module', 'layer', 'name')."
        )
    else:
        for alias in aliases["module"]:
            if alias in out.columns:
                out.rename(columns={alias: "module"}, inplace=True)
                break

    # Ensure phase column exists
    if not any(alias in out.columns for alias in aliases["phase"]):
        # Default to 'train' phase
        out["phase"] = "train"
    else:
        for alias in aliases["phase"]:
            if alias in out.columns:
                out.rename(columns={alias: "phase"}, inplace=True)
                break

    return out


def load_hook_csvs(input_dir: str,
                   pattern: str = DEFAULT_PATTERN,
                   max_files: Optional[int] = None,
                   chunksize: Optional[int] = None,
                   default_run_id: Optional[str] = None) -> pd.DataFrame:
    """Load and concatenate hook statistics CSV files.
    
    This function loads multiple CSV files matching a pattern, normalizes their
    columns using aliases and defaults, and concatenates them into a single DataFrame.
    Supports chunked reading for large files.
    
    Args:
        input_dir: Directory containing hook statistics files.
        pattern: Glob pattern for matching CSV files.
        max_files: Maximum number of files to load (None for unlimited).
        chunksize: Number of rows per chunk for large files (None to load all at once).
        default_run_id: Default run identifier for files without run_id.
        
    Returns:
        Concatenated DataFrame with normalized columns.
        
    Raises:
        FileNotFoundError: If no matching CSV files are found.
    """
    # Find and sort matching files
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if max_files is not None:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}' matching pattern '{pattern}'."
        )

    print(f"[io] Loading {len(paths)} CSV files from {input_dir}...")

    # Load all files (with optional chunking) with progress bar
    frames = []
    for path in tqdm(paths, desc="Loading CSV files", unit="file"):
        if chunksize:
            # Process file in chunks for memory efficiency
            for chunk in pd.read_csv(path, chunksize=chunksize):
                chunk = _ensure_columns(chunk, path, default_run_id)
                frames.append(chunk)
        else:
            # Load entire file at once
            df = pd.read_csv(path)
            df = _ensure_columns(df, path, default_run_id)
            frames.append(df)

    print(f"[io] ✓ Loaded {len(frames)} dataframe(s)")

    # Concatenate all frames
    df = pd.concat(frames, ignore_index=True)

    # Normalize list-like columns if present
    for column in ["activation_values", "grad_values"]:
        if column in df.columns:
            df[column] = df[column].apply(_parse_listlike)

    # Normalize activation statistic column names
    activation_aliases = {
        "mean": "act_mean",
        "std": "act_std",
        "min": "act_min",
        "max": "act_max",
        "l2": "act_l2",
        "l1": "act_l1",
        "sparsity": "act_sparsity",
        "q25": "act_q25",
        "q50": "act_q50",
        "q75": "act_q75",
    }
    for old_name, new_name in activation_aliases.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    return df


def schema_columns(metric_prefix: str = "act") -> List[str]:
    """Get standard schema column names for a metric prefix.
    
    Args:
        metric_prefix: Prefix for metric columns (e.g., 'act', 'grad').
        
    Returns:
        List of standard column names for statistics.
    """
    return [
        f"{metric_prefix}_mean",
        f"{metric_prefix}_std",
        f"{metric_prefix}_min",
        f"{metric_prefix}_max",
        f"{metric_prefix}_q25",
        f"{metric_prefix}_q50",
        f"{metric_prefix}_q75",
        f"{metric_prefix}_sparsity"
    ]
