"""Preparation phase: Feature extraction, importance scoring, and zoning.

This module handles the preparation phase of the compression pipeline:
1. Load meta-predictor checkpoint (portable or legacy format)
2. Extract parameter-level features from target model
3. Compute importance scores using meta-predictor
4. Rank and partition parameters into compression zones

Dependencies:
    - Meta-predictor checkpoint
    - Parameter hook statistics CSV
    - Target model path
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from metapac.src.model_profiles import resolve_model_profile_from_name

from ...feature_extraction.builder import BuildConfig, build_feature_rows_from_dataframe
from ...feature_extraction.io import load_hook_csvs
from ...utils.logging_utils import log_metric

logger = logging.getLogger(__name__)


def _expected_hook_prefixes_for_target_model(target_model_path: str | Path) -> tuple[str, ...]:
    return resolve_model_profile_from_name(target_model_path).expected_hook_prefixes


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    """Resolve a checkpoint path, allowing prefix-based directory matches.

    If the provided path does not exist, this falls back to the newest sibling
    directory whose name starts with the provided basename. This supports configs
    that specify a stable checkpoint prefix while train_meta produces
    timestamp-suffixed portable checkpoint directories.
    """
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        return ckpt_path

    parent = ckpt_path.parent
    prefix = ckpt_path.name
    if not parent.exists() or not prefix:
        return ckpt_path

    candidates = [
        candidate for candidate in parent.iterdir()
        if candidate.is_dir()
        and candidate.name.startswith(prefix)
        and (candidate / "model_state.pt").exists()
        and (candidate / "feature_names.json").exists()
    ]
    if not candidates:
        return ckpt_path

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    logger.info(
        "Resolved meta-predictor checkpoint prefix %s -> %s",
        checkpoint_path,
        latest,
    )
    return latest


# ---------------------------------------------------------------------------
# Meta-predictor loading
# ---------------------------------------------------------------------------

def load_meta_predictor_checkpoint(
    checkpoint_path: str
) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load meta-predictor checkpoint (supports both legacy and portable formats).

    Args:
        checkpoint_path: Path to meta-predictor checkpoint (directory or .joblib file)

    Returns:
        Tuple of (model, imputer, scaler, feature_names, target_col, task_type)

    Raises:
        FileNotFoundError: If checkpoint not found
        ValueError: If checkpoint format is unrecognized
    """
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Meta-predictor checkpoint not found: {checkpoint_path}"
        )

    logger.info(f"Loading meta-predictor from: {ckpt_path}")

    # Portable directory format
    if ckpt_path.is_dir():
        model_state_path = ckpt_path / "model_state.pt"
        preprocess_path = ckpt_path / "preprocess.joblib"
        features_path = ckpt_path / "feature_names.json"

        if (model_state_path.exists() and preprocess_path.exists() 
            and features_path.exists()):
            logger.debug("Detected portable checkpoint format")
            return _load_portable_checkpoint(ckpt_path)

    # Legacy joblib
    if ckpt_path.suffix == ".joblib" or ckpt_path.is_file():
        logger.debug("Detected legacy joblib checkpoint format")
        return _load_legacy_checkpoint(ckpt_path)

    raise ValueError(
        f"Unrecognized checkpoint format: {checkpoint_path}\n"
        "Expected either:\n"
        "  - Directory with model_state.pt, preprocess.joblib, feature_names.json (portable)\n"
        "  - .joblib file (legacy)"
    )


def _load_portable_checkpoint(
    checkpoint_dir: Path
) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load portable checkpoint format (environment-independent).

    Args:
        checkpoint_dir: Directory containing portable checkpoint files

    Returns:
        Tuple of (model, imputer, scaler, feature_names, target_name, task_type)
    """
    from ...models.meta_predictor import load_checkpoint_portable

    model, imputer, scaler, feature_names, target_name, task_type, metadata = \
        load_checkpoint_portable(checkpoint_dir)

    logger.info(
        f"Loaded portable checkpoint: task={task_type}, target={target_name}"
    )
    log_metric(logger, "Model features", len(feature_names))

    return model, imputer, scaler, feature_names, target_name, task_type


def _load_legacy_checkpoint(
    checkpoint_path: Path
) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load legacy joblib checkpoint format.

    Args:
        checkpoint_path: Path to legacy .joblib checkpoint

    Returns:
        Tuple of (pipeline, None, None, features, target, task)

    Raises:
        ModuleNotFoundError: If checkpoint has import path dependencies
        KeyError: If checkpoint is missing required keys
    """
    try:
        checkpoint = joblib.load(checkpoint_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to load legacy checkpoint due to import path issue: {e}\n"
            "This typically happens when the checkpoint was saved with a different environment.\n"
            "Solution: Re-train the meta-predictor to generate a portable checkpoint."
        ) from e

    required_keys = ["pipeline", "features", "target", "task"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Invalid checkpoint format: missing '{key}' key")

    pipeline = checkpoint["pipeline"]
    features = checkpoint["features"]
    target = checkpoint["target"]
    task = checkpoint["task"]

    logger.info(f"Loaded legacy checkpoint: task={task}, target={target}")
    log_metric(logger, "Model features", len(features))
    logger.warning("Legacy format may have import-path dependencies")

    return pipeline, None, None, features, target, task


def create_preprocessed_pipeline(model: Any, imputer: Any, scaler: Any) -> Any:
    """Create a preprocessing pipeline wrapper for meta-predictor.
    
    Wraps raw meta-predictor model with imputer/scaler preprocessing.
    
    Args:
        model: Meta-predictor model
        imputer: Feature imputer (handles missing values)
        scaler: Feature scaler (standardization)
    
    Returns:
        PreprocessedModel wrapper with .predict() method
    """
    import torch
    
    class PreprocessedModel:
        """Wrapper that applies preprocessing before model inference."""
        
        def __init__(self, model, imputer, scaler):
            self.model = model
            self.imputer = imputer
            self.scaler = scaler
        
        @staticmethod
        def _safe_imputer_transform(imputer, X):
            """Version-robust imputer transform with fallback."""
            try:
                return imputer.transform(X)
            except Exception:
                # Fallback for version incompatibilities
                arr = np.asarray(X, dtype=np.float64)
                arr = np.where(np.isfinite(arr), arr, np.nan)
                
                stats = getattr(imputer, "statistics_", None)
                if stats is None:
                    raise
                    
                stats = np.asarray(stats, dtype=np.float64)
                if arr.shape[1] != stats.shape[0]:
                    raise ValueError(f"Shape mismatch: X={arr.shape}, stats={stats.shape}")
                
                out = arr.copy()
                mask = np.isnan(out)
                if mask.any():
                    row_idx, col_idx = np.where(mask)
                    out[row_idx, col_idx] = stats[col_idx]
                return out
        
        @staticmethod
        def _safe_scaler_transform(scaler, X):
            """Version-robust scaler transform with fallback."""
            try:
                return scaler.transform(X)
            except Exception:
                arr = np.asarray(X, dtype=np.float64)
                mean_ = np.asarray(getattr(scaler, "mean_", np.zeros(arr.shape[1])), dtype=np.float64)
                scale_ = np.asarray(getattr(scaler, "scale_", np.ones(arr.shape[1])), dtype=np.float64)
                
                if mean_.shape[0] != arr.shape[1] or scale_.shape[0] != arr.shape[1]:
                    raise ValueError(f"Shape mismatch")
                
                safe_scale = np.where(scale_ == 0.0, 1.0, scale_)
                return (arr - mean_) / safe_scale
        
        def predict(self, X):
            """Apply preprocessing and predict."""
            X_imputed = self._safe_imputer_transform(self.imputer, X)
            X_scaled = self._safe_scaler_transform(self.scaler, X_imputed)
            
            if isinstance(self.model, torch.nn.Module):
                X_tensor = torch.from_numpy(X_scaled).float()
                with torch.no_grad():
                    return self.model(X_tensor).cpu().numpy()
            else:
                return self.model.predict(X_scaled)
    
    return PreprocessedModel(model, imputer, scaler)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_parameter_features(
    target_model_path: str,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract features from target model parameters for importance prediction.

    This function loads parameter-level hook statistics and aggregates them into
    features for meta-predictor inference. Supports both single-epoch and multi-epoch
    statistics with rich epoch-level aggregation.

    Args:
        target_model_path: Path to target model directory
        feature_names: List of feature names expected by meta-predictor

    Returns:
        DataFrame with columns: parameter_name, <feature_names>

    Raises:
        FileNotFoundError: If hook statistics CSV not found
        ValueError: If no parameter-level records found
    """
    logger.info(
        f"Extracting parameter-level features from: {target_model_path}"
    )
    log_metric(logger, "Expected features", len(feature_names))

    model_path = Path(target_model_path)
    local_roots = [model_path]
    if model_path.name.startswith("checkpoint-") and model_path.parent not in local_roots:
        local_roots.append(model_path.parent)

    # Collect candidate hook-stat files (priority order)
    local_hook_candidates = []
    for root in local_roots:
        local_hook_candidates.extend(
            sorted(
                (root / "artifacts" / "raw").glob("hook_stats_epoch*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )

    raw_hook_candidates = sorted(
        Path("metapac")
        .joinpath("artifacts", "raw")
        .glob("hook_stats_epoch*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    possible_paths = [*local_hook_candidates, *raw_hook_candidates]
    for root in local_roots:
        possible_paths.extend(
            [
                root / "runs" / "parameter_stats_epoch0.csv",
                root / "artifacts" / "parameter_stats.csv",
            ]
        )
    possible_paths.extend(
        [
            Path("artifacts") / "distilgpt2_parameter_stats.csv",
            Path("artifacts") / "distilbert_parameter_stats.csv",
            Path("artifacts") / "parameter_hook_stats.csv",
            Path("artifacts") / "parameter_level_stats_demo.csv",
        ]
    )

    # Deduplicate while keeping order
    dedup_paths = []
    seen = set()
    for p in possible_paths:
        rp = str(Path(p))
        if rp not in seen:
            seen.add(rp)
            dedup_paths.append(Path(p))

    # Pick best candidate by namespace compatibility
    expected_prefixes = _expected_hook_prefixes_for_target_model(model_path)

    def _score_namespace(path: Path) -> float:
        """Score how well a CSV matches expected model namespace."""
        try:
            sample = pd.read_csv(path, usecols=["module"], nrows=2048)
            modules = sample["module"].dropna().astype(str)
            if len(modules) == 0:
                return 0.0
            hits = 0
            for pref in expected_prefixes:
                hits += modules.str.startswith(pref).sum()
            return hits / float(len(modules))
        except Exception:
            return -1.0

    def _is_model_local(path: Path) -> bool:
        for root in local_roots:
            try:
                if path.resolve().is_relative_to(root.resolve()):
                    return True
            except Exception:
                continue
        return False

    scored = []
    for path in dedup_paths:
        if path.exists():
            scored.append((path, int(_is_model_local(path)), _score_namespace(path), path.stat().st_mtime))

    hook_csv_path = None
    if scored:
        # Prefer model-local artifacts first, then namespace score, then newer file.
        hook_csv_path = max(scored, key=lambda t: (t[1], t[2], t[3]))[0]

    if hook_csv_path is None:
        raise FileNotFoundError(
            f"No parameter-level hook statistics CSV found. "
            f"Searched paths: {[str(p) for p in dedup_paths]}"
        )

    logger.info(f"Loading hook statistics from: {hook_csv_path}")
    if hook_csv_path.name.startswith("hook_stats_epoch"):
        hook_pattern = "hook_stats_epoch*.csv"
    elif hook_csv_path.name.startswith("parameter_stats_epoch"):
        hook_pattern = "parameter_stats_epoch*.csv"
    else:
        hook_pattern = hook_csv_path.name

    df_hooks = load_hook_csvs(
        str(hook_csv_path.parent),
        pattern=hook_pattern,
        default_run_id=local_roots[-1].name,
    )
    shared_cfg = BuildConfig(
        reducer="mean_pool",
        token_average=True,
        phases=["parameter"],
        input_dir=str(hook_csv_path.parent),
        hook_pattern=hook_pattern,
        run_id=local_roots[-1].name,
    )
    feature_rows = build_feature_rows_from_dataframe(df_hooks, shared_cfg)

    if "module" not in feature_rows.columns:
        raise ValueError("Shared feature pipeline did not produce 'module' column")

    unique_params = feature_rows["module"].dropna().unique()
    log_metric(logger, "Unique parameters", len(unique_params))
    logger.info(
        f"Built training-style feature rows: {len(feature_rows)} rows across "
        f"{len(unique_params)} parameters"
    )

    # Match expected features from meta-predictor
    missing_features = []
    available_features = []
    feature_data = {}

    for feat in feature_names:
        if feat in feature_rows.columns:
            feature_data[feat] = feature_rows[feat]
            available_features.append(feat)
        else:
            # Try with _mean suffix
            feat_with_suffix = f"{feat}_mean"
            if feat_with_suffix in feature_rows.columns:
                feature_data[feat] = feature_rows[feat_with_suffix]
                available_features.append(feat)
            elif feat == 'epoch':
                feature_data[feat] = np.zeros(len(feature_rows), dtype=np.float64)
                missing_features.append(feat)
            elif feat == 'step':
                feature_data[feat] = np.zeros(len(feature_rows), dtype=np.float64)
                missing_features.append(feat)
            elif feat.startswith('act_'):
                # Activation stats may be missing; fill with zeros
                feature_data[feat] = np.zeros(len(feature_rows), dtype=np.float64)
                missing_features.append(feat)
            elif feat.endswith('_epoch_mean') or feat.endswith('_epoch_std'):
                # Epoch-level stats may be missing in single-epoch runs
                feature_data[feat] = np.zeros(len(feature_rows), dtype=np.float64)
                missing_features.append(feat)
            else:
                feature_data[feat] = np.zeros(len(feature_rows), dtype=np.float64)
                missing_features.append(feat)

    # Build once to avoid DataFrame fragmentation
    param_features = pd.DataFrame({
        'parameter_name': feature_rows['module'],
        **feature_data
    })

    logger.info(
        f"Matched {len(available_features)}/{len(feature_names)} features "
        "from hook stats"
    )
    if missing_features:
        logger.warning(
            f"{len(missing_features)} features not found, filled with zeros"
        )
        if len(missing_features) <= 10:
            logger.debug(f"Missing: {missing_features}")

    # Final numeric safety: replace non-finite values
    feature_cols = [c for c in param_features.columns if c != "parameter_name"]
    vals = param_features[feature_cols].to_numpy(dtype=np.float64, copy=True)
    non_finite = ~np.isfinite(vals)
    if non_finite.any():
        n_bad = int(non_finite.sum())
        logger.warning(
            f"Found {n_bad} non-finite feature values; replacing with 0.0"
        )
        vals[non_finite] = 0.0
        param_features[feature_cols] = vals

    logger.info(
        f"Final feature matrix: {len(param_features)} parameters × "
        f"{len(param_features.columns) - 1} features"
    )

    return param_features


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def compute_importance_scores(
    pipeline: Any,
    param_features: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """Compute importance scores for each parameter using meta-predictor.

    Args:
        pipeline: Meta-predictor model (can be raw model or PreprocessedModel wrapper)
        param_features: DataFrame with parameter features
        feature_names: List of feature names expected by meta-predictor

    Returns:
        DataFrame with columns: parameter_name, importance_score

    Raises:
        ValueError: If feature dimensions don't match checkpoint
    """
    logger.info("Computing importance scores...")

    try:
        # Keep exact checkpoint feature dimensionality/order
        X = param_features[feature_names]
        importance_scores = pipeline.predict(X)
    except Exception as e:
        logger.warning(f"Error with full feature set ({len(feature_names)}): {e}")
        logger.info("Trying with base act_* features only...")

        base_features = ['epoch', 'step'] + [
            f'act_{stat}' for stat in
            ['mean', 'std', 'min', 'max', 'l2', 'l1', 'sparsity', 'q25', 'q50', 'q75']
        ]
        base_features = [f for f in base_features if f in param_features.columns]

        logger.debug(f"Using {len(base_features)} base features")
        if len(base_features) != len(feature_names):
            raise ValueError(
                f"Base-feature fallback has {len(base_features)} features, "
                f"but checkpoint expects {len(feature_names)}. "
                f"Please use a matching meta checkpoint for this feature schema."
            )
        X = param_features[base_features]
        importance_scores = pipeline.predict(X)

    row_level_scores = pd.DataFrame({
        "parameter_name": param_features["parameter_name"],
        "importance_score": importance_scores
    })

    result = (
        row_level_scores
        .groupby("parameter_name", as_index=False)["importance_score"]
        .mean()
    )

    if len(row_level_scores) != len(result):
        logger.info(
            "Aggregated row-level importance predictions from %d rows to %d parameters",
            len(row_level_scores),
            len(result),
        )

    logger.info(
        f"Score range: [{result['importance_score'].min():.4f}, {result['importance_score'].max():.4f}], "
        f"mean={result['importance_score'].mean():.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------

def rank_and_partition_parameters(
    importance_df: pd.DataFrame,
    zones_config: Dict[str, Dict[str, Any]],
    zone_assignment_cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Rank parameters by importance and partition into compression zones.

    Supports two assignment methods:
    1. Quantile-based (default): Assign zones based on importance quantile ranges
    2. Cluster-based (kmeans/gmm): Use clustering to identify natural groupings

    Args:
        importance_df: DataFrame with parameter_name and importance_score columns
        zones_config: Zone configuration (high/medium/low zones with actions)
        zone_assignment_cfg: Zone assignment configuration (method, parameters)

    Returns:
        DataFrame with added columns: rank, quantile, zone, action, target_bits
    """
    df = importance_df.copy()
    df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    scores = df['importance_score'].values
    df['quantile'] = df['importance_score'].rank(pct=True)

    log_metric(logger, "Ranked parameters", len(df))
    logger.info(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # Parse zone configuration
    zones = {}
    for zone_name, zone_cfg in zones_config.items():
        zones[zone_name] = {
            'min': float(zone_cfg.get('quantile_min', 0.0)),
            'max': float(zone_cfg.get('quantile_max', 1.0)),
            'action': zone_cfg.get('action', 'keep'),
            'config': zone_cfg
        }

    assignment = zone_assignment_cfg or {}
    assignment_method = str(assignment.get('method', 'quantile')).lower()

    # Cluster-based assignment
    if assignment_method in {'cluster', 'clustering', 'kmeans', 'gmm', 'gaussian_mixture'}:
        kmeans_cfg = assignment.get('kmeans', {}) if isinstance(assignment, dict) else {}
        n_clusters = int(kmeans_cfg.get('n_clusters', 3))
        random_state = int(kmeans_cfg.get('random_state', 42))
        n_init = int(kmeans_cfg.get('n_init', 10))
        min_high_fraction = float(kmeans_cfg.get('min_high_fraction', 0.0))
        min_high_fraction = float(max(0.0, min(1.0, min_high_fraction)))

        if n_clusters != 3:
            logger.warning(
                f"Zone clustering currently supports 3 clusters best; "
                f"got {n_clusters}, forcing 3"
            )
            n_clusters = 3

        score_2d = df[['importance_score']].to_numpy(dtype=np.float64)
        
        if assignment_method in {'gmm', 'gaussian_mixture'}:
            gm = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state,
                n_init=n_init
            )
            labels = gm.fit_predict(score_2d)
            cluster_method_name = "gaussian_mixture clustering"
        else:
            km = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init
            )
            labels = km.fit_predict(score_2d)
            cluster_method_name = "kmeans clustering"

        # Map clusters to zones (low/medium/high) by mean score
        cluster_stats = (
            pd.DataFrame({
                'cluster': labels,
                'score': df['importance_score'].to_numpy()
            })
            .groupby('cluster')['score']
            .mean()
            .sort_values(ascending=True)
        )
        ordered_clusters = list(cluster_stats.index)
        if len(ordered_clusters) != 3:
            score_std = float(np.std(scores)) if len(scores) > 0 else 0.0
            score_min = float(np.min(scores)) if len(scores) > 0 else 0.0
            score_max = float(np.max(scores)) if len(scores) > 0 else 0.0
            rounded_unique = int(np.unique(np.round(scores, decimals=8)).size) if len(scores) > 0 else 0
            raise ValueError(
                "Zone clustering collapsed because predicted importance scores do not separate into 3 distinct groups. "
                f"method={cluster_method_name}, distinct_clusters={len(ordered_clusters)}, "
                f"score_min={score_min:.8g}, score_max={score_max:.8g}, score_std={score_std:.8g}, "
                f"unique_scores_1e-8={rounded_unique}. "
                "This usually indicates a meta-predictor problem (for example poor training signal, degenerate target, "
                "or feature mismatch). Stop and inspect/retrain the meta model instead of defaulting to quantile zoning."
            )

        cluster_to_zone = {
            ordered_clusters[0]: 'low',
            ordered_clusters[1]: 'medium',
            ordered_clusters[2]: 'high',
        }
        df['zone'] = [cluster_to_zone[int(c)] for c in labels]

        # Optional guardrail: enforce minimum HIGH-zone share
        total_n = len(df)
        current_high_n = int((df['zone'] == 'high').sum())
        min_high_n = int(np.ceil(min_high_fraction * total_n)) if total_n > 0 else 0
        if min_high_n > current_high_n:
            need = int(min_high_n - current_high_n)
            promote_idx = (
                df[df['zone'] != 'high']
                .sort_values('importance_score', ascending=False)
                .head(need)
                .index
            )
            if len(promote_idx) > 0:
                df.loc[promote_idx, 'zone'] = 'high'
                logger.info(
                    f"Promoted {len(promote_idx)} parameters to HIGH zone "
                    f"to meet min_high_fraction={min_high_fraction:.2%}"
                )

        logger.info(
            f"Zone assignment method: {cluster_method_name} on predicted importance"
        )
        for cid in ordered_clusters:
            z = cluster_to_zone[cid]
            cdf = df[np.array(labels) == cid]
            logger.info(
                f"  cluster={cid} -> {z.upper():6s}: n={len(cdf)}, "
                f"score_mean={cdf['importance_score'].mean():.2f}, "
                f"range=[{cdf['importance_score'].min():.2f}, "
                f"{cdf['importance_score'].max():.2f}]"
            )
    else:
        # Quantile-based assignment
        def assign_zone(q: float) -> str:
            for zone_name, zone_info in zones.items():
                if zone_info['min'] <= q < zone_info['max']:
                    return zone_name
            return 'medium'

        df['zone'] = df['quantile'].apply(assign_zone)

    # Map zones to actions and target bits
    _zone_to_action = {k: v['action'] for k, v in zones.items()}
    df['action'] = df['zone'].map(_zone_to_action).fillna('keep')

    def get_target_bits(zone_name: str) -> Optional[int]:
        zone_cfg = zones.get(zone_name, {}).get('config', {})
        return zone_cfg.get('bits', None)

    df['target_bits'] = df['zone'].apply(get_target_bits)

    # Log zone statistics
    method_label = (
        "cluster" if assignment_method in {
            'cluster', 'clustering', 'kmeans', 'gmm', 'gaussian_mixture'
        }
        else "quantile"
    )
    
    logger.info(f"Zone assignment complete ({method_label} method):")
    for zone_name in ['high', 'medium', 'low']:
        if zone_name in zones:
            zone_df = df[df['zone'] == zone_name]
            if len(zone_df) > 0:
                logger.info(
                    f"  {zone_name.upper():6s}: {len(zone_df):5d} params "
                    f"({len(zone_df) / len(df) * 100:5.1f}%), "
                    f"action={zones[zone_name]['action']}, "
                    f"score_range=[{zone_df['importance_score'].min():.2f}, "
                    f"{zone_df['importance_score'].max():.2f}]"
                )

    return df
