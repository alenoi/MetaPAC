# metapac/src/compression/strategy.py
"""Compression strategy implementation for three-zone model compression.

This module orchestrates the compression pipeline:
1. Load meta-predictor and compute importance scores
2. Rank weights/activations by importance
3. Partition into three zones (high/medium/low)
4. Apply zone-specific compression (keep/quantize/prune)
5. Save compressed model and metrics
"""
from __future__ import annotations

import json
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd

from .finalize import finalize_artifacts

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

from .quantization import Quantizer, QuantizationConfig, save_quantization_metadata
from ..utils.logging_utils import setup_logger, log_phase_header, log_section, log_metric

# Setup logger
logger = setup_logger(__name__)

import torch
import torch.nn as nn

# Variable-bit helpers: registry and safe export
from .variable_bit_layers import register_quantized_layer, ensure_registry


# ---------------------------------------------------------------------------
# Helper: Load default configuration
# ---------------------------------------------------------------------------

def _load_strategy_defaults() -> Dict[str, Any]:
    """Load default configuration from strategy_defaults.yaml.
    
    Returns:
        Dictionary with default configuration values.
        Returns empty dict if file not found (graceful fallback).
    """
    try:
        # Resolve config path relative to this file
        current_file = Path(__file__).resolve()
        config_path = current_file.parent.parent.parent / "configs" / "strategy_defaults.yaml"
        
        if not config_path.exists():
            logger.warning(f"Strategy defaults config not found: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            defaults = yaml.safe_load(f)
        
        logger.debug(f"Loaded strategy defaults from: {config_path}")
        return defaults or {}
    
    except Exception as e:
        logger.warning(f"Failed to load strategy defaults: {e}")
        return {}


def _merge_with_defaults(user_cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user config with defaults (user config takes precedence).
    
    Args:
        user_cfg: User-provided configuration.
        defaults: Default configuration values.
    
    Returns:
        Merged configuration dictionary.
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(defaults, user_cfg)


# ---------------------------------------------------------------------------
# Helper: build variable-bit registry from combined quantization metadata
# ---------------------------------------------------------------------------

def _resolve_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[Optional[nn.Module], Optional[str]]:
    """
    Resolve 'encoder.layer.2.output.dense' → (parent_module, 'dense').
    Returns (None, None) if resolution fails.
    """
    try:
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]
    except Exception:
        return None, None


def _get_module_by_name(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """
    Return the module with exact dotted name using named_modules() map, fallback to attribute walk.
    """
    name_map = dict(root.named_modules())
    if dotted_name in name_map:
        return name_map[dotted_name]
    parent, attr = _resolve_parent_and_attr(root, dotted_name)
    if parent is not None and attr and hasattr(parent, attr):
        return getattr(parent, attr)
    return None


def _infer_assigned_bits(meta: Dict[str, Any], default_bits: int = 8) -> int:
    """
    Infer effective bit-width from heterogeneous metadata dicts.
    Tries common keys: 'assigned_bits', 'bits', 'target_bits', 'final_bits'.
    """
    for k in ("assigned_bits", "bits", "final_bits", "target_bits"):
        v = meta.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return int(default_bits)


def _attach_quant_meta_and_register(root: nn.Module, layer: nn.Module, layer_name: str, bits: int) -> None:
    """
    Attach minimal quantization metadata to a layer and register it on the root model.
    This makes variable-bit export deterministic without scanning the entire tree.
    """
    if not hasattr(layer, "weight") or getattr(layer, "weight") is None:
        weight_numel = 0
        shape = None
    else:
        try:
            weight_numel = int(layer.weight.numel())
            shape = tuple(layer.weight.shape)
        except Exception:
            weight_numel = 0
            shape = None

    layer.quant_meta = {
        "name": layer_name,
        "bits": int(bits),
        "weight_numel": weight_numel,
        "shape": shape,
    }
    register_quantized_layer(root, layer)


def _build_variable_bit_registry_from_meta(
        model: nn.Module,
        combined_meta: Dict[str, Dict[str, Any]],
        *,
        exclude_layernorm_and_classifier: bool = True,
        fallback_bits: int = 8,
) -> int:
    """
    Use the union of quantization metadata (quant_meta + trim_meta) to attach `quant_meta`
    to actual nn.Modules and register them for variable-bit export.

    Returns the number of successfully registered layers.
    """
    ensure_registry(model)
    name_map = dict(model.named_modules())
    registered = 0

    for name, meta in combined_meta.items():
        # In many pipelines keys are parameter names like "...weight";
        # try to map to module by stripping trailing '.weight' or '.bias' if needed.
        candidate_names = [name]
        if name.endswith(".weight") or name.endswith(".bias"):
            candidate_names.append(name.rsplit(".", 1)[0])

        target_module = None
        target_module_name = None
        for cand in candidate_names:
            m = name_map.get(cand)
            if m is None:
                m = _get_module_by_name(model, cand)
            if isinstance(m, nn.Module):
                target_module = m
                target_module_name = cand
                break

        if target_module is None:
            # Could not resolve; skip silently but keep going
            continue

        # Optional: exclude LayerNorm and classifier from variable-bit accounting
        if exclude_layernorm_and_classifier:
            clsname = target_module.__class__.__name__
            if "LayerNorm" in clsname or "layernorm" in clsname.lower():
                continue
            if target_module_name.endswith("classifier") or "classifier" in target_module_name:
                continue

        bits = _infer_assigned_bits(meta, default_bits=fallback_bits)
        _attach_quant_meta_and_register(model, target_module, target_module_name, bits)
        registered += 1

    return registered


# ---------------------------------------------------------------------------
# Meta-predictor loading utilities (unchanged)
# ---------------------------------------------------------------------------

def load_meta_predictor_checkpoint(checkpoint_path: str) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load meta-predictor checkpoint (supports both legacy and portable formats)."""
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Meta-predictor checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading meta-predictor from: {checkpoint_path}")

    # Portable directory format
    if ckpt_path.is_dir():
        model_state_path = ckpt_path / "model_state.pt"
        preprocess_path = ckpt_path / "preprocess.joblib"
        features_path = ckpt_path / "feature_names.json"

        if model_state_path.exists() and preprocess_path.exists() and features_path.exists():
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


def _load_portable_checkpoint(checkpoint_dir: Path) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load portable checkpoint format (environment-independent)."""
    from ..models.meta_predictor import load_checkpoint_portable

    model, imputer, scaler, feature_names, target_name, task_type, metadata = \
        load_checkpoint_portable(checkpoint_dir)

    logger.info(f"Loaded portable checkpoint: task={task_type}, target={target_name}")
    log_metric(logger, "Model features", len(feature_names))

    return model, imputer, scaler, feature_names, target_name, task_type


def _load_legacy_checkpoint(checkpoint_path: Path) -> Tuple[Any, Any, Any, List[str], str, str]:
    """Load legacy joblib checkpoint format."""
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


# ---------------------------------------------------------------------------
# Feature extraction, scoring, zoning (unchanged)
# ---------------------------------------------------------------------------

def extract_parameter_features(target_model_path: str, feature_names: list) -> pd.DataFrame:
    """Extract features from target model parameters for importance prediction."""
    logger.info(f"Extracting parameter-level features from: {target_model_path}")
    log_metric(logger, "Expected features", len(feature_names))

    model_path = Path(target_model_path)
    possible_paths = [
        Path("artifacts") / "distilbert_parameter_stats.csv",
        model_path / "runs" / "parameter_stats_epoch0.csv",
        model_path / "artifacts" / "parameter_stats.csv",
        Path("artifacts") / "parameter_hook_stats.csv",
        Path("artifacts") / "parameter_level_stats_demo.csv",
    ]

    hook_csv_path = None
    for path in possible_paths:
        if path.exists():
            hook_csv_path = path
            break

    if hook_csv_path is None:
        raise FileNotFoundError(
            f"No parameter-level hook statistics CSV found. Searched paths: {[str(p) for p in possible_paths]}"
        )

    logger.info(f"Loading hook statistics from: {hook_csv_path}")
    df_hooks = pd.read_csv(hook_csv_path)

    if 'phase' in df_hooks.columns:
        df_params = df_hooks[df_hooks['phase'] == 'parameter'].copy()
        logger.debug(f"Filtered {len(df_params)} parameter records from {len(df_hooks)} total")
    else:
        df_params = df_hooks.copy()
        logger.debug(f"Loaded {len(df_params)} parameter records")

    if len(df_params) == 0:
        raise ValueError("No parameter-level records found in hook statistics CSV")

    unique_params = df_params['module'].unique()
    log_metric(logger, "Unique parameters", len(unique_params))

    numeric_cols = df_params.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['step', 't', 'epoch']]

    agg_funcs = ['mean', 'std']
    agg_df = df_params.groupby('module')[numeric_cols].agg(agg_funcs).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                      for col in agg_df.columns.values]
    agg_df = agg_df.rename(columns={'module': 'parameter_name'})

    logger.info(f"Aggregated features: {len(agg_df)} parameters × {len(agg_df.columns) - 1} features")

    param_features = pd.DataFrame({'parameter_name': agg_df['parameter_name']})

    missing_features = []
    available_features = []

    for feat in feature_names:
        if feat in agg_df.columns:
            param_features[feat] = agg_df[feat]
            available_features.append(feat)
        else:
            feat_with_suffix = f"{feat}_mean"
            if feat_with_suffix in agg_df.columns:
                param_features[feat] = agg_df[feat_with_suffix]
                available_features.append(feat)
            elif feat == 'epoch':
                param_features[feat] = 0.0
                available_features.append(feat)
            elif feat == 'step':
                param_features[feat] = agg_df.get('step_mean', 0.0)
                available_features.append(feat)
            elif feat.startswith('act_'):
                stat_name = feat.replace('act_', '')
                grad_col = f'grad_{stat_name}_mean'
                param_col = f'param_{stat_name}_mean'
                if grad_col in agg_df.columns:
                    param_features[feat] = agg_df[grad_col]
                    available_features.append(feat)
                elif param_col in agg_df.columns:
                    param_features[feat] = agg_df[param_col]
                    available_features.append(feat)
                else:
                    param_features[feat] = 0.0
                    missing_features.append(feat)
            elif feat.endswith('_epoch_mean') or feat.endswith('_epoch_std'):
                base_stat = feat.replace('_epoch_mean', '').replace('_epoch_std', '')
                stat_name = base_stat.replace('act_', '')
                if feat.endswith('_epoch_mean'):
                    grad_col = f'grad_{stat_name}_mean'
                    param_col = f'param_{stat_name}_mean'
                else:
                    grad_col = f'grad_{stat_name}_std'
                    param_col = f'param_{stat_name}_std'
                if grad_col in agg_df.columns:
                    param_features[feat] = agg_df[grad_col]
                    available_features.append(feat)
                elif param_col in agg_df.columns:
                    param_features[feat] = agg_df[param_col]
                    available_features.append(feat)
                else:
                    param_features[feat] = 0.0
                    missing_features.append(feat)
            else:
                param_features[feat] = 0.0
                missing_features.append(feat)

    logger.info(f"Matched {len(available_features)}/{len(feature_names)} features from hook stats")
    if missing_features:
        logger.warning(f"{len(missing_features)} features not found, filled with zeros")
        if len(missing_features) <= 10:
            logger.debug(f"Missing: {missing_features}")

    logger.info(f"Final feature matrix: {len(param_features)} parameters × {len(param_features.columns) - 1} features")

    return param_features


def compute_importance_scores(
        pipeline: Any,
        param_features: pd.DataFrame,
        feature_names: list
) -> pd.DataFrame:
    """Compute importance scores for each parameter using meta-predictor."""
    logger.info("Computing importance scores...")

    available_features = []
    for feat in feature_names:
        if feat in param_features.columns and param_features[feat].abs().sum() > 0:
            available_features.append(feat)

    logger.debug(f"Using {len(available_features)}/{len(feature_names)} non-zero features")

    if len(available_features) == 0:
        logger.warning("No non-zero features available! Using all features with potential zeros.")
        available_features = feature_names

    try:
        X = param_features[available_features].values
        importance_scores = pipeline.predict(X)
    except Exception as e:
        logger.warning(f"Error with {len(available_features)} features: {e}")
        logger.info("Trying with base act_* features only...")

        base_features = ['epoch', 'step'] + [f'act_{stat}' for stat in
                                             ['mean', 'std', 'min', 'max', 'l2', 'l1', 'sparsity', 'q25', 'q50', 'q75']]
        base_features = [f for f in base_features if f in param_features.columns]

        logger.debug(f"Using {len(base_features)} base features")
        X = param_features[base_features].values
        importance_scores = pipeline.predict(X)

    result = pd.DataFrame({
        "parameter_name": param_features["parameter_name"],
        "importance_score": importance_scores
    })

    logger.info(f"Score range: [{importance_scores.min():.4f}, {importance_scores.max():.4f}], "
                f"mean={importance_scores.mean():.4f}")

    return result


def rank_and_partition_parameters(
        importance_df: pd.DataFrame,
        zones_config: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Rank parameters by importance and partition into compression zones."""
    df = importance_df.copy()
    df = df.sort_values('importance_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    scores = df['importance_score'].values
    df['quantile'] = df['importance_score'].rank(pct=True)

    log_metric(logger, "Ranked parameters", len(df))
    logger.info(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    zones = {}
    for zone_name, zone_cfg in zones_config.items():
        zones[zone_name] = {
            'min': float(zone_cfg.get('quantile_min', 0.0)),
            'max': float(zone_cfg.get('quantile_max', 1.0)),
            'action': zone_cfg.get('action', 'keep'),
            'config': zone_cfg
        }

    def assign_zone(q: float) -> str:
        for zone_name, zone_info in zones.items():
            if zone_info['min'] <= q <= zone_info['max']:
                return zone_name
        return 'medium'

    df['zone'] = df['quantile'].apply(assign_zone)
    _zone_to_action = {k: v['action'] for k, v in zones.items()}
    df['action'] = df['zone'].map(_zone_to_action).fillna('keep')

    def get_target_bits(zone_name: str) -> Optional[int]:
        zone_cfg = zones.get(zone_name, {}).get('config', {})
        return zone_cfg.get('bits', None)

    df['target_bits'] = df['zone'].apply(get_target_bits)

    log_section(logger, "Zone assignment")
    for zone_name in ['high', 'medium', 'low']:
        if zone_name in zones:
            zone_df = df[df['zone'] == zone_name]
            if len(zone_df) > 0:
                zone_cfg = zones[zone_name]
                bits_info = f", bits={zone_cfg.get('bits', 'rank-aware')}" if zone_cfg['action'] == 'quantize' else ""
                logger.info(f"{zone_name.upper():8s} ({zone_cfg['min']:.0%}-{zone_cfg['max']:.0%}): "
                            f"{len(zone_df)} parameters, action={zone_cfg['action']}{bits_info}")
                logger.info(f"         Score range: [{zone_df['importance_score'].min():.2f}, "
                            f"{zone_df['importance_score'].max():.2f}]")

    total_assigned = len(df)
    for zone_name in zones.keys():
        count = len(df[df['zone'] == zone_name])
        logger.debug(f"{zone_name}: {count}/{total_assigned} parameters ({count / total_assigned * 100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_compression(cfg: Dict[str, Any]) -> int:
    """Main compression pipeline orchestrator.
    
    Loads default configuration from strategy_defaults.yaml and merges with user config.
    User-provided values always take precedence over defaults.
    """
    # Load default configuration
    defaults = _load_strategy_defaults()
    default_compression = defaults.get("compression", {})
    
    # Merge user config with defaults (user must take precedence)
    compression_cfg = _merge_with_defaults(cfg.get("compression", {}), default_compression)
    
    # Extract configuration values (now from merged config)
    target_model = compression_cfg.get("target_model")
    baseline_model_config = compression_cfg.get("baseline_model_config")
    if baseline_model_config is None:
        baseline_model_config = target_model
    
    output_dir = compression_cfg.get("output_dir")
    if output_dir is None:
        output_dir = f"{target_model}/models/experiments/variant_test"
    
    meta_checkpoint = compression_cfg.get("meta_checkpoint")
    zones_config = compression_cfg.get("zones", {})
    quantization_cfg = compression_cfg.get("quantization", {})
    
    log_phase_header(logger, "1: Load Meta-Predictor & Compute Importance")

    # Step 1: Load meta-predictor
    log_section(logger, "Load meta-predictor checkpoint")
    try:
        model, imputer, scaler, feature_names, target_col, task_type = load_meta_predictor_checkpoint(meta_checkpoint)
        if imputer is not None and scaler is not None:
            class PreprocessedModel:
                def __init__(self, model, imputer, scaler):
                    self.model = model
                    self.imputer = imputer
                    self.scaler = scaler

                def predict(self, X):
                    import torch as _torch
                    X_imputed = self.imputer.transform(X)
                    X_scaled = self.scaler.transform(X_imputed)
                    if isinstance(self.model, _torch.nn.Module):
                        X_tensor = _torch.from_numpy(X_scaled).float()
                        with _torch.no_grad():
                            return self.model(X_tensor).cpu().numpy()
                    else:
                        return self.model.predict(X_scaled)

                def __call__(self, X):
                    return self.predict(X)

            pipeline = PreprocessedModel(model, imputer, scaler)
        else:
            pipeline = model
    except Exception as e:
        logger.error(f"Failed to load meta-predictor: {e}")
        return 1

    # Step 2: Extract features
    log_section(logger, "Extract parameter features")
    try:
        param_features = extract_parameter_features(target_model, feature_names)
    except Exception as e:
        logger.error(f"Failed to extract parameter features: {e}")
        return 1

    # Step 3: Compute importance scores
    log_section(logger, "Compute importance scores")
    try:
        importance_df = compute_importance_scores(pipeline, param_features, feature_names)
    except Exception as e:
        logger.error(f"Failed to compute importance scores: {e}")
        return 1

    # Save importance
    log_section(logger, "Save importance scores")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    importance_path = output_path / "parameter_importance_scores.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved importance scores to: {importance_path}")

    # Phase 2: Rank and zone
    log_phase_header(logger, "2: Rank & Partition into Zones")
    try:
        ranked_df = rank_and_partition_parameters(importance_df, zones_config)
    except Exception as e:
        logger.error(f"Failed to rank and partition parameters: {e}")
        return 1

    ranked_path = output_path / "parameter_zones.csv"
    ranked_df.to_csv(ranked_path, index=False)
    logger.info(f"Saved zone assignments to: {ranked_path}")

    # Phase 3: Apply compression
    log_phase_header(logger, "3: Apply Zone-Specific Compression")

    # Load target model
    try:
        model = load_target_model(target_model)
    except Exception as e:
        logger.error(f"Failed to load target model: {e}")
        return 1

    # EVAL: Baseline size
    log_section(logger, "Baseline Model")
    baseline_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    log_metric(logger, "Size", f"{baseline_size_mb:.2f}", "MB")
    log_metric(logger, "Parameters", sum(p.numel() for p in model.parameters()))

    # Map simplified to full names (best-effort)
    try:
        from safetensors.torch import load_file
        model_state_path = Path(target_model) / "model.safetensors"
        if model_state_path.exists():
            state_dict = load_file(str(model_state_path))
            param_names = sorted(state_dict.keys())

            def map_simple_to_full(simple_name: str):
                parts = simple_name.split('.')
                if len(parts) != 2:
                    return simple_name
                idx_str, suffix = parts
                try:
                    idx = int(idx_str)
                except ValueError:
                    return simple_name
                suffix_params = [n for n in param_names if n.endswith('.' + suffix)]
                if 0 <= idx < len(suffix_params):
                    return suffix_params[idx]
                return simple_name

            ranked_df['full_parameter_name'] = ranked_df['parameter_name'].apply(map_simple_to_full)
            logger.info(f"Mapped {len(ranked_df)} parameters to full names")
        else:
            logger.info(f"WARNING: Model state dict not found, using simplified names")
    except Exception as e:
        logger.info(f"WARNING: Failed to map parameter names: {e}, using simplified names")

    if 'full_parameter_name' in ranked_df.columns:
        logger.info(f"Using full parameter names from mapped zones")
        key_col = 'full_parameter_name'
    else:
        logger.info(f"Using simplified parameter names")
        key_col = 'parameter_name'

    plan = ranked_df.set_index(key_col)['action'].to_dict()
    importance_rankings = ranked_df.set_index(key_col)['importance_score'].to_dict()
    target_bits_map = ranked_df.set_index(key_col)['target_bits'].to_dict()

    quantize_params = ranked_df[ranked_df['action'] == 'quantize']
    if len(quantize_params) > 0:
        min_score = quantize_params['importance_score'].min()
        max_score = quantize_params['importance_score'].max()
        if max_score > min_score:
            for param_name in quantize_params[key_col]:
                score = importance_rankings[param_name]
                importance_rankings[param_name] = (score - min_score) / (max_score - min_score)
        else:
            for param_name in quantize_params[key_col]:
                importance_rankings[param_name] = 0.5

    quant_config = None
    quantizer = None
    if quantization_cfg.get('enabled', True):
        quant_config = QuantizationConfig(quantization_cfg)
        quantizer = Quantizer(quant_config)

    # STEP 1: Pruning
    pruning_meta = {}
    pruning_cfg = cfg.get('compression', {}).get('pruning', {})
    if pruning_cfg.get('enabled', False):
        try:
            from .pruning import PruningConfig, TransformerPruner, save_pruning_metadata
            prune_config = PruningConfig(pruning_cfg)
            pruner = TransformerPruner(prune_config)

            logger.info(f"Step 1: Applying structured pruning to prune zone...")
            logger.info(f"Method: {prune_config.method}")
            logger.info(f"Head pruning ratio: {prune_config.head_pruning_ratio:.1%}")
            logger.info(f"FFN pruning ratio: {prune_config.ffn_pruning_ratio:.1%}")

            pruning_meta = pruner.apply_pruning(model, plan, importance_rankings)

            if pruning_meta:
                prune_meta_dir = output_path / "compressed"
                prune_meta_dir.mkdir(parents=True, exist_ok=True)
                save_pruning_metadata(pruning_meta, prune_meta_dir)

            logger.info(f"Pruned {pruning_meta.get('heads_pruned', 0)} heads and "
                        f"{pruning_meta.get('neurons_pruned', 0)} neurons")

            log_section(logger, "After Pruning")
            after_pruning_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            log_metric(logger, "Size", f"{after_pruning_size_mb:.2f}", "MB")
            non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
            log_metric(logger, "Non-zero parameters", non_zero_params)
            sparsity = 1.0 - (non_zero_params / sum(p.numel() for p in model.parameters()))
            log_metric(logger, "Sparsity", f"{sparsity:.2%}")
        except Exception as e:
            logger.info(f"ERROR: Failed to apply pruning: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        logger.info(f"Pruning disabled in config, skipping Step 1")

    # STEP 2: Rank-aware quantization (MOVED BEFORE FINE-TUNING)
    quant_meta = {}
    if quantizer is not None:
        try:
            logger.info(f"Step 2: Applying rank-aware quantization to quantize zone...")
            logger.info(f"Mode: {quant_config.mode}")
            logger.info(f"Bits range: [{quant_config.bits_lower}, {quant_config.bits_upper}]")
            logger.info(f"Utilization target: {quant_config.util_target}")
            logger.info(f"Per-channel: {quant_config.per_channel}")

            quant_meta = quantizer.apply_quantization(model, plan, importance_rankings, target_bits_map)

            logger.info(f"Quantized {len(quant_meta)} parameters in quantize zone")
            
            log_section(logger, "After Quantization")
            after_quant_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            log_metric(logger, "Size (FP32)", f"{after_quant_size_mb:.2f}", "MB")
            logger.info("Variable-bit size will be computed during export")
        except Exception as e:
            logger.info(f"ERROR: Failed to apply quantization: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        logger.info(f"Quantization disabled in config, skipping Step 2")

    # STEP 3: Fine-tuning after quantization (optional, for recovery)
    fine_tuning_cfg = compression_cfg.get("fine_tuning", {})
    ft_results = None
    physical_pruning_active = pruning_cfg.get('enabled', False) and pruning_cfg.get('physical', True)

    if fine_tuning_cfg.get("enabled", False) and quantizer is not None:
        log_phase_header(logger, "4: Fine-tuning quantized model (recovery)")

        if physical_pruning_active:
            logger.warning(f"Auto fine-tuning not fully supported with physical pruning")
            logger.info(f"Reason: Layer-wise heterogeneous architecture")
            logger.info(f"Solution: Use soft pruning (physical: false) for auto fine-tuning")
            logger.info(f"Skipping fine-tuning...")
        else:
            try:
                from metapac.src.compression.fine_tune import run_fine_tuning

                # Save quantized model for fine-tuning
                quantized_dir = output_path / "quantized_before_ft"
                quantized_dir.mkdir(parents=True, exist_ok=True)

                quantized_model_path = quantized_dir / "pytorch_model.bin"
                torch.save(model.state_dict(), quantized_model_path)

                import shutil
                baseline_config = Path(baseline_model_config) / "config.json"
                if baseline_config.exists():
                    shutil.copy(baseline_config, quantized_dir / "config.json")
                    logger.info(f"Copied config.json from baseline model")

                compression_summary = {
                    'target_model': compression_cfg.get('target_model', ''),
                    'pruning_applied': pruning_cfg.get('enabled', False),
                    'quantization_applied': True
                }
                with open(quantized_dir / "compression_summary.json", 'w') as f:
                    json.dump(compression_summary, f, indent=2)

                logger.info(f"Saved quantized model to: {quantized_model_path}")

                ft_config = {
                    'model_checkpoint': str(quantized_dir),
                    'output_dir': fine_tuning_cfg.get('output_dir', f"{output_dir}/finetuned"),
                    'data': fine_tuning_cfg.get('data', {
                        'dataset': 'glue',
                        'dataset_config': 'sst2',
                        'max_length': 128,
                        'batch_size': 8
                    }),
                    'training': fine_tuning_cfg.get('training', {
                        'num_epochs': 3,
                        'learning_rate': 2e-5,
                        'weight_decay': 0.01,
                        'warmup_ratio': 0.1,
                        'gradient_clip': 1.0,
                        'device': 'cuda',
                        'num_workers': 0
                    }),
                    'distillation': fine_tuning_cfg.get('distillation', {
                        'enabled': False,
                        'teacher_model': baseline_model_config,
                        'temperature': 4.0,
                        'alpha': 0.5
                    })
                }

                logger.info(f"Model checkpoint: {quantized_dir}")
                logger.info(f"Output directory: {ft_config['output_dir']}")
                logger.info(f"Epochs: {ft_config['training']['num_epochs']}")
                logger.info(f"Learning rate: {ft_config['training']['learning_rate']}")
                logger.info(f"NOTE: Fine-tuning will update quantized FP32 weights (not QuantizedLinear layers)")

                result = run_fine_tuning(ft_config)

                if result == 0:
                    finetuned_model_path = Path(ft_config['output_dir']) / "pytorch_model.bin"
                    if not finetuned_model_path.exists():
                        finetuned_model_path = Path(ft_config['output_dir']) / "model.safetensors"

                    if finetuned_model_path.exists():
                        logger.info(f"Fine-tuned model saved to: {finetuned_model_path}")

                        results_path = Path(ft_config['output_dir']) / "fine_tune_results.json"
                        if results_path.exists():
                            with open(results_path) as f:
                                ft_results = json.load(f)
                            logger.info(f"Fine-tuning metrics:")
                            logger.info(f"Best accuracy: {ft_results['best_val_accuracy']:.4f}")
                            logger.info(f"Final accuracy: {ft_results['final_val_accuracy']:.4f}")

                        # CRITICAL: Load fine-tuned weights back into model for export
                        logger.info(f"Loading fine-tuned weights into model...")
                        if finetuned_model_path.suffix == '.safetensors':
                            from safetensors.torch import load_file
                            finetuned_state = load_file(finetuned_model_path, device='cpu')
                        else:
                            finetuned_state = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
                        model.load_state_dict(finetuned_state, strict=False)
                        logger.info(f"✓ Fine-tuned weights loaded successfully")

                        # CRITICAL: Re-quantize after fine-tuning to update _q_int values
                        # Fine-tuning changed the weights, so we need fresh quantization
                        logger.info(f"Re-quantizing model after fine-tuning to sync _q_int values...")
                        
                        for name, param in model.named_parameters():
                            if name in quant_meta:
                                qmeta = quant_meta[name]
                                bits = qmeta['bits_final']
                                scale = qmeta['scale']
                                symmetric = qmeta.get('symmetric', True)
                                per_channel = qmeta.get('per_channel', False)
                                zero_point = qmeta.get('zero_point', None)
                                
                                # Quantize with SAME scale as before (preserve scale, update _q_int)
                                if per_channel and isinstance(scale, list):
                                    scale_tensor = torch.tensor(scale, device=param.device, dtype=param.dtype)
                                else:
                                    scale_tensor = torch.tensor([scale], device=param.device, dtype=param.dtype).squeeze()
                                
                                # Quantize (INT values)
                                if symmetric:
                                    qmax = 2 ** (bits - 1) - 1
                                    q_int = torch.round(param.data / scale_tensor).clamp(-qmax - 1, qmax)
                                else:
                                    qmax = 2 ** bits - 1
                                    if zero_point is not None:
                                        if isinstance(zero_point, list):
                                            zp_tensor = torch.tensor(zero_point, device=param.device, dtype=param.dtype)
                                        else:
                                            zp_tensor = torch.tensor([zero_point], device=param.device, dtype=param.dtype).squeeze()
                                        q_int = torch.round(param.data / scale_tensor + zp_tensor).clamp(0, qmax)
                                    else:
                                        q_int = torch.round(param.data / scale_tensor).clamp(0, qmax)
                                
                                # Dequantize (fake-quant FP32)
                                if symmetric:
                                    param.data = (q_int * scale_tensor).to(param.dtype)
                                else:
                                    if zero_point is not None:
                                        param.data = ((q_int - zp_tensor) * scale_tensor).to(param.dtype)
                                    else:
                                        param.data = (q_int * scale_tensor).to(param.dtype)
                                
                                # Update _q_int in metadata
                                quant_meta[name]['_q_int'] = q_int.cpu()
                                
                        logger.info(f"✓ Re-quantization complete, _q_int values updated in metadata")

                        log_section(logger, "After Fine-Tuning (model still quantized)")
                        after_ft_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                        log_metric(logger, "Size", f"{after_ft_size_mb:.2f}", "MB")
                        non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
                        log_metric(logger, "Non-zero parameters", non_zero_params)
                    else:
                        logger.warning(f"Fine-tuned model not found at {finetuned_model_path}")
                else:
                    logger.warning(f"Fine-tuning failed, continuing with quantized weights...")
            except Exception as e:
                logger.warning(f"Fine-tuning failed: {e}")
                import traceback
                traceback.print_exc()
                logger.info(f"Continuing with quantized weights...")
    elif fine_tuning_cfg.get("enabled", False) and quantizer is None:
        logger.info(f"Fine-tuning enabled but quantization disabled - fine-tuning will be skipped")

    # STEP 4: Headroom trimming
    trim_meta = {}
    if quantizer is not None:
        try:
            logger.info(f"Step 4: Applying headroom trimming to ALL zones...")
            logger.info(f"Target utilization: {quant_config.util_target}")
            logger.info(f"Headroom min bits: {quant_config.headroom_min_bits}")
            logger.info(f"This optimizes bit allocation for keep/quantize/prune zones")

            trim_meta = quantizer.apply_headroom_trimming_all_zones(
                model, plan, target_bits_map, importance_rankings=importance_rankings
            )

            zone_savings = {'keep': 0, 'quantize': 0, 'prune': 0}
            for param_name, meta in trim_meta.items():
                zone = meta.get('zone', 'keep')
                if zone not in zone_savings:
                    zone = 'keep'
                if meta.get('bits_saved', 0) >= 2:
                    zone_savings[zone] += 1

            logger.info(f"Headroom trimming complete:")
            logger.info(f"Keep zone: {zone_savings['keep']} params with >=2 bits saved")
            logger.info(f"Quantize zone: {zone_savings['quantize']} params with >=2 bits saved")
            logger.info(f"Prune zone: {zone_savings['prune']} params with >=2 bits saved")

            log_section(logger, "After Quantization & Headroom Trimming")
            after_quant_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            log_metric(logger, "Size (FP32)", f"{after_quant_size_mb:.2f}", "MB")
            logger.info("INT8 size will be computed during export")
            logger.info(f"(INT8 size will be computed during export)")
        except Exception as e:
            logger.info(f"ERROR: Failed to apply headroom trimming: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        logger.info(f"Quantization disabled in config, skipping Step 4 (headroom trimming)")

    # Phase 4: Save compressed model and metadata
    log_phase_header(logger, "5: Save Compressed Model")

    compressed_dir = output_path / "compressed"
    compressed_dir.mkdir(parents=True, exist_ok=True)

    try:
        quantization_cfg.get('export_int', False)
        export_variable_bit = quantization_cfg.get('export_variable_bit', True)  # Variable-bit export works well!
        export_packed = quantization_cfg.get('export_packed', False)

        int8_stats = None
        variable_bit_stats = None
        packed_stats = None

        # Merge quant_meta and trim_meta for export
        # IMPORTANT: Deep merge to preserve scale/zero_point from quant_meta
        combined_meta = {}
        for name in set(list(quant_meta.keys()) + list(trim_meta.keys())):
            if name in quant_meta and name in trim_meta:
                # Merge: start with quant_meta (has scale/zero_point), update with trim_meta (has bits_final)
                combined_meta[name] = {**quant_meta[name], **trim_meta[name]}
            elif name in quant_meta:
                combined_meta[name] = quant_meta[name]
            else:
                combined_meta[name] = trim_meta[name]

        # NEW: Build variable-bit registry from combined_meta so exporter can iterate safely
        try:
            registered_count = _build_variable_bit_registry_from_meta(
                model,
                combined_meta,
                exclude_layernorm_and_classifier=True,
                fallback_bits=quantization_cfg.get('bits_upper', 8),
            )
            logger.info(f"[variable-bit] Registered {registered_count} quantized layers for export")
        except Exception as e:
            logger.info(f"[variable-bit] WARNING: Failed to build registry from meta: {e}")

        if export_variable_bit and combined_meta:
            from metapac.src.compression.variable_bit_export import integrate_variable_bit_export

            logger.info(f"Exporting with variable bit-width quantization...")
            logger.info(f"This provides TRUE runtime memory savings (4-16x compression)")

            variable_bit_stats = integrate_variable_bit_export(
                model,
                combined_meta,
                compressed_dir,
                export_variable_bit=True,
                use_cuda=torch.cuda.is_available(),
                source_model_path=baseline_model_config
            )


        if export_packed and trim_meta:
            from metapac.src.compression.bitpacking import save_packed_model

            logger.info(f"Exporting with variable-bit packing...")
            logger.info(f"Note: This is for disk storage optimization (saves disk space)")
            logger.info(f"For inference, use pytorch_model.bin (HF-compatible variable-bit model)")

            packed_stats = save_packed_model(
                model.state_dict(),
                trim_meta,
                quant_meta,
                compressed_dir
            )

            logger.info(f"Variable-bit packing complete:")
            logger.info(f"Total parameters: {packed_stats['total_params']:,}")
            logger.info(
                f"Packed (variable-bit): {packed_stats['packed_params']:,} ({100 * packed_stats['packed_params'] / packed_stats['total_params']:.1f}%)")
            logger.info(
                f"FP32 (unpacked): {packed_stats['fp32_params']:,} ({100 * packed_stats['fp32_params'] / packed_stats['total_params']:.1f}%)")
            logger.info(f"Original size: {packed_stats['total_original_size'] / (1024 * 1024):.2f} MB")
            logger.info(f"Packed size: {packed_stats['size_mb']:.2f} MB")
            logger.info(f"Compression ratio: {packed_stats['compression_ratio']:.2f}x")
            logger.info(f"Use scripts/pack_model.py for standalone packing")

        if quant_meta:
            save_quantization_metadata(quant_meta, compressed_dir)

        if trim_meta:
            trim_meta_path = compressed_dir / "headroom_trim_meta.json"
            import numpy as _np

            def _convert_types(obj):
                try:
                    import torch as _torch
                except Exception:
                    _torch = None

                if isinstance(obj, dict):
                    return {k: _convert_types(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_convert_types(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_convert_types(v) for v in obj)
                if _torch is not None and isinstance(obj, _torch.Tensor):
                    try:
                        arr = obj.detach().cpu()
                        if arr.numel() == 1:
                            return arr.item()
                        return arr.tolist()
                    except Exception:
                        return None
                if isinstance(obj, _np.generic):
                    return obj.item()
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, float):
                    return float(obj)
                if isinstance(obj, (str, int, bool)) or obj is None:
                    return obj
                try:
                    return float(obj)
                except Exception:
                    try:
                        return str(obj)
                    except Exception:
                        return None

            converted = _convert_types(trim_meta)
            with open(trim_meta_path, 'w', encoding='utf-8') as f:
                json.dump(converted, f, indent=2)
            logger.info(f"Saved headroom trimming metadata to: {trim_meta_path}")

        if pruning_cfg.get('enabled') and pruning_cfg.get('physical', True):
            logger.info(f"WARNING: Physical pruning active - layer-wise heterogeneous architecture")
            logger.info(f"Fine-tuning with physical pruning requires manual model reconstruction")

        summary = {
            'target_model': target_model,
            'meta_checkpoint': meta_checkpoint,
            'total_parameters': len(ranked_df),
            'physical_pruning': pruning_cfg.get('physical', False),
            'zones': {
                zone: {
                    'count': len(ranked_df[ranked_df['zone'] == zone]),
                    'action': zones_config.get(zone, {}).get('action', 'keep')
                }
                for zone in ['high', 'medium', 'low']
            },
            'quantization': {
                'enabled': quantization_cfg.get('enabled', True),
                'quantized_parameters': len(quant_meta),
                'bits_range': [quantization_cfg.get('bits_lower', 4), quantization_cfg.get('bits_upper', 8)],
                'util_target': quantization_cfg.get('util_target', 0.98),
                'export_int': quantization_cfg.get('export_int', False)
            },
            'pruning': {
                'enabled': pruning_cfg.get('enabled', False),
                'heads_pruned': pruning_meta.get('heads_pruned', 0),
                'neurons_pruned': pruning_meta.get('neurons_pruned', 0),
                'total_heads_before': pruning_meta.get('total_heads_before', 0),
                'total_neurons_before': pruning_meta.get('total_neurons_before', 0),
                'compression_ratio': pruning_meta.get('compression_ratio', 1.0)
            },
            'headroom_trimming': {
                'enabled': quantizer is not None,
                'total_parameters': len(trim_meta),
                'zone_stats': {
                    zone: {
                        'params': sum(1 for m in trim_meta.values() if m.get('zone', 'keep') == zone),
                        'significant_savings': sum(1 for m in trim_meta.values()
                                                   if m.get('zone', 'keep') == zone and m.get('bits_saved', 0) >= 2),
                        'total_bits_saved': sum(m.get('bits_saved', 0) for m in trim_meta.values()
                                                if m.get('zone', 'keep') == zone)
                    }
                    for zone in ['keep', 'quantize', 'prune']
                }
            }
        }

        if variable_bit_stats:
            summary['variable_bit_quantization'] = variable_bit_stats
        if int8_stats:
            summary['int8_compression'] = int8_stats
        if packed_stats:
            summary['variable_bit_packing'] = packed_stats

        summary_path = compressed_dir / "compression_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved compression summary to: {summary_path}")

    except Exception as e:
        logger.info(f"ERROR: Failed to save compressed model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finalize_artifacts(
        experiment_dir=output_path,
        keep_tokenizer=True,  # set False if tokenizer/config loaded from HF or baseline
        primary_weight="pytorch_model.bin",  # HF-compatible variable-bit quantized model
        dry_run=False
    )

    # Optional Phase 5: Validation
    validation_cfg = compression_cfg.get("validation", {})
    validation_results = None

    if validation_cfg.get("enabled", False):
        logger.info(f"========================================")
        logger.info(f"Phase 5: Validating Compressed Model")
        logger.info(f"========================================")

        try:
            model_to_validate = compressed_dir

            # Pre-validation: detect presence of exported weights; if absent, fall back to in-memory model
            compressed_weight_paths = [
                compressed_dir / 'model_int8.pt',
                compressed_dir / 'pytorch_model.bin',
                compressed_dir / 'model.safetensors',
            ]

            has_exported_weights = any(p.exists() for p in compressed_weight_paths)
            if not has_exported_weights:
                logger.warning("=" * 60)
                logger.warning("No exported weights found in compressed directory; falling back to in-memory subject model for validation.")
                logger.warning(f"Checked paths:")
                for p in compressed_weight_paths:
                    try:
                        rel = p.relative_to(output_path)
                    except Exception:
                        rel = p
                    logger.warning(f"  - {rel}: {'EXISTS' if p.exists() else 'MISSING'}")
                logger.warning("This is expected for pruning-only runs or when export is disabled.")
                logger.warning("=" * 60)

            logger.info(f"Validating compressed model (after fine-tuning, if applicable)")

            # If baseline_model is None or empty, fall back to target_model
            baseline_model = validation_cfg.get("baseline_model") or target_model

            logger.info(f"Compressed model: {model_to_validate}")
            logger.info(f"Baseline model:   {baseline_model}")
            logger.info(
                f"Dataset:          {validation_cfg.get('dataset', 'glue')}/{validation_cfg.get('dataset_config', 'sst2')}"
            )

            runtime_pref = validation_cfg.get("runtime") or validation_cfg.get("device") or "auto"
            batch_size = validation_cfg.get("batch_size", 32)
            max_length = validation_cfg.get("max_length", 128)
            dataset_name = validation_cfg.get("dataset", "glue")
            dataset_config = validation_cfg.get("dataset_config", "sst2")
            split = validation_cfg.get("split", "validation")
            warmup_steps = validation_cfg.get("warmup_steps", 2)
            disable_progress = validation_cfg.get("disable_progress", False)
            max_samples = validation_cfg.get("max_samples")
            pre_dequant_override = validation_cfg.get("pre_dequantize")

            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            physical_pruning_active = pruning_cfg.get('enabled', False) and pruning_cfg.get('physical', True)
            pruning_meta_exists = (compressed_dir / 'pruning_meta.json').exists()
            # Only use custom loader if we truly have physical pruning AND the required metadata exists
            use_custom_loader = physical_pruning_active and has_exported_weights and pruning_meta_exists

            if has_exported_weights:
                try:
                    if use_custom_loader:
                        logger.info("Detected physical pruning - using custom loader")
                        from metapac.src.compression.load_pruned import load_physically_pruned_model

                        compressed_model, subject_tokenizer, _ = load_physically_pruned_model(
                            str(model_to_validate),
                            device=runtime_pref,
                        )
                    else:
                        logger.info("Using quantized model loader")
                        from metapac.src.compression.load_quantized_model import load_quantized_distilbert

                        compressed_model = load_quantized_distilbert(
                            str(model_to_validate),
                            device=runtime_pref,
                            config_path=baseline_model,
                        )
                        subject_tokenizer = AutoTokenizer.from_pretrained(baseline_model)
                except Exception as load_err:
                    logger.warning(f"Failed to load exported model: {load_err}")
                    logger.warning("Falling back to in-memory subject model for validation")
                    compressed_model = model
                    compressed_model.eval()
                    subject_tokenizer = AutoTokenizer.from_pretrained(baseline_model)
            else:
                # Fallback: validate the current in-memory compressed model (e.g., pruning-only)
                logger.info("Using in-memory subject model for validation")
                compressed_model = model
                compressed_model.eval()
                subject_tokenizer = AutoTokenizer.from_pretrained(baseline_model)

            logger.info("Loading baseline model...")
            baseline_model_obj = AutoModelForSequenceClassification.from_pretrained(baseline_model)
            baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model)
            baseline_model_obj.eval()

            from metapac.src.compression.validate import validate_model

            validation_results = validate_model(
                subject_model=compressed_model,
                subject_tokenizer=subject_tokenizer,
                dataset=dataset_name,
                dataset_config=dataset_config,
                split=split,
                batch_size=batch_size,
                max_length=max_length,
                runtime=runtime_pref,
                warmup_steps=warmup_steps,
                disable_progress=disable_progress,
                pre_dequantize=pre_dequant_override,
                max_samples=max_samples,
                subject_name="compressed",
                subject_path=str(model_to_validate),
                baseline_model=baseline_model_obj,
                baseline_tokenizer=baseline_tokenizer,
                baseline_name="baseline",
                baseline_path=str(baseline_model),
                allow_runtime_fallback=validation_cfg.get("allow_runtime_fallback", True),
            )

            compressed_info = validation_results["subject"]
            baseline_info = validation_results.get("baseline", {})
            comparison_info = validation_results.get("comparison", {})

            logger.info("Results:")
            logger.info(
                "Compressed accuracy:  %.4f (%.2f%%)",
                compressed_info["accuracy"],
                compressed_info["accuracy"] * 100,
            )
            if baseline_info:
                logger.info(
                    "Baseline accuracy:    %.4f (%.2f%%)",
                    baseline_info["accuracy"],
                    baseline_info["accuracy"] * 100,
                )
            if comparison_info:
                logger.info(
                    "Accuracy drop:        %.4f (%.2f%%)",
                    comparison_info.get("accuracy_drop", 0.0),
                    comparison_info.get("accuracy_drop_pct", 0.0),
                )

            if validation_cfg.get("save_results", True):
                results_path = compressed_dir / "validation_results.json"
                with open(results_path, "w") as f:
                    json.dump(validation_results, f, indent=2)
                logger.info(f"Saved results to: {results_path}")



        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            import traceback
            traceback.print_exc()

    # strategy.py (run_strategy) – at the very end, before return
    from metapac.src.utils.experiment_report import generate_experiment_report

    # choose experiment root as the parent of export_dir (e.g., .../experiments/kd_test_v4)
    report_path = generate_experiment_report(output_dir, include_hashes=False)
    print(f"[report] wrote experiment report to: {report_path}")

    logger.info(f"Compression pipeline completed")
    logger.info(f"========================================")

    return 0


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------

def load_target_model(model_path: str) -> nn.Module:
    """Load target model for compression."""
    logger.info(f"Loading target model from: {model_path}")

    model_dir = Path(model_path)

    try:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        logger.info(f"Loaded transformers model: {model.__class__.__name__}")
        return model
    except Exception as e:
        logger.info(f"Could not load as transformers model: {e}")
        logger.info(f"Falling back to state dict loading...")

    ckpt_path = model_dir / "model.safetensors"
    if ckpt_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
        logger.info(f"Loaded safetensors state dict with {len(state_dict)} parameters")
    else:
        ckpt_path = model_dir / "pytorch_model.bin"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "model.pt"

        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location='cpu')
            logger.info(f"Loaded state dict with {len(state_dict)} parameters")
        else:
            state_dict = None

    if state_dict is not None:

        class DummyModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                for name, param in state_dict.items():
                    if param.dim() == 0:
                        self._buffers[name] = param
                    else:
                        self._parameters[name] = nn.Parameter(param)

            def forward(self, x):
                return x

        model = DummyModel(state_dict)
        return model
    else:
        logger.info(f"Warning: Model checkpoint not found, creating dummy model")
        model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        return model


def make_json_serializable(data):
    """Recursively convert data to JSON-serializable types."""
    if np.issubdtype(type(data), np.floating):
        return float(data)
    elif np.issubdtype(type(data), np.integer):
        return int(data)
    elif isinstance(data, dict):
        return {key: make_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(make_json_serializable(item) for item in data)
    elif isinstance(data, (str, bool, type(None))):
        return data
    else:
        return str(data)  # Fallback for unsupported types
