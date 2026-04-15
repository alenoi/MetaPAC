"""Meta-predictor model for estimating layer importance and performance metrics.

This module provides a PyTorch-based neural network that predicts target metrics
(e.g., gradient importance, performance impact) from layer activation statistics.
It supports both regression and classification tasks with configurable architecture.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.analysis import infer_task_type, regression_metrics, binary_metrics


class TorchMetaPredictor(nn.Module):
    """PyTorch neural network for meta-prediction tasks.
    
    This model predicts layer-wise importance metrics from activation statistics.
    Architecture consists of fully-connected layers with batch normalization,
    ReLU activation, and dropout for regularization.
    
    Attributes:
        input_size: Number of input features.
        network: Sequential neural network layers.
    """

    def __init__(self, config: Dict[str, Any], input_size: Optional[int] = None) -> None:
        """Initialize the meta-predictor network.
        
        Args:
            config: Configuration dictionary containing model parameters.
            input_size: Number of input features (inferred from data if None).
        """
        super().__init__()
        model_cfg = config.get("model", {})
        hidden_sizes = model_cfg.get("hidden_sizes", [128, 64, 32])
        dropout = model_cfg.get("dropout", 0.2)

        # Set input dimension (default to 64 if not specified)
        self.input_size = input_size if input_size is not None else 64

        # Build multi-layer feedforward network
        layers = []
        current_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size

        # Final output layer (single scalar prediction)
        layers.append(nn.Linear(current_size, 1))

        # Create sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input feature tensor of shape [batch_size, input_size].
            
        Returns:
            Predicted scalar values of shape [batch_size].
        """
        return self.network(x).squeeze(-1)


def save_checkpoint_portable(
        checkpoint_dir: Path,
        model: torch.nn.Module,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        feature_names: List[str],
        target_name: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """Save model checkpoint in portable, environment-independent format.
    
    Saves three separate files to avoid joblib pickle import-path dependencies:
    1. model_state.pt - PyTorch state_dict (pure tensors)
    2. preprocess.joblib - fitted imputer and scaler
    3. feature_names.json - feature list and metadata
    
    Args:
        checkpoint_dir: Directory to save checkpoint files.
        model: Trained PyTorch model.
        imputer: Fitted SimpleImputer.
        scaler: Fitted StandardScaler.
        feature_names: List of feature column names.
        target_name: Name of target variable.
        task_type: Task type ('regression' or 'classification').
        metadata: Optional additional metadata to store.
        
    Returns:
        Dictionary mapping file types to their paths.
        
    Example:
        >>> paths = save_checkpoint_portable(
        ...     Path("checkpoints/run_001"),
        ...     model, imputer, scaler,
        ...     ["feat1", "feat2"], "target", "regression"
        ... )
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Extract model configuration for reconstruction
    model_config = {}
    if isinstance(model, TorchMetaPredictor):
        # Save the architecture configuration
        hidden_sizes = []
        dropout_rate = 0.0

        # Extract from the sequential network
        i = 0
        while i < len(model.network):
            layer = model.network[i]
            if isinstance(layer, nn.Linear):
                # Check if this is not the final output layer
                if i < len(model.network) - 1:  # Not the last linear layer
                    hidden_sizes.append(layer.out_features)
                i += 1
            elif isinstance(layer, nn.Dropout):
                dropout_rate = layer.p
                i += 1
            else:
                i += 1

        model_config = {
            "input_size": model.input_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout_rate,
        }
    else:
        # Handle MLPRegressor or other models with 'net' attribute
        if hasattr(model, 'net'):
            network = model.net
        elif hasattr(model, 'network'):
            network = model.network
        else:
            network = None

        if network is not None:
            hidden_sizes = []
            dropout_rate = 0.0
            activation = "relu"  # Default

            # Extract from sequential network
            i = 0
            while i < len(network):
                layer = network[i]
                if isinstance(layer, nn.Linear):
                    # Check if this is not the final output layer
                    if i < len(network) - 1:  # Not the last linear layer
                        hidden_sizes.append(layer.out_features)
                    i += 1
                elif isinstance(layer, nn.Dropout):
                    dropout_rate = layer.p
                    i += 1
                elif isinstance(layer, nn.ReLU):
                    activation = "relu"
                    i += 1
                elif isinstance(layer, nn.GELU):
                    activation = "gelu"
                    i += 1
                else:
                    i += 1

            # Get input size from first linear layer
            first_linear = None
            for layer in network:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break

            input_size = first_linear.in_features if first_linear is not None else len(feature_names)

            model_config = {
                "input_size": input_size,
                "hidden_sizes": hidden_sizes,
                "dropout": dropout_rate,
                "activation": activation,
            }

    # 1. Save pure PyTorch state dict (no Python objects)
    model_path = checkpoint_dir / "model_state.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "model_class": model.__class__.__name__,
    }, model_path)

    # 2. Save preprocessing pipeline (imputer + scaler)
    preprocess_path = checkpoint_dir / "preprocess.joblib"
    joblib.dump({
        "imputer": imputer,
        "scaler": scaler,
    }, preprocess_path)

    # 3. Save feature names and metadata as JSON
    features_path = checkpoint_dir / "feature_names.json"
    manifest = {
        "feature_names": feature_names,
        "target_name": target_name,
        "task_type": task_type,
        "n_features": len(feature_names),
        "model_config": model_config,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {}
    }
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[checkpoint] Saved portable checkpoint to: {checkpoint_dir}")
    print(f"  - model_state.pt: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"  - preprocess.joblib: {preprocess_path.stat().st_size / 1024:.1f} KB")
    print(f"  - feature_names.json: {features_path.stat().st_size / 1024:.1f} KB")

    return {
        "model": model_path,
        "preprocess": preprocess_path,
        "features": features_path,
        "dir": checkpoint_dir
    }


def load_checkpoint_portable(
        checkpoint_dir: Path,
        model_class: type = TorchMetaPredictor,
        device: str = "cpu"
) -> Tuple[torch.nn.Module, SimpleImputer, StandardScaler, List[str], str, str, Dict[str, Any]]:
    """Load model checkpoint from portable format.
    
    Reconstructs model and preprocessing pipeline from environment-independent files.
    This avoids pickle import-path issues that occur with joblib.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files.
        model_class: PyTorch model class to instantiate.
        device: Device to load model onto ('cpu' or 'cuda').
        
    Returns:
        Tuple of (model, imputer, scaler, feature_names, target_name, task_type, metadata).
        
    Raises:
        FileNotFoundError: If required checkpoint files are missing.
        
    Example:
        >>> model, imputer, scaler, features, target, task, meta = \\
        ...     load_checkpoint_portable(Path("checkpoints/run_001"))
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check required files exist
    model_path = checkpoint_dir / "model_state.pt"
    preprocess_path = checkpoint_dir / "preprocess.joblib"
    features_path = checkpoint_dir / "feature_names.json"

    missing = []
    if not model_path.exists():
        missing.append(str(model_path))
    if not preprocess_path.exists():
        missing.append(str(preprocess_path))
    if not features_path.exists():
        missing.append(str(features_path))

    if missing:
        raise FileNotFoundError(
            f"Checkpoint incomplete. Missing files: {missing}\n"
            f"Expected portable checkpoint format in: {checkpoint_dir}"
        )

    # Load feature manifest
    with open(features_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    feature_names = manifest["feature_names"]
    target_name = manifest["target_name"]
    task_type = manifest["task_type"]
    metadata = manifest.get("metadata", {})

    # Load preprocessing pipeline (suppress sklearn version warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        preprocess = joblib.load(preprocess_path)
    imputer = preprocess["imputer"]
    scaler = preprocess["scaler"]

    # Reconstruct model architecture
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config from manifest (preferred) or checkpoint
    model_config = manifest.get("model_config") or checkpoint.get("model_config", {})

    # Determine model class from checkpoint if available
    saved_model_class = checkpoint.get("model_class")
    if saved_model_class == "MLPRegressor":
        # Use MLPRegressor constructor
        from ..models.mlp_regressor import MLPRegressor
        model = MLPRegressor(
            in_dim=model_config.get("input_size", len(feature_names)),
            hidden_sizes=model_config.get("hidden_sizes", [128, 64, 32]),
            dropout=model_config.get("dropout", 0.2),
            activation=model_config.get("activation", "relu")
        )
    else:
        # Use TorchMetaPredictor constructor (default)
        cfg = {"model": model_config}
        model = model_class(cfg, input_size=len(feature_names))

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[checkpoint] Loaded portable checkpoint from: {checkpoint_dir}")
    print(f"  - Task: {task_type}, Target: {target_name}")
    print(f"  - Features: {len(feature_names)}")

    return model, imputer, scaler, feature_names, target_name, task_type, metadata


def _select_columns(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, str, List[str]]:
    """Select feature and target columns from meta-dataset.
    
    Extracts numeric features and target variable from dataframe, automatically
    dropping identifier columns and user-specified columns to avoid data leakage.
    
    Args:
        df: Meta-dataset dataframe.
        cfg: Configuration dictionary with data and drop_cols settings.
        
    Returns:
        Tuple of (features, target, target_column_name, feature_column_names).
        
    Raises:
        KeyError: If target column is not found in dataframe.
        ValueError: If no numeric features remain after filtering.
    """
    # Extract target column name from configuration
    data_cfg = cfg.get("data", {})
    target_spec = data_cfg.get("target", "proxy_importance")

    # Handle both string and dict format for target
    if isinstance(target_spec, dict):
        target_col = target_spec.get("column_name", "proxy_importance")
    else:
        target_col = target_spec

    if target_col not in df.columns:
        raise KeyError(f"Target column not found: '{target_col}'")

    # Build set of columns to drop
    drop_cols = set(cfg.get("drop_cols", []))

    # Automatically drop identifier-like columns to prevent data leakage
    id_like_columns = [c for c in df.columns if c.lower() in {"layer_id", "module", "name", "id"}]
    drop_cols.update(id_like_columns)

    # Select numeric feature columns
    numeric_cols = [
        c for c in df.columns
        if c != target_col and c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        raise ValueError(
            "No numeric feature columns remained after filtering. "
            "Check drop_cols configuration and dataset schema."
        )

    X = df[numeric_cols].copy()
    y = df[target_col].copy()
    return X, y, target_col, numeric_cols


def _build_model(task_type: str, cfg: Dict[str, Any]) -> Pipeline:
    """Build scikit-learn compatible model pipeline.
    
    Creates a TorchMetaPredictor and wraps it with TorchModelWrapper for
    compatibility with scikit-learn's fit/predict interface.
    
    Args:
        task_type: Task type ('regression' or 'classification').
        cfg: Configuration dictionary with model and training parameters.
        
    Returns:
        Scikit-learn compatible model wrapper.
    """
    # Create PyTorch neural network
    input_size = cfg.get("model", {}).get("input_size", 64)
    model = TorchMetaPredictor(cfg, input_size=input_size)

    training_cfg = cfg.get("training", {})

    # Wrap with scikit-learn compatible interface
    from .wrappers import TorchModelWrapper
    wrapped_model = TorchModelWrapper(
        model=model,
        batch_size=int(training_cfg.get("batch_size", cfg.get("batch_size", 32))),
        epochs=int(training_cfg.get("max_epochs", cfg.get("epochs", 100))),
        learning_rate=float(training_cfg.get("lr", cfg.get("learning_rate", 0.001))),
        device=training_cfg.get("device", None),
    )

    return wrapped_model


def _feature_importances(model) -> Optional[np.ndarray]:
    """Extract feature importances from model if available.
    
    Args:
        model: Trained model (may or may not have feature_importances_).
        
    Returns:
        Feature importance array, or None if not available.
    """
    try:
        return model["model"].feature_importances_
    except Exception:
        return None


def _paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Generate output paths for model artifacts.
    
    Creates directory structure and generates timestamped filenames for
    checkpoints, reports, and feature importance files.
    
    Args:
        cfg: Configuration dictionary with output settings and run_tag.
        
    Returns:
        Dictionary mapping artifact types to their file paths.
    """
    # Resolve output directories
    runs_dir = Path(cfg.get("outputs", {}).get("runs_dir", "metapac/runs")).resolve()
    ckpt_dir = runs_dir / "checkpoints"
    results_dir = Path(cfg.get("outputs", {}).get("results_dir", "metapac/results")).resolve()

    # Create directories if they don't exist
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped base filename
    tag = cfg.get("run_tag", "meta_baseline")
    timestamp = int(time.time())
    base_name = f"{tag}_{timestamp}"

    return {
        "ckpt_dir": ckpt_dir / base_name,  # Directory for portable checkpoint
        "ckpt": ckpt_dir / f"{base_name}.joblib",  # Legacy format (kept for compatibility)
        "report": results_dir / f"{base_name}_report.json",
        "fi_csv": results_dir / f"{base_name}_featimp.csv",
    }


def train_and_eval(cfg: Dict[str, Any]) -> int:
    """Train and evaluate meta-predictor model.
    
    Main training pipeline that:
    1. Loads meta-dataset
    2. Preprocesses features and handles missing values
    3. Trains model
    4. Evaluates on validation set
    5. Saves model checkpoint and metrics
    
    Args:
        cfg: Configuration dictionary with all training parameters.
        
    Returns:
        Exit code (0 for success).
        
    Raises:
        FileNotFoundError: If meta-dataset file doesn't exist.
    """
    # Load meta-dataset - prioritize data.path over meta_dataset_path
    data_cfg = cfg.get("data", {})
    data_path_str = data_cfg.get("path") or cfg.get("meta_dataset_path",
                                                    "metapac/artifacts/meta_dataset/meta_dataset.parquet")
    data_path = Path(data_path_str)

    if not data_path.exists():
        raise FileNotFoundError(f"Meta-dataset not found: {data_path}")

    print(f"[meta_predictor] Loading meta-dataset from {data_path}")

    # Auto-detect file format and load appropriately
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        # Try parquet first, fallback to CSV
        try:
            df = pd.read_parquet(data_path)
        except:
            df = pd.read_csv(data_path)

    # Drop rows with NaN target values
    data_cfg = cfg.get("data", {})
    target_spec = data_cfg.get("target", "grad_l1")

    # Handle both string and dict format for target
    if isinstance(target_spec, dict):
        target_col = target_spec.get("column_name", "grad_l1")
    else:
        target_col = target_spec

    df = df.dropna(subset=[target_col])

    # Select features and target
    X, y, target_col, feature_cols = _select_columns(df, cfg)

    # Force numeric types and sanitize non-finite values (vectorized for speed)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    X_values = X.to_numpy(dtype=np.float64, copy=True)
    x_non_finite_mask = ~np.isfinite(X_values)
    if x_non_finite_mask.any():
        X_values[x_non_finite_mask] = np.nan
    X = pd.DataFrame(X_values, columns=X.columns, index=X.index)

    y_values = y.to_numpy(dtype=np.float64, copy=True)
    y_non_finite_mask = ~np.isfinite(y_values)
    if y_non_finite_mask.any():
        y_values[y_non_finite_mask] = np.nan
    y = pd.Series(y_values, index=y.index)

    # Drop rows with invalid targets after conversion/sanitization
    valid_target_mask = y.notna()
    dropped_target_rows = int((~valid_target_mask).sum())
    if dropped_target_rows > 0:
        print(f"[meta_predictor] Dropping {dropped_target_rows} rows with invalid target values")
        X = X.loc[valid_target_mask]
        y = y.loc[valid_target_mask]

    # Infer or use configured task type
    task = cfg.get("task_type") or infer_task_type(y.to_numpy())
    print(f"[meta_predictor] Task type: {task}  | target: {target_col}  | n_features: {len(feature_cols)}")

    # Remove features with all NaN values
    nan_counts = X.isna().sum()
    all_nan_cols = nan_counts[nan_counts == len(X)].index
    X = X.drop(columns=all_nan_cols)
    print(f"[meta_predictor] Dropped {len(all_nan_cols)} all-NaN features. Remaining: {len(X.columns)}")

    # Impute remaining NaN values using mean strategy
    from sklearn.impute import SimpleImputer
    feature_imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(feature_imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Final numeric safety pass for features/target
    X_values = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    X = pd.DataFrame(X_values, columns=X.columns, index=X.index)
    y = pd.Series(np.nan_to_num(y.to_numpy(dtype=np.float64), nan=0.0, posinf=1e12, neginf=-1e12), index=y.index)

    # Optional robust clipping to reduce gradient explosion on heavy-tailed targets
    clip_pct = float(cfg.get("target_clip_percentile", 99.9))
    if 50.0 <= clip_pct < 100.0:
        limit = float(np.nanpercentile(np.abs(y.to_numpy()), clip_pct))
        if np.isfinite(limit) and limit > 0:
            y = y.clip(lower=-limit, upper=limit)
            print(f"[meta_predictor] Clipped target to ±{limit:.4g} (p{clip_pct})")

    # Update configuration with actual input size after preprocessing
    cfg["model"] = cfg.get("model", {})
    cfg["model"]["input_size"] = len(X.columns)

    # Split data into training and validation sets
    test_size = float(cfg.get("val_size", 0.2))
    seed = int(cfg.get("seed", 42))
    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=test_size, random_state=seed
    )

    # Build and train model
    model_pipeline = _build_model(task, cfg)
    model_pipeline.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    training_history = getattr(model_pipeline, "training_history_", [])
    best_epoch = getattr(model_pipeline, "best_epoch_", None)
    best_val_rmse = getattr(model_pipeline, "best_val_rmse_", None)

    # Generate predictions and compute metrics
    if task == "regression":
        y_pred = model_pipeline.predict(X_val)
        metrics_dict = regression_metrics(y_val, y_pred)
    else:
        # For classification: use predict_proba if available, else decision_function
        try:
            y_proba = model_pipeline.predict_proba(X_val)[:, 1]
        except Exception:
            from sklearn.preprocessing import MinMaxScaler
            scores = model_pipeline.decision_function(X_val)
            y_proba = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
        metrics_dict = binary_metrics(y_val, y_proba)

    # Save model artifacts
    output_paths = _paths(cfg)

    # Extract the underlying PyTorch model from the wrapper
    pytorch_model = model_pipeline.model if hasattr(model_pipeline, 'model') else None

    # Save in portable format (recommended)
    if pytorch_model is not None and isinstance(pytorch_model, torch.nn.Module):
        # Get preprocessing objects (we just created them above)
        # Note: feature_imputer was used to preprocess X
        # For scaling, we need to fit a new scaler on the imputed data
        scaler = StandardScaler()
        scaler.fit(X.to_numpy())  # Fit on imputed data

        checkpoint_paths = save_checkpoint_portable(
            checkpoint_dir=output_paths["ckpt_dir"],
            model=pytorch_model,
            imputer=feature_imputer,
            scaler=scaler,
            feature_names=list(X.columns),
            target_name=target_col,
            task_type=task,
            metadata={
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "metrics": metrics_dict,
                "training_history": training_history,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_val_rmse": float(best_val_rmse) if best_val_rmse is not None else None,
                "config": cfg
            }
        )
        checkpoint_path_str = str(checkpoint_paths["dir"])
    else:
        # Fallback to legacy joblib format for non-PyTorch models
        print("[meta_predictor] Warning: Using legacy joblib checkpoint format")
        checkpoint_data = {
            "pipeline": model_pipeline,
            "features": list(X.columns),
            "target": target_col,
            "task": task
        }
        joblib.dump(checkpoint_data, output_paths["ckpt"])
        checkpoint_path_str = str(output_paths["ckpt"])

    # Save feature importances if available
    feature_importances = _feature_importances(model_pipeline)
    if feature_importances is not None and len(feature_importances) == len(X.columns):
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": feature_importances
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(output_paths["fi_csv"], index=False)

    # Generate and save evaluation report
    report = {
        "task": task,
        "target": target_col,
        "n_features": len(X.columns),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "metrics": metrics_dict,
        "training_history": training_history,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_rmse": float(best_val_rmse) if best_val_rmse is not None else None,
        "checkpoint": checkpoint_path_str,
        "feature_importances_csv": str(output_paths["fi_csv"]) if feature_importances is not None else None,
        "config_used": cfg,
    }
    Path(output_paths["report"]).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Print summary
    print("[meta_predictor] Metrics:", json.dumps(metrics_dict, indent=2, ensure_ascii=False))
    print(f"[meta_predictor] Saved checkpoint to: {checkpoint_path_str}")
    print(f"[meta_predictor] Saved report to: {output_paths['report']}")
    if feature_importances is not None:
        print(f"[meta_predictor] Saved feature importances to: {output_paths['fi_csv']}")

    return 0
