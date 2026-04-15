# data/meta_dataset.py
# Robust Parquet/CSV loading, flexible auto-inference, group z-score targets,
# early dropping of empty columns, and storing the imputer/scaler as attributes.
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

_FLOAT_REGEX = re.compile(
    r"""
    [+-]?              # optional sign
    (?:
        (?:\d+\.\d*)|  # 123. or 123.45
        (?:\.\d+)|     # .45
        (?:\d+)        # 123
    )
    (?:[eE][+-]?\d+)?  # optional exponent
    """,
    re.VERBOSE,
)


def _extract_first_float(s: str) -> float | None:
    if s is None:
        return None
    m = _FLOAT_REGEX.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _coerce_numeric_series(sr: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
      1) pd.to_numeric(..., errors="coerce")
      2) if too many NaNs remain, regex extraction of first float-like token
    """
    s1 = pd.to_numeric(sr, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s1.isna().mean() < 0.80:
        return s1
    extracted = sr.astype(str).map(_extract_first_float)
    s2 = pd.to_numeric(extracted, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return s2


class MetaDataset:
    """
        Accepted configuration format (either top-level cfg or cfg['data']):
    {
      "data": {
        "path": "data/meta_dataset.parquet",
                "features": [...],                  # optional; auto_infer is used if omitted
        "auto_infer": {
          "mode": "prefix_then_any",        # "prefix_only" | "any_numeric" | "prefix_then_any"
          "prefixes": ["act_","grad_","l1","l2","taylor_","act","grad"],
          "denylist": ["module","layer","param","param_name","name","id","index","idx","epoch","step","target","y"],
          "min_numeric_ratio": 0.35,
          "also_allow_if_dtype_numeric": true
        },
        "split": {"test_size":0.2,"val_size":0.1,"group_col":"module","random_state":42},
        "target": {"mode":"column_group_zscore","column_name":"grad_l1","eps":1.0e-8,"clip_sigma":5.0}
      }
    }
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg: Dict[str, Any] = cfg["data"] if "data" in cfg else cfg
        # Persist these for reuse during training/inference.
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names_: List[str] = []
        self.dropped_feature_names_: List[str] = []

    # ---------- IO ----------

    def _read(self) -> pd.DataFrame:
        path = self.cfg["path"]

        # Memory optimization: first read column names only, then select subset
        if path.lower().endswith(".parquet"):
            # Read just the schema to get column names
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            all_columns = schema.names

            # Determine which columns we might need based on config
            ai = self.cfg.get("auto_infer", {}) or {}
            prefixes = list(ai.get("prefixes", ["act_", "grad_", "param_", "l1", "l2"]))

            # Always include metadata columns
            meta_cols = ["module", "group", "layer", "param", "param_name", "name",
                         "id", "index", "idx", "epoch", "step", "run_id", "phase", "reducer"]

            # Select columns: metadata + prefix-matching + target column
            tcfg = self.cfg.get("target", {})
            target_col = tcfg.get("column_name") if isinstance(tcfg, dict) else tcfg

            selected_cols = set()
            for col in all_columns:
                # Include metadata
                if any(meta in col.lower() for meta in meta_cols):
                    selected_cols.add(col)
                # Include prefix-matching columns
                elif any(col.startswith(p) for p in prefixes):
                    selected_cols.add(col)
                # Include target
                elif target_col and col == target_col:
                    selected_cols.add(col)

            selected_cols = list(selected_cols)
            print(
                f"[MetaDataset] Reading {len(selected_cols)}/{len(all_columns)} columns from parquet (memory optimization)")

            # Additional memory optimization: sample rows if dataset is very large
            parquet_file = pq.ParquetFile(path)
            total_rows = parquet_file.metadata.num_rows
            sample_fraction = self.cfg.get("sample_fraction", None)

            if total_rows > 100_000 and sample_fraction is None:
                # Auto-enable sampling for large datasets
                sample_fraction = 0.15  # 15% sample
                print(f"[MetaDataset] Large dataset detected ({total_rows:,} rows), using {sample_fraction:.1%} sample")

            if sample_fraction and sample_fraction < 1.0:
                # Read in batches and sample
                print(f"[MetaDataset] Reading and sampling {sample_fraction:.1%} of {total_rows:,} rows...")
                import random
                random.seed(42)

                # Calculate how many rows to sample
                target_rows = int(total_rows * sample_fraction)
                batch_size = 50_000
                sampled_dfs = []
                rows_read = 0

                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_cols):
                    batch_df = batch.to_pandas()
                    # Sample from this batch
                    n_sample = min(len(batch_df), max(1, int(len(batch_df) * sample_fraction)))
                    if n_sample < len(batch_df):
                        batch_df = batch_df.sample(n=n_sample, random_state=42 + rows_read)
                    sampled_dfs.append(batch_df)
                    rows_read += len(batch_df)

                    if rows_read >= target_rows:
                        break

                df = pd.concat(sampled_dfs, ignore_index=True)
                print(f"[MetaDataset] Sampled {len(df):,} rows from {total_rows:,} ({len(df) / total_rows:.1%})")
            else:
                df = pd.read_parquet(path, columns=selected_cols)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"Input file read empty DataFrame: {path}")

        # Optional phase filtering
        phase_filter = self.cfg.get("phase_filter")
        if phase_filter and "phase" in df.columns:
            original_len = len(df)
            df = df[df["phase"] == phase_filter].reset_index(drop=True)
            print(f"[MetaDataset] Phase filter '{phase_filter}': {original_len} -> {len(df)} rows")

        return df

    # ---------- Feature selection ----------

    def _load_columns_manifest(self) -> Dict[str, Any] | None:
        """Load columns manifest (columns.json) if available.
        
        This manifest is generated by the feature builder and contains
        the canonical list of feature columns that passed NaN filtering.
        Using this ensures consistency between feature extraction and training.
        
        Returns:
            Dictionary with feature list, or None if manifest not found.
        """
        path = Path(self.cfg.get("path", ""))
        if not path.exists():
            return None

        # Look for columns.json in same directory as the dataset
        columns_path = path.parent / "columns.json"
        if not columns_path.exists():
            return None

        try:
            with open(columns_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"[MetaDataset] Loaded column manifest from {columns_path}")
            return manifest
        except Exception as e:
            print(f"[MetaDataset] Warning: Failed to load columns.json: {e}")
            return None

    def _drop_high_nan_columns(
            self,
            Xdf: pd.DataFrame,
            threshold: float = 0.5,
            prefer_activation_features: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop feature columns with excessive NaN ratios.
        
        Implements column-wise NaN threshold filtering to remove unreliable features.
        Uses separate thresholds for activation/gradient features vs parameter features,
        as parameter features typically have lower NaN ratios.
        
        Args:
            Xdf: Feature dataframe to filter.
            threshold: Maximum allowed NaN ratio (0.0 to 1.0).
            prefer_activation_features: If True, use stricter threshold for param_ features.
            
        Returns:
            Tuple of (filtered_dataframe, list_of_dropped_column_names).
        """
        nan_ratios = Xdf.isna().mean()

        # Separate thresholds for different feature types
        act_grad_threshold = threshold
        param_threshold = threshold * 0.7 if prefer_activation_features else threshold

        dropped = []
        kept = []

        for col in Xdf.columns:
            nan_ratio = nan_ratios[col]

            # Apply appropriate threshold based on feature type
            if col.startswith("param_"):
                max_allowed = param_threshold
            else:
                max_allowed = act_grad_threshold

            if nan_ratio > max_allowed:
                dropped.append(col)
            else:
                kept.append(col)

        if dropped:
            print(f"[MetaDataset] Dropping {len(dropped)} high-NaN columns (threshold={threshold:.2f}):")
            # Show top offenders
            dropped_with_ratios = [(c, nan_ratios[c]) for c in dropped]
            dropped_with_ratios.sort(key=lambda x: x[1], reverse=True)
            for col, ratio in dropped_with_ratios[:10]:
                print(f"  - {col}: {ratio:.3f}")
            if len(dropped) > 10:
                print(f"  ... and {len(dropped) - 10} more")

        return Xdf[kept].copy(), dropped

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        # Explicit list.
        if "features" in self.cfg and self.cfg["features"]:
            feats = list(self.cfg["features"])
            missing = [c for c in feats if c not in df.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            print(f"[MetaDataset] Using explicit features ({len(feats)}).")
            return feats

        # Auto-infer.
        ai = self.cfg.get("auto_infer", {}) or {}
        mode = str(ai.get("mode", "prefix_then_any"))
        prefixes: List[str] = list(ai.get("prefixes", ["act_", "grad_", "l1", "l2", "taylor_", "act", "grad"]))
        deny: set[str] = {str(x) for x in ai.get("denylist", [])}
        min_numeric_ratio: float = float(ai.get("min_numeric_ratio", 0.35))
        also_allow_if_dtype_numeric: bool = bool(ai.get("also_allow_if_dtype_numeric", True))

        # Exclude columns that are obviously not features.
        deny_dynamic = set(deny)
        heuristic_deny = {"module", "group", "layer", "param", "param_name", "name", "id", "index", "idx", "epoch",
                          "step", "target", "y"}
        deny_dynamic |= {c for c in df.columns if str(c).lower() in heuristic_deny}

        def usable(sr: pd.Series) -> bool:
            # Fast path: if already numeric, just check NaN ratio
            if pd.api.types.is_numeric_dtype(sr.dtype):
                return (1.0 - sr.isna().mean()) >= min_numeric_ratio
            # Slow path: try coercion
            s = _coerce_numeric_series(sr)
            return (1.0 - s.isna().mean()) >= min_numeric_ratio

        def select_prefix_candidates() -> List[str]:
            cand: List[str] = []
            for c in df.columns:
                if c in deny_dynamic:
                    continue
                if any(str(c).startswith(p) for p in prefixes):
                    if usable(df[c]):
                        cand.append(c)
                    elif also_allow_if_dtype_numeric and pd.api.types.is_numeric_dtype(df[c].dtype):
                        cand.append(c)
            return sorted(cand)

        def select_any_numeric() -> List[str]:
            cand: List[str] = []
            for c in df.columns:
                if c in deny_dynamic:
                    continue
                col = df[c]
                if pd.api.types.is_numeric_dtype(col.dtype):
                    cand.append(c)
                    continue
                if usable(col):
                    cand.append(c)
            return sorted(cand)

        chosen: List[str] = []
        tried_paths: List[Tuple[str, int]] = []

        if mode in ("prefix_only", "prefix_then_any"):
            chosen = select_prefix_candidates()
            tried_paths.append(("prefix", len(chosen)))

        if mode == "prefix_only":
            if not chosen:
                raise ValueError("Auto-infer found no features with prefixes; "
                                 "consider mode='prefix_then_any' or provide explicit 'features'.")
        elif mode == "any_numeric":
            chosen = select_any_numeric()
            tried_paths.append(("any_numeric", len(chosen)))
        elif mode == "prefix_then_any":
            if not chosen:
                chosen = select_any_numeric()
                tried_paths.append(("fallback:any_numeric", len(chosen)))
        else:
            raise ValueError(f"Unknown auto_infer.mode: {mode}")

        print(f"[MetaDataset] Auto-infer tried: {tried_paths}; selected={len(chosen)}")

        if not chosen:
            sample_cols = list(df.columns)[:20]
            raise ValueError(
                "Auto-infer found no usable numeric feature columns.\n"
                f"mode={mode}, prefixes={prefixes}, min_numeric_ratio={min_numeric_ratio}\n"
                f"Example columns: {sample_cols}\n"
                "Try lowering min_numeric_ratio, adjusting prefixes, or specifying explicit 'features'."
            )

        # Sanity check: never include the target or group column.
        tcfg = self.cfg.get("target", {})
        # Handle both dict and string target configurations
        if isinstance(tcfg, dict):
            tcol = tcfg.get("column_name")
        elif isinstance(tcfg, str):
            tcol = tcfg
        else:
            tcol = None

        group_col = self.cfg.get("split", {}).get("group_col")
        chosen = [c for c in chosen if c not in {tcol, group_col}]
        print(f"[MetaDataset] Selected features (first 30): {chosen[:30]}")
        return chosen

    # ---------- Target ----------

    def _column_group_zscore(
            self, df: pd.DataFrame, col: str, group_key: str, eps: float, clip_sigma: float | None
    ) -> np.ndarray:
        vals = _coerce_numeric_series(df[col])
        groups = df[group_key]
        mu = vals.groupby(groups).transform("mean")
        var = vals.groupby(groups).transform("var")
        std = np.sqrt(var.fillna(0.0) + float(eps))
        z = (vals - mu) / std
        if clip_sigma is not None:
            z = z.clip(-float(clip_sigma), float(clip_sigma))
        return z.to_numpy(dtype=float)

    def _make_target(self, df: pd.DataFrame) -> np.ndarray:
        tgt = self.cfg["target"]

        # Handle both dict and simple string target configurations
        if isinstance(tgt, str):
            # Simple case: target is just a column name
            col = tgt
            if col not in df.columns:
                raise KeyError(f"Target column '{col}' not found.")
            y = _coerce_numeric_series(df[col]).to_numpy(dtype=float)
        elif isinstance(tgt, dict):
            mode = tgt["mode"]

            if mode == "column":
                col = tgt["column_name"]
                if col not in df.columns:
                    raise KeyError(f"Target column '{col}' not found.")
                y = _coerce_numeric_series(df[col]).to_numpy(dtype=float)

            elif mode == "column_group_zscore":
                col = tgt["column_name"]
                group_key = self.cfg["split"]["group_col"]
                if group_key not in df.columns:
                    raise KeyError(f"group_col '{group_key}' not found in dataframe.")
                if col not in df.columns:
                    raise KeyError(f"Target column '{col}' not found in dataframe.")
                eps = float(tgt.get("eps", 1.0e-8))
                clip = tgt.get("clip_sigma", 5.0)
                y = self._column_group_zscore(df, col, group_key, eps, clip)

            elif mode == "ga_taylor":
                g = _coerce_numeric_series(df["grad_mean"]).to_numpy(dtype=float)
                a = _coerce_numeric_series(df["act_mean"]).to_numpy(dtype=float)
                y = np.abs(g * a)

            elif mode == "ga_simple":
                g = _coerce_numeric_series(df["grad_mean"]).to_numpy(dtype=float)
                a = _coerce_numeric_series(df["act_mean"]).to_numpy(dtype=float)
                y = np.abs(g) + np.abs(a)

            else:
                raise ValueError(f"Unknown target mode: {mode}")
        else:
            raise ValueError(f"Invalid target configuration type: {type(tgt)}")

        finite_mask = np.isfinite(y)
        with np.errstate(all="ignore"):
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)
            y_mean = np.nanmean(y)
        print(
            f"[MetaDataset] target stats — shape={y.shape} finite%={finite_mask.mean():.3f} "
            f"min={y_min:.4e} max={y_max:.4e} mean={y_mean:.4e}"
        )
        return y

    # ---------- X prep ----------

    def _sanitize_X(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        Xdf = df[features].copy()
        for c in Xdf.columns:
            Xdf[c] = _coerce_numeric_series(Xdf[c])
        return Xdf

    def _build_mask(self, y: np.ndarray, Xdf: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        mask_y = np.isfinite(y)
        all_nan_rows = Xdf.isna().all(axis=1).to_numpy()
        mask = mask_y & (~all_nan_rows)
        return mask, {"nonfinite_y": int((~mask_y).sum()), "all_nan_X": int(all_nan_rows.sum())}

    # ---------- API ----------

    def load(self):
        df = self._read()

        # Try to load columns manifest from builder output
        manifest = self._load_columns_manifest()

        # Feature selection: prefer manifest, otherwise auto-infer
        if manifest and "feature_columns" in manifest:
            # Use pre-validated feature list from builder
            features = manifest["feature_columns"]
            # Filter to only those present in current dataframe
            features = [f for f in features if f in df.columns]
            print(f"[MetaDataset] Using {len(features)} features from columns.json manifest")
        else:
            # Fallback to auto-inference
            features = self._resolve_features(df)

        # X coercion + NaN ratios.
        Xdf_full = self._sanitize_X(df, features)
        nan_ratios_full = Xdf_full.isna().mean().sort_values(ascending=False)

        # Show NaN statistics before filtering
        high_nan_count = (nan_ratios_full > 0.5).sum()
        print(f"[MetaDataset] X per-feature NaN ratio (after coercion): {high_nan_count} features >50% NaN")
        if high_nan_count > 0:
            print(f"[MetaDataset] Top 10 highest NaN ratios:")
            print(nan_ratios_full.head(10).to_string())

        # Drop columns with excessive NaN ratios
        nan_threshold = float(self.cfg.get("drop_nan_threshold", 0.5))
        Xdf_full, dropped_high_nan = self._drop_high_nan_columns(
            Xdf_full,
            threshold=nan_threshold,
            prefer_activation_features=True
        )

        print(f"[MetaDataset] After NaN filtering: {len(Xdf_full.columns)} features remain")

        # Target.
        y_full = self._make_target(df).astype(float)

        # Masking.
        mask, reasons = self._build_mask(y_full, Xdf_full)
        total = int(len(df))
        used = int(mask.sum())
        dropped = int((~mask).sum())
        print(
            f"[MetaDataset] Row filter — total={total} used={used} dropped={dropped} "
            f"(nonfinite_y={reasons['nonfinite_y']}, all_nan_X={reasons['all_nan_X']})"
        )
        if used == 0:
            top_bad = nan_ratios_full.head(12).to_dict()
            raise ValueError(
                "After numeric coercion and masking, no rows remained.\n"
                f"Top features by NaN ratio: {top_bad}\n"
                "Hints:\n"
                " - Check target.column_name and split.group_col exist and hold numeric/coercible values for target.\n"
                " - Relax auto_infer.min_numeric_ratio or expand prefixes; or set explicit 'features'."
            )

        # Filtered data.
        df = df.loc[mask].reset_index(drop=True)
        Xdf = Xdf_full.loc[mask].reset_index(drop=True)
        y = y_full[mask]

        # Drop columns that are entirely empty on the masked subset.
        nonempty_cols = [c for c in Xdf.columns if not Xdf[c].isna().all()]
        dropped_cols = [c for c in Xdf.columns if c not in nonempty_cols]
        if dropped_cols:
            print(
                f"[MetaDataset] Dropping {len(dropped_cols)} feature(s) with all-NaN on used rows: {dropped_cols[:30]}")
        Xdf = Xdf[nonempty_cols].copy()

        # Impute + scale, then store the fitted objects on the instance.
        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(Xdf.to_numpy(dtype=float))

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = self.scaler.fit_transform(X_imp)

        # Feature metadata.
        self.feature_names_ = list(nonempty_cols)
        self.dropped_feature_names_ = dropped_cols + dropped_high_nan

        # split
        split = self.cfg["split"]
        test_size = float(split["test_size"])
        val_size = float(split["val_size"])
        rnd = int(split.get("random_state", 42))
        group_col = split["group_col"]
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' not in dataframe.")
        groups = df[group_col].astype(str).to_numpy()

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd)
        train_val_idx, test_idx = next(gss.split(X_scaled, y, groups))

        X_trainval, X_test = X_scaled[train_val_idx], X_scaled[test_idx]
        y_trainval, y_test = y[train_val_idx], y[test_idx]
        df_trainval, df_test = df.iloc[train_val_idx], df.iloc[test_idx]

        adj_val = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val, random_state=rnd)
        groups_trainval = df_trainval[group_col].astype(str).to_numpy()
        train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        df_train, df_val = df_trainval.iloc[train_idx], df_trainval.iloc[val_idx]

        print(
            f"[MetaDataset] After imputation: X shape={X_scaled.shape}, finite%={np.isfinite(X_scaled).mean():.3f} | "
            f"y finite%={np.isfinite(y).mean():.3f}"
        )
        print(
            f"[MetaDataset] Shapes — "
            f"X_train:{X_train.shape} X_val:{X_val.shape} X_test:{X_test.shape} | "
            f"y_train:{y_train.shape} y_val:{y_val.shape} y_test:{y_test.shape}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
