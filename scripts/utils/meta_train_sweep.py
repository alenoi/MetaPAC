#!/usr/bin/env python3
"""Meta-predictor training sweep runner.

Runs `python -m metapac --mode train_meta --config <trial.yaml>` for each trial,
collects `metrics.rmse` / `metrics.mae` / `metrics.r2` from generated report JSON,
and writes a sweep summary.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _set_in(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = d
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _get_in(d: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _extract_metric(obj: Dict[str, Any], dotted: str) -> float | None:
    cur: Any = obj
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    try:
        return float(cur)
    except Exception:
        return None


def _find_report(results_dir: Path, exp_name: str) -> Path | None:
    candidates = sorted(results_dir.glob(f"{exp_name}_*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _compute_posthoc_spearman(checkpoint_dir: str, cfg: Dict[str, Any], data_path: Path) -> float | None:
    """Recompute validation Spearman for a saved checkpoint.

    This mirrors train_meta preprocessing and split behavior so Spearman can be
    used as sweep metric even if original report does not store it.
    """
    try:
        import numpy as np
        import pandas as pd
        import torch
        from scipy.stats import spearmanr
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        from metapac.src.models.meta_predictor import load_checkpoint_portable
    except Exception:
        return None

    try:
        model, imputer, _saved_scaler, feature_names, target_name, _task, meta = load_checkpoint_portable(
            Path(checkpoint_dir), device="cpu"
        )

        if not data_path.exists():
            return None

        if data_path.suffix.lower() == ".csv":
            df = pd.read_csv(data_path)
        else:
            df = pd.read_parquet(data_path)

        X_df = df[feature_names].apply(pd.to_numeric, errors="coerce")
        y_ser = pd.to_numeric(df[target_name], errors="coerce")

        X = X_df.to_numpy(dtype=np.float64, copy=True)
        y = y_ser.to_numpy(dtype=np.float64, copy=True)
        X[~np.isfinite(X)] = np.nan
        y[~np.isfinite(y)] = np.nan

        valid = ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            X = imputer.transform(X)

        X = np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12)
        y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)

        used_cfg = (meta or {}).get("config", cfg) or cfg
        clip_pct = float(used_cfg.get("target_clip_percentile", cfg.get("target_clip_percentile", 99.9)))
        if 50.0 <= clip_pct < 100.0:
            limit = float(np.nanpercentile(np.abs(y), clip_pct))
            if np.isfinite(limit) and limit > 0:
                y = np.clip(y, -limit, limit)

        val_size = float(used_cfg.get("val_size", cfg.get("val_size", 0.2)))
        seed = int(used_cfg.get("seed", cfg.get("seed", 42)))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=seed)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float()).cpu().numpy().reshape(-1)

        rho = spearmanr(y_val, preds).correlation
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            return None
        return float(rho)
    except Exception:
        return None


@dataclass
class TrialResult:
    trial_id: str
    return_code: int
    params: Dict[str, Any]
    config_path: str
    report_path: str | None = None
    checkpoint: str | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    spearman: float | None = None
    objective: float | None = None
    training_curve_json: str | None = None
    training_curve_csv: str | None = None
    error: str | None = None


def _sample_variants(parameters: Dict[str, List[Any]], mode: str, max_trials: int, seed: int) -> List[Dict[str, Any]]:
    keys = list(parameters.keys())
    if mode == "grid":
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*[parameters[k] for k in keys])]
    else:
        rng = random.Random(seed)
        combos = []
        seen = set()
        attempts = max_trials * 20
        while len(combos) < max_trials and attempts > 0:
            attempts -= 1
            c = {k: rng.choice(parameters[k]) for k in keys}
            sig = tuple((k, _canon(c[k])) for k in sorted(c.keys()))
            if sig in seen:
                continue
            seen.add(sig)
            combos.append(c)
    if mode == "grid" and max_trials > 0:
        return combos[:max_trials]
    return combos


def _manual_variants(sweep_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    trial_overrides = sweep_cfg.get("trial_overrides", [])
    if not isinstance(trial_overrides, list) or not trial_overrides:
        raise ValueError("'trial_overrides' must be a non-empty list when provided")

    seen_ids = set()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for i, item in enumerate(trial_overrides, start=1):
        if isinstance(item, dict) and "params" in item:
            params = item.get("params", {})
            trial_id = str(item.get("id", f"t{i:03d}"))
        elif isinstance(item, dict):
            params = item
            trial_id = f"t{i:03d}"
        else:
            raise ValueError(f"trial_overrides[{i}] must be dict or dict with 'params'")

        if not isinstance(params, dict) or not params:
            raise ValueError(f"trial_overrides[{i}] has empty or invalid params")

        if trial_id in seen_ids:
            raise ValueError(f"Duplicate trial id in trial_overrides: {trial_id}")
        seen_ids.add(trial_id)

        out.append((trial_id, params))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Meta-predictor training sweep")
    ap.add_argument("--base-config", required=True, help="Base train_meta config")
    ap.add_argument("--sweep-config", required=True, help="Sweep spec yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_cfg = _read_yaml(Path(args.base_config))
    sweep_cfg = _read_yaml(Path(args.sweep_config))

    manual_mode = "trial_overrides" in sweep_cfg and sweep_cfg.get("trial_overrides") is not None

    parameters = sweep_cfg.get("parameters", {})
    if not manual_mode:
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("Sweep config must contain non-empty 'parameters' or 'trial_overrides'")

        for k, v in parameters.items():
            if not isinstance(v, list) or not v:
                raise ValueError(f"Parameter '{k}' must be a non-empty list")

    mode = str(sweep_cfg.get("search_mode", "random")).lower().strip()
    if not manual_mode and mode not in {"random", "grid"}:
        raise ValueError("search_mode must be 'random' or 'grid'")

    max_trials = int(sweep_cfg.get("max_trials", 12))
    seed = int(sweep_cfg.get("seed", 42))
    objective_key = str(sweep_cfg.get("objective", "metrics.rmse"))
    objective_direction = str(sweep_cfg.get("objective_direction", "min")).lower().strip()
    if objective_direction not in {"min", "max"}:
        raise ValueError("objective_direction must be 'min' or 'max'")

    enable_posthoc_spearman = bool(sweep_cfg.get("compute_posthoc_spearman", True))

    if manual_mode:
        trial_specs = _manual_variants(sweep_cfg)
    else:
        variants = _sample_variants(parameters, mode=mode, max_trials=max_trials, seed=seed)
        trial_specs = [(f"t{i:03d}", p) for i, p in enumerate(variants, start=1)]

    sweep_id = f"meta_train_sweep_{_now()}"
    sweep_root = repo_root / "experiments" / "meta_train_sweep" / sweep_id
    cfg_root = sweep_root / "configs"
    results_path = sweep_root / "results" / "summary.json"

    print(f"[sweep] id: {sweep_id}")
    print(f"[sweep] base: {args.base_config}")
    print(f"[sweep] objective: {objective_key} ({objective_direction})")
    mode_label = "manual" if manual_mode else mode
    print(f"[sweep] mode={mode_label} | trials={len(trial_specs)}")

    if args.dry_run:
        print("[dry-run] first variants:")
        for i, (tid, v) in enumerate(trial_specs[: min(5, len(trial_specs))], start=1):
            print(f"  {i}: {tid} -> {v}")
        print(f"[dry-run] posthoc_spearman: {enable_posthoc_spearman}")
        print(f"[dry-run] output dir: {sweep_root}")
        return

    results_dir = repo_root / str(_get_in(base_cfg, "outputs.results_dir") or "metapac/results")
    base_data_path = repo_root / str(_get_in(base_cfg, "data.path") or "metapac/artifacts/meta_dataset/meta_dataset.parquet")

    rows: List[TrialResult] = []
    best: TrialResult | None = None

    for i, (trial_id, params) in enumerate(trial_specs, start=1):
        exp_name = f"{sweep_id}_{trial_id}"

        cfg = copy.deepcopy(base_cfg)
        cfg["mode"] = "train_meta"
        cfg["experiment_name"] = exp_name
        cfg["run_tag"] = exp_name

        for k, v in params.items():
            _set_in(cfg, k, v)

        trial_cfg_path = cfg_root / f"{trial_id}.yaml"
        _write_yaml(trial_cfg_path, cfg)

        print(f"[trial] {trial_id} ({i}/{len(trial_specs)}) running: {exp_name}")
        cmd = [sys.executable, "-m", "metapac", "--mode", "train_meta", "--config", str(trial_cfg_path)]
        proc = subprocess.run(cmd, text=True)

        row = TrialResult(
            trial_id=trial_id,
            return_code=int(proc.returncode),
            params=params,
            config_path=str(trial_cfg_path),
        )

        if proc.returncode != 0:
            row.error = "train_meta process returned non-zero exit code"
            rows.append(row)
            continue

        rep = _find_report(results_dir, exp_name)
        if rep is None:
            row.error = f"Missing report for experiment_name={exp_name}"
            rows.append(row)
            continue

        data = json.loads(rep.read_text(encoding="utf-8"))
        row.report_path = str(rep)
        row.checkpoint = data.get("checkpoint")
        m = data.get("metrics", {}) if isinstance(data, dict) else {}
        row.mae = _extract_metric({"metrics": m}, "metrics.mae")
        row.rmse = _extract_metric({"metrics": m}, "metrics.rmse")
        row.r2 = _extract_metric({"metrics": m}, "metrics.r2")

        trial_data_path = repo_root / str(
            _get_in(cfg, "data.path")
            or _get_in(base_cfg, "data.path")
            or "metapac/artifacts/meta_dataset/meta_dataset.parquet"
        )
        if enable_posthoc_spearman and row.checkpoint:
            row.spearman = _compute_posthoc_spearman(row.checkpoint, cfg, trial_data_path)

        if objective_key in {"spearman", "metrics.spearman", "posthoc.spearman"}:
            row.objective = row.spearman
        else:
            row.objective = _extract_metric(data, objective_key)

        # Persist learning curves per-trial (for later plotting)
        history = data.get("training_history", []) if isinstance(data, dict) else []
        step_history = data.get("train_step_history", []) if isinstance(data, dict) else []
        if isinstance(history, list) and history:
            lc_dir = sweep_root / "results" / "learning_curves"
            lc_dir.mkdir(parents=True, exist_ok=True)
            lc_json = lc_dir / f"{trial_id}_training_history.json"
            lc_csv = lc_dir / f"{trial_id}_training_history.csv"
            lc_json.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame(history).to_csv(lc_csv, index=False)
            row.training_curve_json = str(lc_json)
            row.training_curve_csv = str(lc_csv)

        if isinstance(step_history, list) and step_history:
            lc_dir = sweep_root / "results" / "learning_curves"
            lc_dir.mkdir(parents=True, exist_ok=True)
            step_json = lc_dir / f"{trial_id}_train_step_history.json"
            step_csv = lc_dir / f"{trial_id}_train_step_history.csv"
            step_json.write_text(json.dumps(step_history, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame(step_history).to_csv(step_csv, index=False)

        if row.objective is None:
            if objective_key in {"spearman", "metrics.spearman", "posthoc.spearman"}:
                row.error = f"Objective unavailable: {objective_key} (posthoc Spearman returned None)"
            else:
                row.error = f"Objective not found: {objective_key}"
        else:
            if best is None:
                best = row
            else:
                if objective_direction == "min" and row.objective < (best.objective if best.objective is not None else float("inf")):
                    best = row
                if objective_direction == "max" and row.objective > (best.objective if best.objective is not None else float("-inf")):
                    best = row

        rows.append(row)

        out = {
            "sweep_id": sweep_id,
            "base_config": str(Path(args.base_config)),
            "sweep_config": str(Path(args.sweep_config)),
            "objective": objective_key,
            "objective_direction": objective_direction,
            "search_mode": mode_label,
            "num_trials": len(rows),
            "num_valid": sum(1 for r in rows if r.return_code == 0 and r.objective is not None),
            "best_trial": asdict(best) if best else None,
            "trials": [asdict(r) for r in rows],
        }
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sweep] done. summary: {results_path}")


if __name__ == "__main__":
    main()
