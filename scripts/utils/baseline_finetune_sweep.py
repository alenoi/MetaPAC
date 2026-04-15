#!/usr/bin/env python3
"""
Baseline fine-tune hyperparameter sweep for MetaPAC.

Runs multiple `mode: baseline_finetune` trials by patching a base YAML,
executes `python -m metapac --config <trial.yaml>`, and ranks trials by
an evaluation metric from `summary_main.json`.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


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
    keys = dotted_key.split(".")
    cur = d
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


def default_space() -> Dict[str, List[Any]]:
    return {
        "baseline_finetune.train.learning_rate": [1.0e-5, 2.0e-5, 5.0e-5],
        "baseline_finetune.train.weight_decay": [0.0, 0.01],
        "baseline_finetune.train.num_train_epochs": [3, 5],
        "baseline_finetune.train.per_device_train_batch_size": [64, 128],
        "baseline_finetune.train.per_device_eval_batch_size": [64, 128],
        "baseline_finetune.train.gradient_accumulation_steps": [4, 8],
        "baseline_finetune.train.warmup_ratio": [0.03, 0.05],
    }


def load_space(config_path: str | None) -> Dict[str, List[Any]]:
    """Load search space from YAML or fall back to defaults.

    Expected YAML format:
      search_space:
        baseline_finetune.train.learning_rate: [2e-5, 5e-5]
        ...
    Or directly as a mapping of dotted-keys to value lists.
    """
    if not config_path:
        return default_space()

    cfg = _read_yaml(Path(config_path))
    space = cfg.get("search_space", cfg)

    if not isinstance(space, dict) or not space:
        raise ValueError(f"Invalid sweep config format: {config_path}")

    normalized: Dict[str, List[Any]] = {}
    for k, v in space.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid parameter key in sweep config: {k}")
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Each parameter must map to a non-empty list. Problem key: {k}")
        normalized[k] = v

    return normalized


def variants(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*vals):
        out.append(dict(zip(keys, combo)))
    return out


def score_from_summary(summary: Dict[str, Any], metric_key: str) -> float | None:
    try:
        node: Any = summary
        for k in metric_key.split("."):
            node = node[k]
        return float(node)
    except Exception:
        return None


def run_trial(cfg_path: Path) -> int:
    cmd = [sys.executable, "-m", "metapac", "--config", str(cfg_path)]
    p = subprocess.run(cmd, text=True)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline fine-tune hyperparameter sweep")
    ap.add_argument("--base-config", required=True, help="Base baseline_finetune YAML")
    ap.add_argument("--max-experiments", "-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default="eval.eval_accuracy", help="Metric key in summary_main.json")
    ap.add_argument("--maximize", action="store_true", help="Maximize metric (default true for eval_f1)")
    ap.add_argument("--sweep-config", type=str, default=None,
                    help="Path to YAML search space config (default: internal search space)")
    ap.add_argument("--collect-hooks", action="store_true",
                    help="Enable hook CSV collection during fine-tune (default: disabled)")
    ap.add_argument("--keep-trial-artifacts", action="store_true",
                    help="Keep per-trial output folders under targets/distilgpt2/runs (default: delete after metric extraction)")
    ap.add_argument("--resume-dir", type=str, default=None,
                    help="Resume an interrupted sweep from experiments/baseline_sweep/<sweep_id>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    maximize = args.maximize or args.metric.startswith("eval.")

    base_path = Path(args.base_config)
    base_cfg = _read_yaml(base_path)

    space = load_space(args.sweep_config)
    all_variants = variants(space)

    rng = random.Random(args.seed)
    rng.shuffle(all_variants)
    chosen = all_variants[: args.max_experiments]

    if args.resume_dir:
        root = Path(args.resume_dir)
        sweep_id = root.name
    else:
        sweep_id = f"baseline_sweep_{_now()}"
        root = Path("experiments") / "baseline_sweep" / sweep_id

    cfg_dir = root / "configs"
    results_dir = root / "results"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] id: {sweep_id}")
    print(f"[sweep] variants sampled: {len(chosen)} / {len(all_variants)}")

    if args.dry_run:
        for i, v in enumerate(chosen, 1):
            print(f"  trial_{i:03d}: {v}")
        return 0

    trial_rows: List[Dict[str, Any]] = []
    start_idx = 1

    progress_path = results_dir / "progress.json"
    if args.resume_dir and progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                trial_rows = json.load(f)
            done_trials = [int(r.get("trial", 0)) for r in trial_rows if isinstance(r.get("trial"), int)]
            start_idx = (max(done_trials) + 1) if done_trials else 1
            print(f"[sweep] resume mode: found {len(trial_rows)} recorded trial(s), continuing from trial {start_idx}")
        except Exception as e:
            print(f"[sweep][warn] could not load existing progress.json: {e}")

    if start_idx > len(chosen):
        print("[sweep] nothing to run: all requested trials already completed.")
        return 0

    for i in range(start_idx, len(chosen) + 1):
        var = chosen[i - 1]
        cfg = copy.deepcopy(base_cfg)

        base_name = _get_in(cfg, "baseline_finetune.experiment_name") or "baseline"
        trial_name = f"{base_name}_sweep_t{i:03d}"
        trial_out = f"targets/distilgpt2/runs/{trial_name}"

        _set_in(cfg, "baseline_finetune.experiment_name", trial_name)
        _set_in(cfg, "baseline_finetune.output_dir", trial_out)
        _set_in(cfg, "baseline_finetune.train.collect_hooks", bool(args.collect_hooks))
        _set_in(cfg, "baseline_finetune.train.metric_for_best_model", "accuracy")
        _set_in(cfg, "baseline_finetune.train.greater_is_better", True)
        # Disk-safe defaults for sweeps: no checkpoints
        _set_in(cfg, "baseline_finetune.train.save_strategy", "no")
        _set_in(cfg, "baseline_finetune.train.load_best_model_at_end", False)
        _set_in(cfg, "baseline_finetune.train.save_total_limit", 1)

        for k, v in var.items():
            _set_in(cfg, k, v)

        trial_cfg_path = cfg_dir / f"trial_{i:03d}.yaml"
        _write_yaml(trial_cfg_path, cfg)

        print(f"\n[sweep] ({i}/{len(chosen)}) running {trial_name}")
        rc = run_trial(trial_cfg_path)

        row: Dict[str, Any] = {
            "trial": i,
            "trial_name": trial_name,
            "return_code": rc,
            "config": var,
            "output_dir": trial_out,
            "metric": None,
            "summary_path": None,
        }

        if rc == 0:
            summary_path = Path(trial_out) / "summary_main.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                m = score_from_summary(summary, args.metric)
                row["metric"] = m
                row["summary_path"] = str(summary_path)
                print(f"[sweep] metric {args.metric} = {m}")
            else:
                print(f"[sweep][warn] summary not found: {summary_path}")
        else:
            print(f"[sweep][warn] trial failed with rc={rc}")

        # Keep only config + result row by default; delete trial artifacts to save disk
        if not args.keep_trial_artifacts:
            trial_out_path = Path(trial_out)
            if trial_out_path.exists():
                try:
                    shutil.rmtree(trial_out_path, ignore_errors=True)
                except Exception as e:
                    print(f"[sweep][warn] cleanup failed for {trial_out_path}: {e}")
            # summary file was consumed into row['metric']; avoid stale path references
            row["summary_path"] = None

        trial_rows.append(row)

        with (results_dir / "progress.json").open("w", encoding="utf-8") as f:
            json.dump(trial_rows, f, indent=2)

    valid = [r for r in trial_rows if isinstance(r.get("metric"), (int, float))]
    best = None
    if valid:
        best = max(valid, key=lambda r: r["metric"]) if maximize else min(valid, key=lambda r: r["metric"])

    summary = {
        "sweep_id": sweep_id,
        "base_config": str(base_path),
        "metric": args.metric,
        "maximize": maximize,
        "num_trials": len(trial_rows),
        "num_valid": len(valid),
        "best": best,
        "trials": trial_rows,
    }

    out = results_dir / "summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[sweep] summary: {out}")
    if best:
        print(f"[sweep] best: {best['trial_name']} -> {args.metric}={best['metric']}")
    else:
        print("[sweep] no successful trial with metric")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
