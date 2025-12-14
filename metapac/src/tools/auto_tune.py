# src/tools/auto_tune.py
"""
Lightweight auto-tuner for MetaPAC without external dependencies.

It generates trial configs from a base YAML, writes them under runs/sweeps/<stamp>/,
invokes:   python -m src.meta.train_meta --config <trial.yaml>
collects:  runs/<experiment_name>_metrics.json
and selects the best trial by the chosen objective (default: test_uncalib.rmse).

Usage:
  python -m src.tools.auto_tune --base configs/meta_baseline.yaml --trials 16 --mode random
  python -m src.tools.auto_tune --base configs/meta_baseline.yaml --mode grid
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# -------------------------
# Utilities
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def read_metrics(run_name: str) -> Dict[str, Any] | None:
    p = Path(f'"metapac/runs/{run_name}_metrics.json"')
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def objective_from_metrics(metrics: Dict[str, Any], key: str) -> float | None:
    """
    Extract objective value from metrics JSON.
    Default key 'test_uncalib.rmse' → lower is better.
    """
    try:
        if "." in key:
            node = metrics
            for k in key.split("."):
                node = node[k]
            return float(node)
        return float(metrics[key])
    except Exception:
        return None


def call_training(config_path: str, python_exe: str = sys.executable, module: str = "src.meta.train_meta") -> int:
    """Run: python -m src.meta.train_meta --config <config>"""
    cmd = [python_exe, "-m", module, "--config", config_path]
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc.returncode


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# -------------------------
# Search spaces
# -------------------------
def default_search_space() -> Dict[str, List[Any]]:
    """
    Define a sensible small search space.
    You can edit these lists freely; grid mode will do Cartesian product,
    random mode will sample uniformly from each.
    """
    return {
        # training hypers
        "training.lr": [1e-3, 5e-4, 2.5e-4],
        "training.weight_decay": [1e-4, 5e-5, 1e-5],
        "training.batch_size": [256, 512],
        "training.loss": ["huber"],
        "training.huber_delta": [1.0, 0.5, 0.25],
        "training.early_stop_patience": [8, 10, 12],

        # model hypers
        "model.hidden_sizes": [
            [256, 128, 64],
            [512, 256, 128],
            [512, 256, 64]
        ],
        "model.dropout": [0.05, 0.10, 0.15],
        "model.activation": ["relu", "gelu"],

        # feature handling (optional: narrow to grad_* only)
        # Turn on to enforce grad-only features:
        # "data.auto_infer.prefixes": [["grad_"]],
    }


def set_in(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in nested dict using dotted key, creating intermediate dicts."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def trial_variants_grid(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    trials = []
    for combo in itertools.product(*values):
        t = dict(zip(keys, combo))
        trials.append(t)
    return trials


def trial_variants_random(space: Dict[str, List[Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(space.keys())
    trials = []
    for _ in range(n):
        t = {k: rng.choice(space[k]) for k in keys}
        trials.append(t)
    return trials


# -------------------------
# Main autotune routine
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="Base YAML config path")
    ap.add_argument("--trials", type=int, default=16, help="Number of trials (for random mode)")
    ap.add_argument("--mode", type=str, choices=["random", "grid"], default="random", help="Search mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", type=str, default="test_uncalib.rmse",
                    help="Metrics JSON dotted key; lower is better")
    ap.add_argument("--python", type=str, default=sys.executable)
    args = ap.parse_args()

    base_cfg = load_yaml(args.base)
    # Safety defaults
    base_cfg.setdefault("training", {}).setdefault("calibration_enabled", False)
    base_cfg["training"].setdefault("num_workers", 4)
    base_cfg["training"].setdefault("device", "cuda")

    space = default_search_space()
    if args.mode == "grid":
        variants = trial_variants_grid(space)
    else:
        variants = trial_variants_random(space, n=args.trials, seed=args.seed)

    sweep_id = f"sweep-{timestamp()}"
    sweep_dir = Path("runs") / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, Dict[str, Any], float]] = []
    print(f"[tune] starting sweep {sweep_id} with {len(variants)} trials (mode={args.mode})")

    for i, var in enumerate(variants, 1):
        trial_cfg = copy.deepcopy(base_cfg)

        # Unique experiment name per trial
        base_name = base_cfg.get("experiment_name", "meta_baseline")
        trial_name = f"{base_name}_t{i:03d}_{int(time.time())}"
        trial_cfg["experiment_name"] = trial_name

        # Apply variant overrides
        for k, v in var.items():
            set_in(trial_cfg, k, v)

        # Write trial config
        trial_cfg_path = sweep_dir / f"{trial_name}.yaml"
        dump_yaml(trial_cfg, str(trial_cfg_path))

        print(f"[tune] ({i}/{len(variants)}) running {trial_name} ...")
        rc = call_training(str(trial_cfg_path), python_exe=args.python)
        if rc != 0:
            print(f"[tune][warn] trial {trial_name} exited with code {rc}; skipping.")
            continue

        # Read metrics & objective
        metrics = read_metrics(trial_name)
        if metrics is None:
            print(f"[tune][warn] metrics missing for {trial_name}; skipping.")
            continue
        obj = objective_from_metrics(metrics, args.objective)
        if obj is None:
            print(f"[tune][warn] objective '{args.objective}' not found in metrics for {trial_name}; skipping.")
            continue

        results.append((trial_name, metrics, obj))
        print(f"[tune] {trial_name} → {args.objective} = {obj:.6f}")

    if not results:
        print("[tune][fatal] no successful trials.")
        return

    # Select best (lower is better)
    results.sort(key=lambda x: x[2])
    best_name, best_metrics, best_obj = results[0]
    print("\n================= BEST TRIAL =================")
    print(f"Name: {best_name}")
    print(f"Objective ({args.objective}): {best_obj:.6f}")
    print(json.dumps(best_metrics, indent=2))
    print("==============================================")

    # Save sweep summary
    summary = {
        "sweep_id": sweep_id,
        "base_config": args.base,
        "mode": args.mode,
        "objective": args.objective,
        "results": [
            {"name": n, "objective": float(o), "metrics": m} for (n, m, o) in results
        ],
        "best": {"name": best_name, "objective": float(best_obj), "metrics": best_metrics},
    }
    out_json = sweep_dir / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[tune] summary saved to: {out_json}")


if __name__ == "__main__":
    main()
