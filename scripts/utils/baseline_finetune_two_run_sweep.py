#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
    cur = d
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _run(cfg_path: Path) -> int:
    cmd = [sys.executable, "-m", "metapac", "--config", str(cfg_path)]
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def _load_summary(output_dir: Path) -> Dict[str, Any]:
    p = output_dir / "summary_main.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-run baseline finetune sweep: global optimum vs t131")
    ap.add_argument("--base-config", default="metapac/configs/baseline_finetune_distilgpt2_imdb_fast.yaml")
    ap.add_argument("--output-root", default="experiments/baseline_sweep")
    args = ap.parse_args()

    root = Path(args.output_root)
    sweep_id = f"two_run_global_vs_t131_{_now()}"
    sweep_dir = root / sweep_id
    cfg_dir = sweep_dir / "configs"
    results_dir = sweep_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_yaml(Path(args.base_config))

    trials: List[Dict[str, Any]] = [
        {
            "name": "global_optimum_t029",
            "params": {
                "baseline_finetune.train.learning_rate": 5e-05,
                "baseline_finetune.train.weight_decay": 0.0,
                "baseline_finetune.train.num_train_epochs": 3,
                "baseline_finetune.train.per_device_train_batch_size": 64,
                "baseline_finetune.train.per_device_eval_batch_size": 64,
                "baseline_finetune.train.gradient_accumulation_steps": 4,
                "baseline_finetune.train.warmup_ratio": 0.03,
            },
        },
        {
            "name": "best_accuracy_t131",
            "params": {
                "baseline_finetune.train.learning_rate": 5e-05,
                "baseline_finetune.train.weight_decay": 0.01,
                "baseline_finetune.train.num_train_epochs": 5,
                "baseline_finetune.train.per_device_train_batch_size": 64,
                "baseline_finetune.train.per_device_eval_batch_size": 128,
                "baseline_finetune.train.gradient_accumulation_steps": 4,
                "baseline_finetune.train.warmup_ratio": 0.03,
            },
        },
    ]

    rows = []
    print(f"[two-run] sweep_id: {sweep_id}")

    for i, t in enumerate(trials, 1):
        cfg = copy.deepcopy(base_cfg)
        trial_name = f"baseline_distilgpt2_imdb_{t['name']}"
        output_dir = Path("targets/distilgpt2/runs") / trial_name

        _set_in(cfg, "mode", "baseline_finetune")
        _set_in(cfg, "baseline_finetune.experiment_name", trial_name)
        _set_in(cfg, "baseline_finetune.output_dir", str(output_dir).replace('\\', '/'))

        # keep this as pure train+eval benchmark
        _set_in(cfg, "baseline_finetune.train.collect_hooks", False)
        _set_in(cfg, "baseline_finetune.train.metric_for_best_model", "accuracy")
        _set_in(cfg, "baseline_finetune.train.greater_is_better", True)

        for k, v in t["params"].items():
            _set_in(cfg, k, v)

        cfg_path = cfg_dir / f"{i:02d}_{t['name']}.yaml"
        _write_yaml(cfg_path, cfg)

        print(f"[two-run] ({i}/2) running {t['name']}")
        rc = _run(cfg_path)

        sm = _load_summary(output_dir)
        row = {
            "name": t["name"],
            "config_path": str(cfg_path).replace('\\', '/'),
            "output_dir": str(output_dir).replace('\\', '/'),
            "return_code": rc,
            "params": t["params"],
            "eval_accuracy": sm.get("eval", {}).get("eval_accuracy"),
            "eval_f1": sm.get("eval", {}).get("eval_f1"),
            "train_runtime": sm.get("train", {}).get("train_runtime"),
            "max_cuda_mem_mb": sm.get("max_cuda_mem_mb"),
        }
        rows.append(row)

        with (results_dir / "progress.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    valid = [r for r in rows if isinstance(r.get("eval_accuracy"), (int, float))]
    best = max(valid, key=lambda r: r["eval_accuracy"]) if valid else None

    out = {
        "sweep_id": sweep_id,
        "description": "Two-run sweep: global optimum (t029) vs best-accuracy profile (t131)",
        "base_config": args.base_config,
        "runs": rows,
        "best_by_accuracy": best,
    }

    out_path = results_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[two-run] summary: {out_path}")
    if best:
        print(f"[two-run] winner: {best['name']} acc={best['eval_accuracy']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
