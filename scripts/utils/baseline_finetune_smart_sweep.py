#!/usr/bin/env python3
"""
Smart (multi-fidelity) baseline fine-tune sweep.

Idea:
1) Sample candidate configs from search space
2) Run cheap proxy training (few epochs)
3) Keep top-K
4) Re-run top-K with full epochs

This reduces wall-time vs running all configs at full budget.
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


def load_space(config_path: str) -> Dict[str, List[Any]]:
    cfg = _read_yaml(Path(config_path))
    space = cfg.get("search_space", cfg)
    if not isinstance(space, dict) or not space:
        raise ValueError(f"Invalid sweep config format: {config_path}")
    out: Dict[str, List[Any]] = {}
    for k, v in space.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"Invalid parameter list for {k}")
        out[k] = v
    return out


def variants(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def run_trial(cfg_path: Path) -> int:
    cmd = [sys.executable, "-m", "metapac", "--config", str(cfg_path)]
    p = subprocess.run(cmd, text=True)
    return p.returncode


def score_from_summary(summary: Dict[str, Any], metric_key: str) -> float | None:
    try:
        node: Any = summary
        for k in metric_key.split("."):
            node = node[k]
        return float(node)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Smart multi-fidelity baseline sweep")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--metric", default="eval.eval_accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-candidates", type=int, default=48,
                    help="How many configs to evaluate in cheap stage")
    ap.add_argument("--proxy-epochs", type=float, default=2.0,
                    help="Epoch budget for cheap stage")
    ap.add_argument("--finalists", type=int, default=8,
                    help="How many top configs to re-run with full epochs")
    ap.add_argument("--keep-trial-artifacts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_cfg = _read_yaml(Path(args.base_config))
    space = load_space(args.sweep_config)
    all_variants = variants(space)

    rng = random.Random(args.seed)
    rng.shuffle(all_variants)
    candidates = all_variants[: min(args.max_candidates, len(all_variants))]

    sweep_id = f"smart_sweep_{_now()}"
    root = Path("experiments") / "baseline_sweep" / sweep_id
    cfg_dir = root / "configs"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smart] id: {sweep_id}")
    print(f"[smart] total search-space: {len(all_variants)}")
    print(f"[smart] stage1 candidates: {len(candidates)} | proxy_epochs={args.proxy_epochs}")
    print(f"[smart] stage2 finalists: {args.finalists}")

    if args.dry_run:
        for i, v in enumerate(candidates[:10], 1):
            print(f"  c{i:03d}: {v}")
        return 0

    stage1_rows: List[Dict[str, Any]] = []

    # Stage 1 (cheap)
    for i, var in enumerate(candidates, 1):
        cfg = copy.deepcopy(base_cfg)
        trial_name = f"smart_s1_t{i:03d}"
        trial_out = f"targets/distilgpt2/runs/{trial_name}"

        _set_in(cfg, "mode", "baseline_finetune")
        _set_in(cfg, "baseline_finetune.experiment_name", trial_name)
        _set_in(cfg, "baseline_finetune.output_dir", trial_out)
        _set_in(cfg, "baseline_finetune.train.collect_hooks", False)
        _set_in(cfg, "baseline_finetune.train.metric_for_best_model", "accuracy")
        _set_in(cfg, "baseline_finetune.train.greater_is_better", True)
        _set_in(cfg, "baseline_finetune.train.save_strategy", "no")
        _set_in(cfg, "baseline_finetune.train.load_best_model_at_end", False)
        _set_in(cfg, "baseline_finetune.train.save_total_limit", 1)

        for k, v in var.items():
            _set_in(cfg, k, v)

        original_epochs = _get_in(cfg, "baseline_finetune.train.num_train_epochs")
        if original_epochs is None:
            original_epochs = 3
        _set_in(cfg, "baseline_finetune.train.num_train_epochs", min(float(original_epochs), float(args.proxy_epochs)))

        cfg_path = cfg_dir / f"s1_trial_{i:03d}.yaml"
        _write_yaml(cfg_path, cfg)

        print(f"\n[smart][stage1] ({i}/{len(candidates)}) {trial_name}")
        rc = run_trial(cfg_path)

        metric = None
        sp = Path(trial_out) / "summary_main.json"
        if rc == 0 and sp.exists():
            summary = _read_yaml(sp) if sp.suffix in {'.yaml', '.yml'} else json.loads(sp.read_text(encoding='utf-8'))
            metric = score_from_summary(summary, args.metric)

        row = {
            "stage": 1,
            "idx": i,
            "trial_name": trial_name,
            "return_code": rc,
            "metric": metric,
            "config": var,
            "output_dir": trial_out,
        }
        stage1_rows.append(row)

        if not args.keep_trial_artifacts:
            shutil.rmtree(Path(trial_out), ignore_errors=True)

        (results_dir / "stage1_progress.json").write_text(json.dumps(stage1_rows, indent=2), encoding='utf-8')

    valid_s1 = [r for r in stage1_rows if isinstance(r.get("metric"), (int, float))]
    valid_s1.sort(key=lambda r: r["metric"], reverse=True)
    finalists = valid_s1[: min(args.finalists, len(valid_s1))]

    # Stage 2 (full)
    stage2_rows: List[Dict[str, Any]] = []
    for j, pick in enumerate(finalists, 1):
        var = pick["config"]
        cfg = copy.deepcopy(base_cfg)
        trial_name = f"smart_s2_t{j:03d}"
        trial_out = f"targets/distilgpt2/runs/{trial_name}"

        _set_in(cfg, "mode", "baseline_finetune")
        _set_in(cfg, "baseline_finetune.experiment_name", trial_name)
        _set_in(cfg, "baseline_finetune.output_dir", trial_out)
        _set_in(cfg, "baseline_finetune.train.collect_hooks", False)
        _set_in(cfg, "baseline_finetune.train.metric_for_best_model", "accuracy")
        _set_in(cfg, "baseline_finetune.train.greater_is_better", True)
        _set_in(cfg, "baseline_finetune.train.save_strategy", "no")
        _set_in(cfg, "baseline_finetune.train.load_best_model_at_end", False)
        _set_in(cfg, "baseline_finetune.train.save_total_limit", 1)

        for k, v in var.items():
            _set_in(cfg, k, v)

        cfg_path = cfg_dir / f"s2_trial_{j:03d}.yaml"
        _write_yaml(cfg_path, cfg)

        print(f"\n[smart][stage2] ({j}/{len(finalists)}) {trial_name}")
        rc = run_trial(cfg_path)

        metric = None
        sp = Path(trial_out) / "summary_main.json"
        if rc == 0 and sp.exists():
            summary = _read_yaml(sp) if sp.suffix in {'.yaml', '.yml'} else json.loads(sp.read_text(encoding='utf-8'))
            metric = score_from_summary(summary, args.metric)

        row = {
            "stage": 2,
            "idx": j,
            "trial_name": trial_name,
            "return_code": rc,
            "metric": metric,
            "config": var,
            "output_dir": trial_out,
            "from_stage1_metric": pick.get("metric"),
        }
        stage2_rows.append(row)

        if not args.keep_trial_artifacts:
            shutil.rmtree(Path(trial_out), ignore_errors=True)

        (results_dir / "stage2_progress.json").write_text(json.dumps(stage2_rows, indent=2), encoding='utf-8')

    valid_s2 = [r for r in stage2_rows if isinstance(r.get("metric"), (int, float))]
    best = max(valid_s2, key=lambda r: r["metric"]) if valid_s2 else None

    summary = {
        "sweep_id": sweep_id,
        "method": "multi_fidelity_successive_halving",
        "base_config": args.base_config,
        "sweep_config": args.sweep_config,
        "metric": args.metric,
        "stage1_candidates": len(candidates),
        "stage2_finalists": len(finalists),
        "best": best,
        "stage1": stage1_rows,
        "stage2": stage2_rows,
    }

    out = results_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\n[smart] summary: {out}")
    if best:
        print(f"[smart] best: {best['trial_name']} -> {args.metric}={best['metric']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
