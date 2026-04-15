#!/usr/bin/env python3
"""
Smart compression parameter sweep for distilgpt2/imdb (or any auto:compress config).

Goals:
- Tune compression parameters (NOT baseline fine-tune hyperparameters).
- Keep only trial config + numeric result artifacts for sweep bookkeeping.
- Use an adaptive search:
  1) Per-parameter scale probing (coordinate-style), keep top-K values per parameter.
  2) Combination search over top-K values with early stopping on no-improvement.

Result quality is evaluated by:
- compression_degree_pct (higher is better)
- accuracy_drop_pct (lower is better)
- score = w_compression * compression_degree_pct - w_accuracy_drop * accuracy_drop_pct
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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# -----------------------------
# generic helpers
# -----------------------------

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


def _canonical_signature_value(v: Any) -> str:
    """Return stable hashable representation for nested sweep values."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _apply_special_params(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply non-trivial sweep parameters and return remaining simple dotted params.

    Supported aliases:
    - compression.zone_assignment.protected_fraction
    - compression.zone_assignment.quantize_fraction
      -> translated to compression.zones.{high,medium,low}.quantile_* boundaries
    """
    remaining = dict(params)

    prot_key = "compression.zone_assignment.protected_fraction"
    quant_key = "compression.zone_assignment.quantize_fraction"

    if prot_key in remaining and quant_key in remaining:
        protected = float(remaining.pop(prot_key))
        quantize = float(remaining.pop(quant_key))

        # Convert keep/quantize ratio into quantile boundaries; prune is remainder.
        protected = max(0.0, min(1.0, protected))
        quantize = max(0.0, min(1.0, quantize))
        if protected + quantize > 1.0:
            quantize = max(0.0, 1.0 - protected)

        low_max = max(0.0, 1.0 - protected - quantize)
        med_min = low_max
        med_max = max(med_min, 1.0 - protected)
        high_min = med_max

        _set_in(cfg, "compression.zones.low.quantile_min", 0.0)
        _set_in(cfg, "compression.zones.low.quantile_max", float(low_max))
        _set_in(cfg, "compression.zones.medium.quantile_min", float(med_min))
        _set_in(cfg, "compression.zones.medium.quantile_max", float(med_max))
        _set_in(cfg, "compression.zones.high.quantile_min", float(high_min))
        _set_in(cfg, "compression.zones.high.quantile_max", 1.0)

    # Ensure quantization lower/upper bits are valid together if both are present.
    lower_key = "compression.quantization.bits_lower"
    upper_key = "compression.quantization.bits_upper"
    lower_val = remaining.get(lower_key, _get_in(cfg, lower_key))
    upper_val = remaining.get(upper_key, _get_in(cfg, upper_key))
    if lower_val is not None and upper_val is not None:
        lower_i = int(lower_val)
        upper_i = int(upper_val)
        if lower_i > upper_i:
            lower_i, upper_i = upper_i, lower_i
        remaining[lower_key] = lower_i
        remaining[upper_key] = upper_i

    return remaining


# -----------------------------
# sweep data model
# -----------------------------

@dataclass
class TrialResult:
    trial_id: str
    stage: str
    return_code: int
    params: Dict[str, Any]
    output_dir: str
    baseline_accuracy: float | None = None
    compressed_accuracy: float | None = None
    accuracy_drop: float | None = None
    accuracy_drop_pct: float | None = None
    compression_degree_pct: float | None = None
    file_reduction_bytes: int | None = None
    compression_ratio_variable_bit: float | None = None
    score: float | None = None
    error: str | None = None


# -----------------------------
# config loading
# -----------------------------

def load_sweep_spec(path: Path) -> Dict[str, Any]:
    cfg = _read_yaml(path)

    params = cfg.get("parameters")
    if not isinstance(params, dict) or not params:
        raise ValueError("Sweep config must contain non-empty 'parameters' mapping")

    for k, v in params.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid parameter key: {k}")
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Parameter '{k}' must map to a non-empty list")

    spec = {
        "parameters": params,
        "parameter_order": cfg.get("parameter_order", list(params.keys())),
        "topk_per_parameter": int(cfg.get("topk_per_parameter", 3)),
        "max_combination_trials": int(cfg.get("max_combination_trials", 60)),
        "min_improvement": float(cfg.get("min_improvement", 0.05)),
        "patience": int(cfg.get("patience", 12)),
        "weights": {
            "compression": float(cfg.get("weights", {}).get("compression", 1.0)),
            "accuracy_drop": float(cfg.get("weights", {}).get("accuracy_drop", 1.0)),
        },
    }

    # Normalize order to existing keys only.
    ordered = [k for k in spec["parameter_order"] if k in params]
    for k in params.keys():
        if k not in ordered:
            ordered.append(k)
    spec["parameter_order"] = ordered

    return spec


# -----------------------------
# evaluation / execution
# -----------------------------

def compute_score(compression_pct: float, accuracy_drop_pct: float, w_comp: float, w_drop: float) -> float:
    return (w_comp * compression_pct) - (w_drop * accuracy_drop_pct)


def parse_trial_metrics(output_dir: Path, w_comp: float, w_drop: float) -> Dict[str, Any]:
    val_path = output_dir / "compressed" / "validation_results.json"
    comp_path = output_dir / "artifacts" / "phase4_export" / "compression_summary.json"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation results: {val_path}")

    val = json.loads(val_path.read_text(encoding="utf-8"))
    cmp_node = val.get("comparison", {})
    base_node = val.get("baseline", {})
    sub_node = val.get("subject", {})

    compression_degree_pct = float(cmp_node.get("file_bytes_reduction_pct", 0.0))
    accuracy_drop = float(cmp_node.get("accuracy_drop", 0.0))
    accuracy_drop_pct = float(cmp_node.get("accuracy_drop_pct", 0.0))

    ratio_var = None
    if comp_path.exists():
        summary = json.loads(comp_path.read_text(encoding="utf-8"))
        ratio_var = float(summary.get("variable_bit_quantization", {}).get("compression_ratio", 0.0) or 0.0)

    score = compute_score(compression_degree_pct, accuracy_drop_pct, w_comp, w_drop)

    return {
        "baseline_accuracy": float(base_node.get("accuracy", 0.0)),
        "compressed_accuracy": float(sub_node.get("accuracy", 0.0)),
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_pct": accuracy_drop_pct,
        "compression_degree_pct": compression_degree_pct,
        "file_reduction_bytes": int(cmp_node.get("file_bytes_reduction", 0) or 0),
        "compression_ratio_variable_bit": ratio_var,
        "score": float(score),
    }


def run_compress_trial(
    base_cfg: Dict[str, Any],
    params: Dict[str, Any],
    trial_id: str,
    stage: str,
    sweep_root: Path,
    keep_trial_artifacts: bool,
    w_comp: float,
    w_drop: float,
) -> TrialResult:
    cfg = copy.deepcopy(base_cfg)

    # Force compress stage only for each trial.
    cfg["mode"] = "auto:compress"

    # Per-trial isolated output dir.
    trial_out = sweep_root / "tmp_trials" / trial_id
    _set_in(cfg, "compression.output_dir", str(trial_out).replace("\\", "/"))

    params_to_set = _apply_special_params(cfg, params)
    for k, v in params_to_set.items():
        _set_in(cfg, k, v)

    cfg_path = sweep_root / "configs" / f"{trial_id}.yaml"
    _write_yaml(cfg_path, cfg)

    cmd = [sys.executable, "-m", "metapac", "--mode", "auto:compress", "--config", str(cfg_path)]
    proc = subprocess.run(cmd, text=True)

    row = TrialResult(
        trial_id=trial_id,
        stage=stage,
        return_code=int(proc.returncode),
        params=params,
        output_dir=str(trial_out),
    )

    if proc.returncode == 0:
        try:
            metrics = parse_trial_metrics(trial_out, w_comp=w_comp, w_drop=w_drop)
            for k, v in metrics.items():
                setattr(row, k, v)
        except Exception as e:
            row.error = f"Metric parse failed: {e}"
    else:
        row.error = "Compression process returned non-zero exit code"

    if not keep_trial_artifacts:
        shutil.rmtree(trial_out, ignore_errors=True)

    return row


def pareto_front(trials: List[TrialResult]) -> List[TrialResult]:
    valid = [t for t in trials if t.score is not None]
    out: List[TrialResult] = []

    for t in valid:
        dominated = False
        for u in valid:
            if u is t:
                continue
            # u dominates t if: higher/equal compression and lower/equal accuracy_drop,
            # and strictly better in at least one objective.
            if (
                (u.compression_degree_pct or -1e9) >= (t.compression_degree_pct or -1e9)
                and (u.accuracy_drop_pct or 1e9) <= (t.accuracy_drop_pct or 1e9)
                and (
                    (u.compression_degree_pct or -1e9) > (t.compression_degree_pct or -1e9)
                    or (u.accuracy_drop_pct or 1e9) < (t.accuracy_drop_pct or 1e9)
                )
            ):
                dominated = True
                break
        if not dominated:
            out.append(t)

    out.sort(key=lambda x: (x.accuracy_drop_pct if x.accuracy_drop_pct is not None else 1e9,
                            -(x.compression_degree_pct if x.compression_degree_pct is not None else -1e9)))
    return out


# -----------------------------
# main adaptive search
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Smart compression sweep (adaptive, low-artifact)")
    ap.add_argument("--base-config", required=True, help="Base auto config (distilgpt2/imdb) YAML")
    ap.add_argument("--sweep-config", required=True, help="Compression sweep spec YAML")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-trial-artifacts", action="store_true",
                    help="Keep per-trial compression output dirs (default: delete)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_cfg = _read_yaml(Path(args.base_config))
    spec = load_sweep_spec(Path(args.sweep_config))

    params_space: Dict[str, List[Any]] = spec["parameters"]
    param_order: List[str] = spec["parameter_order"]
    topk = spec["topk_per_parameter"]
    max_combo_trials = spec["max_combination_trials"]
    min_improvement = spec["min_improvement"]
    patience = spec["patience"]
    w_comp = spec["weights"]["compression"]
    w_drop = spec["weights"]["accuracy_drop"]

    sweep_id = f"compression_smart_sweep_{_now()}"
    root = Path("experiments") / "compression_sweep" / sweep_id
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    print(f"[sweep] id: {sweep_id}")
    print(f"[sweep] base: {args.base_config}")
    print(f"[sweep] params: {len(params_space)} | topk={topk} | max_combo_trials={max_combo_trials}")

    # Start from base config values where possible.
    best_params: Dict[str, Any] = {}
    for k in param_order:
        base_val = _get_in(base_cfg, k)
        if base_val in params_space[k]:
            best_params[k] = base_val
        else:
            best_params[k] = params_space[k][0]

    if args.dry_run:
        print("[dry-run] parameter order:")
        for k in param_order:
            print(f"  - {k}: {params_space[k]}")
        print("[dry-run] initial best params:")
        for k in param_order:
            print(f"  - {k} = {best_params[k]}")
        print(f"[dry-run] output dir: {root}")
        return 0

    all_trials: List[TrialResult] = []
    trial_idx = 0

    def _save_progress() -> None:
        payload = {
            "sweep_id": sweep_id,
            "weights": {"compression": w_comp, "accuracy_drop": w_drop},
            "topk_per_parameter": topk,
            "max_combination_trials": max_combo_trials,
            "min_improvement": min_improvement,
            "patience": patience,
            "trials": [asdict(t) for t in all_trials],
        }
        (root / "results" / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Baseline trial with initial params.
    trial_idx += 1
    t0 = run_compress_trial(
        base_cfg=base_cfg,
        params=copy.deepcopy(best_params),
        trial_id=f"t{trial_idx:04d}_baseline",
        stage="baseline",
        sweep_root=root,
        keep_trial_artifacts=args.keep_trial_artifacts,
        w_comp=w_comp,
        w_drop=w_drop,
    )
    all_trials.append(t0)
    _save_progress()

    best_trial = t0 if t0.score is not None else None

    # Phase 1: per-parameter scale probing.
    top_values: Dict[str, List[Any]] = {}

    for p in param_order:
        param_trials: List[Tuple[Any, TrialResult]] = []
        print(f"\n[sweep][phase1] probing parameter: {p}")

        for v in params_space[p]:
            trial_params = copy.deepcopy(best_params)
            trial_params[p] = v

            trial_idx += 1
            tr = run_compress_trial(
                base_cfg=base_cfg,
                params=trial_params,
                trial_id=f"t{trial_idx:04d}_p1_{p.replace('.', '_')}",
                stage=f"phase1:{p}",
                sweep_root=root,
                keep_trial_artifacts=args.keep_trial_artifacts,
                w_comp=w_comp,
                w_drop=w_drop,
            )
            all_trials.append(tr)
            _save_progress()
            param_trials.append((v, tr))

            if tr.score is not None and (best_trial is None or tr.score > (best_trial.score or -1e9)):
                best_trial = tr

        valid_param_trials = [(v, t) for v, t in param_trials if t.score is not None]
        valid_param_trials.sort(key=lambda vt: vt[1].score, reverse=True)

        best_vals = [v for v, _ in valid_param_trials[:topk]]
        if not best_vals:
            # If all failed, keep current value as fallback.
            best_vals = [best_params[p]]
        top_values[p] = best_vals

        # Greedy coordinate update for next parameters.
        best_params[p] = best_vals[0]
        print(f"[sweep][phase1] top values for {p}: {best_vals}")

    # Phase 2: combinations of top-K per parameter.
    print("\n[sweep][phase2] testing parameter combinations")
    combo_keys = param_order
    combo_lists = [top_values[k] for k in combo_keys]
    all_combos = list(itertools.product(*combo_lists))

    # Ensure deterministic but manageable exploration order.
    rng.shuffle(all_combos)

    # Put greedy-top combo first.
    greedy_combo = tuple(top_values[k][0] for k in combo_keys)
    if greedy_combo in all_combos:
        all_combos.remove(greedy_combo)
    combo_queue = [greedy_combo] + all_combos

    tested_combo_signatures = set()
    no_improve = 0
    phase2_done = 0

    for combo in combo_queue:
        if phase2_done >= max_combo_trials:
            break

        combo_params = dict(zip(combo_keys, combo))
        sig = tuple((k, _canonical_signature_value(combo_params[k])) for k in combo_keys)
        if sig in tested_combo_signatures:
            continue
        tested_combo_signatures.add(sig)

        trial_idx += 1
        tr = run_compress_trial(
            base_cfg=base_cfg,
            params=combo_params,
            trial_id=f"t{trial_idx:04d}_p2_combo",
            stage="phase2:combo",
            sweep_root=root,
            keep_trial_artifacts=args.keep_trial_artifacts,
            w_comp=w_comp,
            w_drop=w_drop,
        )
        all_trials.append(tr)
        _save_progress()

        phase2_done += 1

        improved = False
        if tr.score is not None:
            if best_trial is None:
                best_trial = tr
                improved = True
            else:
                if tr.score > (best_trial.score or -1e9) + min_improvement:
                    best_trial = tr
                    improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[sweep][phase2] early stop: no improvement in {patience} consecutive combos")
            break

    valid_trials = [t for t in all_trials if t.score is not None]
    valid_trials.sort(key=lambda x: x.score, reverse=True)
    pareto = pareto_front(valid_trials)

    final = {
        "sweep_id": sweep_id,
        "base_config": args.base_config,
        "sweep_config": args.sweep_config,
        "weights": {"compression": w_comp, "accuracy_drop": w_drop},
        "topk_per_parameter": topk,
        "max_combination_trials": max_combo_trials,
        "min_improvement": min_improvement,
        "patience": patience,
        "num_trials": len(all_trials),
        "num_valid": len(valid_trials),
        "best_trial": asdict(best_trial) if best_trial else None,
        "top_values_per_parameter": top_values,
        "pareto_front": [asdict(t) for t in pareto],
        "trials": [asdict(t) for t in all_trials],
    }

    (root / "results" / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    # Compact CSV-like json list for quick processing.
    rows = []
    for t in all_trials:
        r = {
            "trial_id": t.trial_id,
            "stage": t.stage,
            "return_code": t.return_code,
            "compression_degree_pct": t.compression_degree_pct,
            "accuracy_drop_pct": t.accuracy_drop_pct,
            "score": t.score,
            "params": t.params,
        }
        rows.append(r)
    (root / "results" / "trials_min.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n[sweep] done: {root}")
    if best_trial is not None:
        print(f"[sweep] best trial: {best_trial.trial_id}")
        print(f"[sweep] best score: {best_trial.score:.4f}")
        print(f"[sweep] compression_degree_pct={best_trial.compression_degree_pct:.4f}, "
              f"accuracy_drop_pct={best_trial.accuracy_drop_pct:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
