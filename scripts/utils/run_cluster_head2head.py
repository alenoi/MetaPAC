#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


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


def _parse_metrics(output_dir: Path) -> Dict[str, Any]:
    val_path = output_dir / "compressed" / "validation_results.json"
    comp_path = output_dir / "artifacts" / "phase4_export" / "compression_summary.json"
    if not val_path.exists():
        return {"error": f"Missing {val_path}"}

    val = json.loads(val_path.read_text(encoding="utf-8"))
    cmp_node = val.get("comparison", {})
    base_node = val.get("baseline", {})
    sub_node = val.get("subject", {})

    out = {
        "baseline_accuracy": float(base_node.get("accuracy", 0.0)),
        "compressed_accuracy": float(sub_node.get("accuracy", 0.0)),
        "accuracy_drop": float(cmp_node.get("accuracy_drop", 0.0)),
        "accuracy_drop_pct": float(cmp_node.get("accuracy_drop_pct", 0.0)),
        "compression_degree_pct": float(cmp_node.get("file_bytes_reduction_pct", 0.0)),
        "file_reduction_bytes": int(cmp_node.get("file_bytes_reduction", 0) or 0),
        "compression_ratio_variable_bit": None,
    }

    if comp_path.exists():
        csum = json.loads(comp_path.read_text(encoding="utf-8"))
        out["compression_ratio_variable_bit"] = float(
            csum.get("variable_bit_quantization", {}).get("compression_ratio", 0.0) or 0.0
        )

    return out


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    base_cfg_path = repo_root / "metapac" / "configs" / "auto_distilgpt2_imdb_fast.yaml"
    base_cfg = _read_yaml(base_cfg_path)

    run_id = f"cluster_head2head_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = repo_root / "experiments" / "compression_sweep" / run_id
    cfg_root = run_root / "configs"
    results_path = run_root / "results.json"

    top3_kmeans_param_sets: List[Dict[str, Any]] = [
        {
            "compression.zone_assignment.kmeans.n_clusters": 3,
            "compression.zone_assignment.kmeans.n_init": 10,
            "compression.zone_assignment.kmeans.min_high_fraction": 0.18,
            "compression.pruning.sparsity_level": 0.25,
            "compression.pruning.min_importance_threshold": 0.004,
            "compression.quantization.bits_lower": 4,
            "compression.quantization.bits_upper": 8,
            "compression.quantization.layer_overrides": [],
            "compression.kd.alpha": 0.7,
            "compression.kd.temperature": 4.0,
        },
        {
            "compression.zone_assignment.kmeans.n_clusters": 3,
            "compression.zone_assignment.kmeans.n_init": 20,
            "compression.zone_assignment.kmeans.min_high_fraction": 0.18,
            "compression.pruning.sparsity_level": 0.2,
            "compression.pruning.min_importance_threshold": 0.004,
            "compression.quantization.bits_lower": 4,
            "compression.quantization.bits_upper": 8,
            "compression.quantization.layer_overrides": [
                {"pattern": ".*attn.*", "bits": 8},
                {"pattern": ".*score.*", "bits": 8},
            ],
            "compression.kd.alpha": 0.9,
            "compression.kd.temperature": 2.0,
        },
        {
            "compression.zone_assignment.kmeans.n_clusters": 3,
            "compression.zone_assignment.kmeans.n_init": 20,
            "compression.zone_assignment.kmeans.min_high_fraction": 0.18,
            "compression.pruning.sparsity_level": 0.2,
            "compression.pruning.min_importance_threshold": 0.004,
            "compression.quantization.bits_lower": 4,
            "compression.quantization.bits_upper": 8,
            "compression.quantization.layer_overrides": [
                {"pattern": ".*attn.*", "bits": 8},
                {"pattern": ".*score.*", "bits": 8},
            ],
            "compression.kd.alpha": 0.5,
            "compression.kd.temperature": 2.0,
        },
    ]

    methods = ["kmeans", "gmm"]
    trial_rows: List[Dict[str, Any]] = []

    trial_idx = 0
    for base_i, pset in enumerate(top3_kmeans_param_sets, start=1):
        for method in methods:
            trial_idx += 1
            trial_id = f"t{trial_idx:02d}_base{base_i}_{method}"
            cfg = copy.deepcopy(base_cfg)
            cfg["mode"] = "auto:compress"

            out_dir = run_root / "tmp_trials" / trial_id
            _set_in(cfg, "compression.output_dir", str(out_dir).replace("\\", "/"))
            _set_in(cfg, "compression.zone_assignment.method", method)

            for k, v in pset.items():
                _set_in(cfg, k, v)

            cfg_path = cfg_root / f"{trial_id}.yaml"
            _write_yaml(cfg_path, cfg)

            cmd = [sys.executable, "-m", "metapac", "--mode", "auto:compress", "--config", str(cfg_path)]
            proc = subprocess.run(cmd, text=True)

            row: Dict[str, Any] = {
                "trial_id": trial_id,
                "base_param_set": base_i,
                "method": method,
                "return_code": int(proc.returncode),
                "config_path": str(cfg_path),
                "output_dir": str(out_dir),
                "params": {"compression.zone_assignment.method": method, **pset},
            }

            if proc.returncode == 0:
                row.update(_parse_metrics(out_dir))
            else:
                row["error"] = "Compression process returned non-zero exit code"

            trial_rows.append(row)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps({"run_id": run_id, "trials": trial_rows}, indent=2), encoding="utf-8")

    # Method-level quick summary
    summary = {"run_id": run_id, "num_trials": len(trial_rows), "trials": trial_rows}
    good = [r for r in trial_rows if r.get("return_code") == 0 and r.get("accuracy_drop_pct") is not None]
    for m in methods:
        rows = [r for r in good if r.get("method") == m]
        if rows:
            summary[f"{m}_avg_accuracy_drop_pct"] = sum(float(r["accuracy_drop_pct"]) for r in rows) / len(rows)
            summary[f"{m}_avg_compression_degree_pct"] = sum(float(r["compression_degree_pct"]) for r in rows) / len(rows)

    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[head2head] completed: {run_id}")
    print(f"[head2head] summary: {run_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
