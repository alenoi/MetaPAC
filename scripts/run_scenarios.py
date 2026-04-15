#!/usr/bin/env python
"""
Run all scenario configs sequentially.

Usage (PowerShell):
  python scripts/run_scenarios.py
  python scripts/run_scenarios.py --max-samples 200
  python scripts/run_scenarios.py --only quant_vb_headroom_on prune_proxy_physical_50

Notes:
- Adds/overrides compression.validation.max_samples when --max-samples is provided.
- Continues on failure, prints a concise summary at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import yaml

from metapac.src.pipeline import run as run_pipeline
from metapac.src.compression.pipeline.config_manager import load_strategy_defaults, merge_with_defaults

SCENARIO_DIR = Path("metapac/configs/scenarios").resolve()


def load_and_merge_config(path: Path, max_samples: int | None) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Ensure mode present
    cfg.setdefault("mode", cfg.get("mode", "compress"))
    # Inject validation override if requested
    if max_samples is not None:
        comp = cfg.setdefault("compression", {})
        val = comp.setdefault("validation", {})
        # Only set if not already set by the scenario
        val.setdefault("max_samples", int(max_samples))
        # Prefer a smaller batch size for quicker CPU runs if unset
        val.setdefault("batch_size", 32)
        val.setdefault("disable_progress", True)
    return cfg


def discover_scenarios(only: List[str] | None) -> List[Path]:
    paths = sorted(SCENARIO_DIR.glob("*.yaml"))
    if only:
        filt = set(only)
        paths = [p for p in paths if any(tag in p.stem for tag in filt)]
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Override validation.max_samples for faster runs (e.g., 200)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Filter scenarios by substring of filename stem")
    args = ap.parse_args()

    scenarios = discover_scenarios(args.only)
    if not scenarios:
        print(f"No scenarios found under {SCENARIO_DIR}")
        return 1

    results: List[Dict[str, Any]] = []
    for cfg_path in scenarios:
        label = cfg_path.stem
        print("=" * 80)
        print(f"[run_scenarios] Running scenario: {label}")
        print("=" * 80)
        try:
            cfg = load_and_merge_config(cfg_path, args.max_samples)
            mc = cfg.get('compression', {}).get('meta_checkpoint')
            print(f"[run_scenarios] meta_checkpoint (user cfg): {mc}")
            defaults = load_strategy_defaults().get('compression', {})
            merged = merge_with_defaults(defaults, cfg.get('compression', {}))
            print(f"[run_scenarios] meta_checkpoint (merged): {merged.get('meta_checkpoint')}")
            # Ensure each scenario writes to a unique experiment folder
            comp = cfg.setdefault('compression', {})
            if not comp.get('output_dir'):
                comp['output_dir'] = str(Path('targets') / 'distilbert' / 'models' / 'experiments' / label)
            print(f"[run_scenarios] output_dir: {comp.get('output_dir')}")
            code = run_pipeline(cfg)
            results.append({"scenario": label, "status": "ok" if code == 0 else f"fail({code})"})
        except Exception as e:
            print(f"[run_scenarios] ERROR in {label}: {e}")
            results.append({"scenario": label, "status": f"error({type(e).__name__})"})

    print("\n" + "-" * 80)
    print("[run_scenarios] Summary:")
    for r in results:
        print(f"  - {r['scenario']}: {r['status']}")

    failures = [r for r in results if r['status'] != 'ok']
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
