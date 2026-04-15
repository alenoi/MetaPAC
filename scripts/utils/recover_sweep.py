#!/usr/bin/env python3
"""
Recover sweep results from a partially completed run.
Reads the model files and generates the results JSONs.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config_sweep import ConfigSweep, SweepConfig, SweepResults


def recover_sweep(sweep_dir: str):
    """Recover results from a sweep directory"""

    sweep_path = Path(sweep_dir)
    if not sweep_path.exists():
        print(f"❌ Sweep directory not found: {sweep_dir}")
        return

    print(f"🔄 Recovering sweep: {sweep_path.name}")

    # Create a temporary sweep object for helper methods
    sweep = ConfigSweep(output_dir=sweep_path.parent)
    sweep.sweep_dir = sweep_path
    sweep.configs_dir = sweep_path / "configs"
    sweep.models_dir = sweep_path / "models"
    sweep.results_dir = sweep_path / "results"

    # Find all config files
    config_files = sorted(sweep.configs_dir.glob("exp_*.yaml"))
    print(f"📄 Found {len(config_files)} config files")

    results = []

    for config_file in config_files:
        exp_id = config_file.stem
        print(f"\n🔍 Processing {exp_id}...")

        # Check if model directory exists
        exp_model_dir = sweep.models_dir / exp_id
        if not exp_model_dir.exists():
            print(f"  ⚠️  No model directory found, skipping")
            continue

        # Load the YAML config to reconstruct SweepConfig
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        # Extract sweep config parameters from YAML
        comp = cfg.get("compression", {})
        zones = comp.get("zones", {})
        quant = comp.get("quantization", {})
        prune = comp.get("pruning", {})
        ft = comp.get("fine_tuning", {})

        sweep_config = SweepConfig(
            exp_id=exp_id,
            timestamp=config_file.stat().st_mtime,
            high_zone_min=zones.get("high", {}).get("quantile_min", 0.7),
            medium_zone_min=zones.get("medium", {}).get("quantile_min", 0.3),
            bits_lower=quant.get("bits_lower", 4),
            bits_upper=quant.get("bits_upper", 8),
            util_target=quant.get("util_target", 0.98),
            pruning_enabled=prune.get("enabled", False),
            head_pruning_ratio=prune.get("head_pruning_ratio", 0.0),
            ffn_pruning_ratio=prune.get("ffn_pruning_ratio", 0.0),
            pruning_method=prune.get("method", "magnitude"),
            fine_tuning_enabled=ft.get("enabled", False),
            fine_tune_epochs=ft.get("training", {}).get("num_epochs", 0),
            fine_tune_lr=ft.get("training", {}).get("learning_rate", 0.0),
        )

        # Extract results
        sweep_config = sweep._extract_results(sweep_config)
        sweep_config.success = True

        results.append(sweep_config)

        # Save individual result
        result_file = sweep.results_dir / f"{exp_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(sweep_config), f, indent=2)

        print(f"  ✅ Recovered: {sweep_config.compression_ratio:.2f}x, acc={sweep_config.finetuned_accuracy:.2%}")

    if not results:
        print("\n❌ No results recovered")
        return

    # Analyze and save
    successful = [r for r in results if r.success]

    # Find best configs by different metrics
    best_accuracy = max(successful, key=lambda c: c.finetuned_accuracy) if successful else None
    best_compression = max(successful, key=lambda c: c.compression_ratio) if successful else None
    best_combined = max(successful,
                        key=lambda c: sweep.calculate_combined_score(c, 0.5, 0.5)) if successful else None
    best_efficiency = max(successful,
                          key=lambda c: sweep.calculate_efficiency_score(c)) if successful else None

    sweep_results = SweepResults(
        total_experiments=len(results),
        successful=len(successful),
        failed=len(results) - len(successful),
        best_accuracy=best_accuracy,
        best_compression=best_compression,
        best_combined=best_combined,
        best_efficiency=best_efficiency,
        pareto_optimal=sweep.find_pareto_optimal(results),
        all_configs=results
    )

    sweep.save_results(sweep_results)

    print(f"\n✅ Recovery complete!")
    print(f"📊 {len(results)} experiments recovered")
    print(f"📁 Results: {sweep.results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recover sweep results")
    parser.add_argument("sweep_dir",
                        help="Path to sweep directory (e.g., experiments/config_sweep/sweep_20251021_022753)")

    args = parser.parse_args()
    recover_sweep(args.sweep_dir)
