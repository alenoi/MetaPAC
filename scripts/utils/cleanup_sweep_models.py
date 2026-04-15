#!/usr/bin/env python3
"""
Cleanup Sweep Model Files
==========================
Deletes model checkpoint files (.pt, .safetensors, etc.) from sweep experiments
to save disk space while keeping configs and results for reproducibility.
"""

import os
import sys
from pathlib import Path


def cleanup_sweep_models(sweep_dir: Path, dry_run: bool = False):
    """Delete model files from a sweep directory.
    
    Args:
        sweep_dir: Path to sweep directory (e.g., experiments/config_sweep/sweep_20251022_142953)
        dry_run: If True, only print what would be deleted without actually deleting
    """
    models_dir = sweep_dir / "models"

    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        return

    # Extensions to delete
    model_extensions = ['.pt', '.pth', '.bin', '.safetensors', '.onnx', '.h5']

    total_size_mb = 0
    total_files = 0

    print(f"[INFO] Scanning {models_dir}")
    print(f"[INFO] Model extensions: {', '.join(model_extensions)}")
    print(f"[INFO] Dry run: {dry_run}\n")

    # Walk through all experiments
    for exp_dir in sorted(models_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_size_mb = 0
        exp_files = []

        # Find all model files
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    file_path = Path(root) / file
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    exp_size_mb += file_size_mb
                    exp_files.append((file_path, file_size_mb))

        if exp_files:
            print(f"[{exp_dir.name}] {len(exp_files)} files, {exp_size_mb:.1f} MB")

            if not dry_run:
                for file_path, size_mb in exp_files:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"  [WARNING] Could not delete {file_path.name}: {e}")

                print(f"  → Deleted {len(exp_files)} files")

            total_size_mb += exp_size_mb
            total_files += len(exp_files)

    print(f"\n{'=' * 80}")
    print(f"[SUMMARY]")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")

    if dry_run:
        print(f"  [DRY RUN] No files were deleted")
        print(f"  Run without --dry-run to actually delete files")
    else:
        print(f"  [DONE] {total_size_mb:.1f} MB freed")
    print(f"{'=' * 80}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup sweep model files")
    parser.add_argument(
        'sweep_dir',
        type=str,
        help='Path to sweep directory (e.g., experiments/config_sweep/sweep_20251022_142953)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Clean up all sweeps in experiments/config_sweep/'
    )

    args = parser.parse_args()

    if args.all:
        # Clean up all sweeps
        sweeps_root = Path("experiments/config_sweep")
        if not sweeps_root.exists():
            print(f"[ERROR] Sweeps root not found: {sweeps_root}")
            return 1

        sweep_dirs = sorted([d for d in sweeps_root.iterdir() if d.is_dir() and d.name.startswith('sweep_')])

        if not sweep_dirs:
            print(f"[INFO] No sweep directories found in {sweeps_root}")
            return 0

        print(f"[INFO] Found {len(sweep_dirs)} sweep directories\n")

        for sweep_dir in sweep_dirs:
            print(f"\n{'=' * 80}")
            print(f"[SWEEP] {sweep_dir.name}")
            print(f"{'=' * 80}")
            cleanup_sweep_models(sweep_dir, args.dry_run)
    else:
        # Clean up single sweep
        sweep_path = Path(args.sweep_dir)
        if not sweep_path.exists():
            print(f"[ERROR] Sweep directory not found: {sweep_path}")
            return 1

        cleanup_sweep_models(sweep_path, args.dry_run)

    return 0


if __name__ == '__main__':
    sys.exit(main())
