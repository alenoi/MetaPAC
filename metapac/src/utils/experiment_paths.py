# File: metapac/src/utils/experiment_paths.py
# NEW FILE – centralizes experiment folder naming and ensures backward compatibility.

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentDirs:
    root: Path
    pruned: Path
    finetuned: Path
    quantized: Path


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _migrate_if_exists(src: Path, dst: Path) -> None:
    """
    If an old-named folder already exists and the new one doesn't, move it to the new standardized name.
    If both exist, do nothing (assume the new structure already in use).
    """
    if src.exists() and not dst.exists():
        # move content preserving files
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
        except Exception:
            # best-effort: copytree then keep original
            shutil.copytree(src, dst, dirs_exist_ok=True)


def resolve_experiment_dirs(experiment_root: Path) -> ExperimentDirs:
    """
    Create and/or migrate experiment subfolders to standardized names:
      01_pruned, 02_finetuned, 03_quantized
    Backward compatibility with legacy names:
      pruned_before_quant -> 01_pruned
      finetuned           -> 02_finetuned
      compressed          -> 03_quantized
    """
    experiment_root = Path(experiment_root)

    # Standardized names
    pruned_std = experiment_root / "01_pruned"
    finetuned_std = experiment_root / "02_finetuned"
    quantized_std = experiment_root / "03_quantized"

    # Legacy names to migrate from (if present)
    legacy_pruned = experiment_root / "pruned_before_quant"
    legacy_finetuned = experiment_root / "finetuned"
    legacy_quantized = experiment_root / "compressed"

    # Migrate if needed
    _migrate_if_exists(legacy_pruned, pruned_std)
    _migrate_if_exists(legacy_finetuned, finetuned_std)
    _migrate_if_exists(legacy_quantized, quantized_std)

    # Ensure dirs exist
    _mkdir(experiment_root)
    _mkdir(pruned_std)
    _mkdir(finetuned_std)
    _mkdir(quantized_std)

    return ExperimentDirs(
        root=experiment_root,
        pruned=pruned_std,
        finetuned=finetuned_std,
        quantized=quantized_std,
    )
