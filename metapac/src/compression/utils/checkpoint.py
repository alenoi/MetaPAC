"""Checkpoint resolution and selection utilities.

This module provides utilities for resolving and selecting model checkpoints:
- Local path resolution with fallback
- Checkpoint selection by mode (best/last/exact)
- Meta-predictor checkpoint resolution
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def extract_checkpoint_step(path: Path) -> int:
    """Extract numeric step from checkpoint directory name.

    Args:
        path: Path to checkpoint directory (e.g., checkpoint-1000)

    Returns:
        Checkpoint step number, or -1 if not parseable.
    """
    try:
        return int(path.name.split("checkpoint-")[-1])
    except Exception:
        return -1


def latest_checkpoint_in_dir(run_dir: Path) -> Optional[Path]:
    """Return latest checkpoint-* directory under run_dir, if any.

    Args:
        run_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    candidates = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=extract_checkpoint_step)


def best_checkpoint_in_dir(run_dir: Path) -> Optional[Path]:
    """Resolve best checkpoint from trainer_state.json if available.

    Reads best_model_checkpoint from the latest checkpoint's trainer_state.json.
    Falls back to latest checkpoint if state is missing/invalid.

    Args:
        run_dir: Directory containing checkpoint subdirectories

    Returns:
        Path to best checkpoint, or None if no checkpoints found.
    """
    latest = latest_checkpoint_in_dir(run_dir)
    if latest is None:
        return None

    state_path = latest / "trainer_state.json"
    if not state_path.exists():
        return latest

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        best_ref = state.get("best_model_checkpoint")
        if not best_ref:
            return latest

        best_path = Path(best_ref)
        candidates: List[Path] = []
        if best_path.is_absolute():
            candidates.append(best_path)
        else:
            candidates.append((Path.cwd() / best_path).resolve())
            candidates.append((run_dir / best_path).resolve())
            candidates.append((run_dir.parent / best_path).resolve())
            if best_path.name.startswith("checkpoint-"):
                candidates.append((run_dir / best_path.name).resolve())

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
    except Exception as exc:
        logger.warning(f"Failed to read best checkpoint from {state_path}: {exc}")

    return latest


def resolve_local_model_dir(
    model_ref: Optional[str], 
    allow_checkpoint_fallback: bool = True
) -> Optional[str]:
    """Resolve local model directories while preserving HF model IDs.

    Returns absolute local path when ``model_ref`` exists on disk. If it points to a
    missing ``checkpoint-*`` directory and fallback is enabled, returns the latest
    sibling checkpoint. Otherwise returns original ``model_ref`` unchanged.

    Args:
        model_ref: Model path or HuggingFace model ID
        allow_checkpoint_fallback: If True, fall back to latest checkpoint when missing

    Returns:
        Resolved model path or original reference.
    """
    if not model_ref:
        return model_ref

    path = Path(model_ref)
    if path.exists():
        return str(path.resolve())

    if allow_checkpoint_fallback and path.name.startswith("checkpoint-"):
        parent = path.parent
        if parent.exists() and parent.is_dir():
            candidates = [p for p in parent.glob("checkpoint-*") if p.is_dir()]
            if candidates:
                latest = max(candidates, key=extract_checkpoint_step)
                logger.info(
                    f"Checkpoint fallback: {model_ref} -> {latest}"
                )
                return str(latest.resolve())

    return model_ref


def select_checkpoint(
    model_ref: Optional[str],
    mode: Optional[str],
    exact_step: Optional[int],
) -> Optional[str]:
    """Select checkpoint path by mode: best/last/exact.

    Args:
        model_ref: Base model path or checkpoint directory
        mode: Selection mode - 'best', 'last', or 'exact'
        exact_step: Checkpoint step for 'exact' mode

    Returns:
        Path to selected checkpoint, or original model_ref if mode is None.

    Notes:
        - If mode is unset, returns model_ref unchanged.
        - For exact mode, when checkpoint-{exact_step} is missing, falls back to last.
    """
    if not model_ref:
        return model_ref

    selected_mode = str(mode or "").strip().lower()
    if selected_mode not in {"best", "last", "exact"}:
        return model_ref

    ref_path = Path(model_ref)
    if ref_path.name.startswith("checkpoint-"):
        run_dir = ref_path.parent
    else:
        run_dir = ref_path

    if not run_dir.exists() or not run_dir.is_dir():
        logger.warning(
            f"Checkpoint selection skipped; run directory not found: {run_dir}"
        )
        return model_ref

    if selected_mode == "last":
        last_ckpt = latest_checkpoint_in_dir(run_dir)
        if last_ckpt is not None:
            return str(last_ckpt.resolve())
        return model_ref

    if selected_mode == "best":
        best_ckpt = best_checkpoint_in_dir(run_dir)
        if best_ckpt is not None:
            return str(best_ckpt.resolve())
        return model_ref

    # exact mode
    step = None
    try:
        step = int(exact_step) if exact_step is not None else None
    except Exception:
        step = None

    if step is not None:
        exact_path = run_dir / f"checkpoint-{step}"
        if exact_path.exists() and exact_path.is_dir():
            return str(exact_path.resolve())
        logger.warning(
            f"Exact checkpoint not found: {exact_path}; falling back to latest"
        )

    last_ckpt = latest_checkpoint_in_dir(run_dir)
    if last_ckpt is not None:
        return str(last_ckpt.resolve())
    return model_ref


def resolve_meta_checkpoint_dir(checkpoint_ref: Optional[str]) -> Optional[str]:
    """Resolve meta-predictor checkpoint path.

    Supports both exact checkpoint paths and prefix-style references like
    ``metapac/runs/checkpoints/metapac_meta_qwen3_wos_fast`` by selecting the
    latest matching timestamped checkpoint directory.

    Args:
        checkpoint_ref: Path to meta-predictor checkpoint or prefix

    Returns:
        Resolved checkpoint path, or original reference if not found.
    """
    if not checkpoint_ref:
        return checkpoint_ref

    ref_path = Path(checkpoint_ref)
    if ref_path.exists():
        return str(ref_path.resolve())

    parent = ref_path.parent
    stem = ref_path.name
    if not parent.exists() or not parent.is_dir():
        return checkpoint_ref

    candidates = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(f"{stem}_"):
            continue
        if (p / "model_state.pt").exists() and (p / "feature_names.json").exists():
            candidates.append(p)

    if not candidates:
        return checkpoint_ref

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info(
        f"Resolved meta checkpoint by prefix: {checkpoint_ref} -> {latest}"
    )
    return str(latest.resolve())
