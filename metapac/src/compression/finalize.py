# metapac/src/compression/finalize.py
# Clean-up utility to move non-essential artifacts out of <experiment>/compressed
# while keeping everything needed to load and demo the quantized model in place.

from __future__ import annotations

import json
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List


def finalize_artifacts(
        experiment_dir: str | Path,
        *,
        keep_tokenizer: bool = True,
        primary_weight: str = "model_variable_bit.pt",
        dry_run: bool = False,
) -> Dict[str, List[str]]:
    """
    Post-process an experiment folder so that <experiment>/compressed contains only
    the files required to *load and run* the quantized model. Everything else is moved
    to <experiment>/artifacts/<phase_folder>.

    Parameters
    ----------
    experiment_dir : str | Path
        Path to the experiment root, e.g. ".../targets/distilbert/models/experiments/kd_test_v4".
    keep_tokenizer : bool
        If True, keep tokenizer/config assets in 'compressed' for fully self-contained demo.
        If False, move them under artifacts as they are not required if the loader uses HF/baseline assets.
    primary_weight : {"model_variable_bit.pt","model_state.pt"}
        Which quantized weight file should remain in 'compressed'. The other (if present) is archived.
    dry_run : bool
        If True, only print planned actions without moving files.

    Returns
    -------
    Dict[str, List[str]]
        A summary mapping of {"kept": [...], "moved": [...], "skipped": [...]} relative paths.
    """
    exp = Path(experiment_dir)
    compressed = exp / "compressed"
    artifacts = exp / "artifacts"

    compressed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Phase folders
    phase0_assets = artifacts / "phase0_assets"
    phase2_prune_ft = artifacts / "phase2_prune_ft"
    phase4_export = artifacts / "phase4_export"
    phase4_alt = phase4_export / "alt_format"
    phase5_validate = artifacts / "phase5_validate"
    misc_dir = artifacts / "misc"

    for d in (phase0_assets, phase2_prune_ft, phase4_export, phase4_alt, phase5_validate, misc_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Files that should remain in 'compressed' for an independent, demo-able quantized package.
    tokenizer_set = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    }
    keep_set = {
                   primary_weight,  # one quantized weight (pytorch_model.bin or model_state.pt)
                   # "model_int8.pt" removed - fake-quant INT8 export disabled
                   "variable_bit_meta.json",  # structural + quantization mapping needed by loader
                   "model_packed.bin",  # optional: packed bits for disk storage (5.45x compression)
                   "packing_metadata.json",  # required by load_packed_model
               } | (tokenizer_set if keep_tokenizer else set())

    # Candidates that are always reports or dev backups, never required to *run* the quantized model.
    # NOTE: pytorch_model.bin is NOT a backup when it's the primary_weight (variable-bit export)
    fp32_backups = {
        "model.safetensors",
        "_model_state_backup.pth",
        "model_state.pt",  # FP32 debug backup (unless it's primary_weight)
    }
    export_reports = {
        "compression_summary.json",
        "variable_bit_stats.json",
        "headroom_trim_meta.json",
        "pruning_meta.json",  # keep as report; loader should not need it
        "quant_meta.json",  # keep as report; loader should not need it
    }
    validation_outputs = {
        "validation_results.json",
    }

    # If both quantized formats exist, archive the non-primary to alt_format.
    quant_dual = {"model_variable_bit.pt", "model_state.pt"}

    moved, kept, skipped = [], [], []

    def _move(fname: str, dest_dir: Path) -> None:
        src = compressed / fname
        if not src.exists():
            return
        dst = dest_dir / fname
        if dry_run:
            print(f"[dry-run] MOVE {src.relative_to(exp)} -> {dst.relative_to(exp)}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(str(dst.relative_to(exp)))

    def _keep(fname: str) -> None:
        kept.append(str((compressed / fname).relative_to(exp)))

    def _is_keep(fname: str) -> bool:
        return fname in keep_set

    # First pass: ensure only one quantized weight stays
    if (compressed / "model_variable_bit.pt").exists() and (compressed / "model_state.pt").exists():
        other = (quant_dual - {primary_weight}).pop()
        _move(other, phase4_alt)

    # Go through all items under 'compressed'
    for p in sorted(compressed.iterdir()):
        if p.is_dir():
            # Keep subfolders intact unless they are known report folders (none expected by default)
            skipped.append(str(p.relative_to(exp)))
            continue

        fname = p.name

        if _is_keep(fname):
            _keep(fname)
            continue

        if fname in fp32_backups:
            _move(fname, phase2_prune_ft)
            continue

        if fname in export_reports:
            _move(fname, phase4_export)
            continue

        if fname in validation_outputs:
            _move(fname, phase5_validate)
            continue

        # Unknown files: treat conservatively — keep tokenizer/config if requested, otherwise archive to misc.
        if keep_tokenizer and fname in tokenizer_set:
            _keep(fname)
            continue

        # Default: archive to misc
        _move(fname, misc_dir)

    # Write a minimal manifest next to kept files for traceability
    manifest = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment": str(exp),
        "compressed_dir": str(compressed),
        "policy": {
            "keep_tokenizer": keep_tokenizer,
            "primary_weight": primary_weight,
        },
        "kept_files": sorted([Path(k).name for k in kept]),
        "notes": "Only files strictly required to load and run the quantized model are kept in 'compressed'. "
                 "All other artifacts are moved under 'artifacts/<phase_folder>/'.",
    }
    manifest_path = compressed / "manifest.json"
    if dry_run:
        print(f"[dry-run] WRITE {manifest_path.relative_to(exp)}")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return {"kept": kept, "moved": moved, "skipped": skipped}

# Example usage inside strategy.py, after Phase 5 validation succeeds:
#
# from metapac.src.compression.finalize import finalize_artifacts
# _ = finalize_artifacts(
#     experiment_dir=experiment_dir,
#     keep_tokenizer=True,                # set False if tokenizer/config loaded from HF or baseline
#     primary_weight="model_variable_bit.pt",
#     dry_run=False
# )
