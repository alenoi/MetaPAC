# file: metapac/src/utils/experiment_report.py
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def _safe_json_load(p: Path) -> Optional[Dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _sha256_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _try_git_meta(root: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root), stderr=subprocess.DEVNULL, text=True
        ).strip()
        return branch, commit
    except Exception:
        return None, None

def _torch_meta() -> Dict[str, str]:
    meta = {}
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_device_count"] = str(torch.cuda.device_count())
            meta["cuda_current_device"] = str(torch.cuda.current_device())
            meta["cuda_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        meta["torch_version"] = "unavailable"
    return meta

def _gather_known_summaries(root: Path) -> List[str]:
    lines: List[str] = []
    # compression summary
    comp = root / "compressed" / "compression_summary.json"
    if comp.exists():
        js = _safe_json_load(comp)
        if js:
            lines.append("[compression_summary.json]")
            for k in ("accuracy_comp", "accuracy_base", "accuracy_delta", "quantized_count"):
                if k in js:
                    lines.append(f"  {k}: {js[k]}")
            zs = js.get("zones") or {}
            if zs:
                lines.append(f"  zones: keep={len(zs.get('keep', []))} "
                             f"quantize={len(zs.get('quantize', []))} prune={len(zs.get('prune', []))}")
            lines.append("")
    # common training artifacts
    for name in ("metrics.json", "trainer_state.json", "training.log"):
        p = root / name
        if p.exists():
            if name.endswith(".json"):
                js = _safe_json_load(p)
                lines.append(f"[{name}]")
                if js:
                    # print only top-level scalars
                    for k, v in js.items():
                        if isinstance(v, (int, float, str, bool)) and len(lines) - 1 < 20:
                            lines.append(f"  {k}: {v}")
                lines.append("")
            else:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                    tail = txt[-10:] if len(txt) > 10 else txt
                    lines.append(f"[{name} :: tail]")
                    lines += ["  " + t for t in tail]
                    lines.append("")
                except Exception:
                    pass
    # tensorboard events (just list)
    tfevents = list(root.glob("**/events.out.tfevents*"))
    if tfevents:
        lines.append("[tensorboard events]")
        for ev in tfevents[:20]:
            lines.append(f"  {ev.relative_to(root)} ({_fmt_bytes(ev.stat().st_size)})")
        if len(tfevents) > 20:
            lines.append(f"  ... and {len(tfevents) - 20} more")
        lines.append("")
    return lines

def _extension_counts(files: List[Path], root: Path) -> List[str]:
    from collections import Counter
    c = Counter([p.suffix.lower() or "<noext>" for p in files])
    total = sum(c.values())
    lines = ["[by extension]"]
    for ext, cnt in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {ext:>10}: {cnt}")
    lines.append(f"  {'TOTAL':>10}: {total}")
    lines.append("")
    return lines

def generate_experiment_report(
    experiment_dir: Path,
    out_file: Optional[Path] = None,
    include_hashes: bool = False,
) -> Path:
    """
    Generate a human-readable text report about an experiment directory.
    The report includes: environment info, optional git meta, known summaries,
    recursive file listing with size/mtime (and optional SHA-256), and totals.
    """
    experiment_dir = Path(experiment_dir).resolve()
    assert experiment_dir.exists(), f"Experiment dir does not exist: {experiment_dir}"

    if out_file is None:
        out_file = experiment_dir / "EXPERIMENT_REPORT.txt"
    else:
        out_file = Path(out_file)

    # 1) Header & environment
    header: List[str] = []
    header.append("=" * 100)
    header.append(f"EXPERIMENT REPORT :: {experiment_dir}")
    header.append("=" * 100)
    header.append(f"generated_utc: {datetime.now(timezone.utc).isoformat()}")
    header.append(f"hostname     : {platform.node()}")
    header.append(f"os           : {platform.platform()}")
    header.append(f"python       : {platform.python_version()}")
    tm = _torch_meta()
    for k, v in tm.items():
        header.append(f"{k:13}: {v}")

    # 2) Git metadata (best-effort, walks up to repo root)
    repo_root = None
    cur = experiment_dir
    for _ in range(6):  # walk up to 6 levels
        if (cur / ".git").exists():
            repo_root = cur
            break
        cur = cur.parent
    if repo_root:
        br, cm = _try_git_meta(repo_root)
        header.append(f"git_branch   : {br or 'n/a'}")
        header.append(f"git_commit   : {cm or 'n/a'}")
    header.append("")

    # 3) Known summaries
    known = _gather_known_summaries(experiment_dir)

    # 4) Recursive file listing
    files: List[Path] = [p for p in experiment_dir.rglob("*") if p.is_file()]
    files_sorted = sorted(files, key=lambda p: str(p.relative_to(experiment_dir)))
    listing: List[str] = []
    listing.append("[files]")
    total_size = 0
    for p in files_sorted:
        st = p.stat()
        rel = p.relative_to(experiment_dir)
        total_size += st.st_size
        line = f"  {rel}  |  size={_fmt_bytes(st.st_size)}  mtime={_iso(st.st_mtime)}"
        if include_hashes and st.st_size <= (1 << 28):  # <= 256MB safeguard
            try:
                line += f"  sha256={_sha256_of_file(p)}"
            except Exception:
                line += "  sha256=<error>"
        listing.append(line)
    listing.append("")

    # 5) Extension summary + totals
    ext_summary = _extension_counts(files_sorted, experiment_dir)
    totals = [
        "[totals]",
        f"  file_count : {len(files_sorted)}",
        f"  total_size : {_fmt_bytes(total_size)}",
        "",
    ]

    # 6) Write report
    content = "\n".join(header + known + ext_summary + listing + totals)
    out_file.write_text(content, encoding="utf-8")
    return out_file
