# metapac/src/utils/debug_compare_weights.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForSequenceClassification


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "targets").exists() and (cur / "metapac").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()


def must_exist(p: Path, name: str):
    if not p.exists():
        print(f"[error] {name} nem található: {p}")
        sys.exit(1)


def load_reference_model(finetuned_dir: Path):
    ref = AutoModelForSequenceClassification.from_pretrained(
        finetuned_dir.as_posix(), local_files_only=True
    )
    return ref.eval()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() != b.numel():
        return -1.0
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def load_quantized_model_wrapper(comp_dir: Path, finetuned_dir: Path, baseline_dir: Path):
    from metapac.src.compression.load_quantized_model import load_quantized_model

    # prefer FINETUNED weights as axis/reference
    if (finetuned_dir / "model.safetensors").exists():
        ref_sf = (finetuned_dir / "model.safetensors").as_posix()
    elif (finetuned_dir / "pytorch_model.bin").exists():
        ref_sf = (finetuned_dir / "pytorch_model.bin").as_posix()
    else:
        ref_sf = None

    # baseline optional (legacy)
    if (baseline_dir / "model.safetensors").exists():
        base_sf = (baseline_dir / "model.safetensors").as_posix()
    elif (baseline_dir / "pytorch_model.bin").exists():
        base_sf = (baseline_dir / "pytorch_model.bin").as_posix()
    else:
        base_sf = None

    return load_quantized_model(
        comp_dir.as_posix(),
        device="cpu",
        skeleton_dir=finetuned_dir.as_posix(),
        baseline_state_dict_path=base_sf,
        reference_state_dict_path=ref_sf,
    ).eval()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, help="Override repository root")
    ap.add_argument("--baseline", type=str, help="Baseline model dir (weights)")
    ap.add_argument("--finetuned", type=str, help="Finetuned dir (has config.json)")
    ap.add_argument("--compressed", type=str, help="Compressed export dir")
    return ap.parse_args()


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path(__file__).parent)
    default_baseline = repo_root / "targets" / "distilbert" / "models" / "baseline"
    default_finetuned = repo_root / "targets" / "distilbert" / "models" / "experiments" / "kd_test_v4" / "finetuned"
    default_compressed = repo_root / "targets" / "distilbert" / "models" / "experiments" / "kd_test_v4" / "compressed"

    baseline_dir = Path(args.baseline or default_baseline)
    finetuned_dir = Path(args.finetuned or default_finetuned)
    comp_dir = Path(args.compressed or default_compressed)

    print("[info] REPO_ROOT:", repo_root.as_posix())
    print("[info] baseline_dir:", baseline_dir.as_posix())
    print("[info] finetuned_dir:", finetuned_dir.as_posix())
    print("[info] compressed_dir:", comp_dir.as_posix())

    must_exist(baseline_dir, "baseline dir")
    must_exist(finetuned_dir, "finetuned dir")
    must_exist(comp_dir, "compressed dir")

    reference_model = load_reference_model(finetuned_dir)
    q_model = load_quantized_model_wrapper(comp_dir, finetuned_dir, baseline_dir)

    ref_sd = dict(reference_model.state_dict())
    q_sd = dict(q_model.state_dict())

    rows = []
    with torch.no_grad():
        for k in sorted(set(ref_sd) & set(q_sd)):
            wa = ref_sd[k]
            wb = q_sd[k]
            if not (isinstance(wa, torch.Tensor) and isinstance(wb, torch.Tensor)):
                continue
            if wa.numel() == 0 or wb.numel() == 0:
                continue
            c = cosine(wa, wb)
            ct = None
            if wb.ndim == 2 and wa.shape == wb.t().shape:
                ct = cosine(wa, wb.t())
            rows.append((k, c, ct))

    rows.sort(key=lambda x: (x[1] if x[1] is not None else -1.0))
    print("\n=== Worst 20 cosine matches (FINETUNED vs COMPRESSED) ===")
    for k, c, ct in rows[:20]:
        if ct is None:
            print(f"{k:60s} cos={c:.4f}")
        else:
            print(f"{k:60s} cos={c:.4f}  cos(T)={ct:.4f}")


if __name__ == "__main__":
    main()
