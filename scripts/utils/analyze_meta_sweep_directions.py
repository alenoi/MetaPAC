#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def objective_col(summary: Dict[str, Any]) -> str:
    obj = str(summary.get("objective", "posthoc.spearman"))
    if obj in {"posthoc.spearman", "spearman", "metrics.spearman"}:
        return "spearman"
    if obj in {"metrics.rmse", "rmse"}:
        return "rmse"
    return "objective"


def direction_mode(summary: Dict[str, Any]) -> str:
    d = str(summary.get("objective_direction", "max")).lower().strip()
    return "max" if d not in {"min", "max"} else d


def flatten_trials(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in summary.get("trials", []):
        row = {
            "trial_id": t.get("trial_id"),
            "objective": t.get("objective"),
            "rmse": t.get("rmse"),
            "spearman": t.get("spearman"),
            "return_code": t.get("return_code"),
        }
        params = t.get("params", {}) or {}
        for k, v in params.items():
            row[f"param::{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_effects(df: pd.DataFrame, obj_col: str, mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    param_cols = [c for c in df.columns if c.startswith("param::")]

    valid = df[df["return_code"] == 0].copy()
    valid = valid[pd.to_numeric(valid[obj_col], errors="coerce").notna()].copy()
    valid[obj_col] = pd.to_numeric(valid[obj_col], errors="coerce")

    for col in param_cols:
        part = valid[[col, obj_col]].dropna().copy()
        if part.empty:
            continue
        vals = sorted(part[col].dropna().unique().tolist(), key=lambda x: str(x))
        if len(vals) < 2:
            continue

        stats = part.groupby(col, dropna=True)[obj_col].agg(["mean", "count"]).reset_index()
        stats = stats.sort_values("mean", ascending=(mode == "min")).reset_index(drop=True)
        best_val = stats.iloc[0][col]
        worst_val = stats.iloc[-1][col]
        effect = float(stats.iloc[0]["mean"] - stats.iloc[-1]["mean"]) if mode == "max" else float(stats.iloc[-1]["mean"] - stats.iloc[0]["mean"])

        rows.append({
            "parameter": col.replace("param::", ""),
            "best_value": best_val,
            "worst_value": worst_val,
            "n_values": int(len(vals)),
            "effect_strength": effect,
        })

    if not rows:
        return pd.DataFrame(columns=["parameter", "best_value", "worst_value", "n_values", "effect_strength"])
    return pd.DataFrame(rows).sort_values("effect_strength", ascending=False).reset_index(drop=True)


def build_followup_trials(effect_df: pd.DataFrame, max_trials: int) -> List[Dict[str, Any]]:
    if effect_df.empty:
        return []

    top = effect_df.head(4).to_dict(orient="records")
    if len(top) < 2:
        return []

    # Preferred values from stage-1
    preferred = {r["parameter"]: r["best_value"] for r in top}
    opposite = {r["parameter"]: r["worst_value"] for r in top}
    keys = list(preferred.keys())

    trials: List[Dict[str, Any]] = []

    # Exploit top pairs
    pairs = [
        (keys[0], keys[1]),
        (keys[0], keys[2]) if len(keys) > 2 else None,
        (keys[1], keys[2]) if len(keys) > 2 else None,
        (keys[0], keys[3]) if len(keys) > 3 else None,
        (keys[1], keys[3]) if len(keys) > 3 else None,
    ]
    pairs = [p for p in pairs if p is not None]

    for i, (a, b) in enumerate(pairs[:max_trials], start=1):
        params = {
            a: preferred[a],
            b: preferred[b],
        }
        # Keep one sanity counter-trial in first slot if enough budget
        if i == 1 and max_trials >= 4:
            params = {a: preferred[a], b: opposite[b]}
        trials.append({"id": f"f{i:02d}", "params": params})

    return trials[:max_trials]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze meta sweep and generate follow-up scenarios")
    ap.add_argument("--summary", required=True, help="Path to sweep summary.json")
    ap.add_argument("--out-csv", required=False, help="Optional path for parameter effect CSV")
    ap.add_argument("--emit-followup", required=False, help="Optional path for follow-up sweep YAML")
    ap.add_argument("--followup-trials", type=int, default=5, help="Number of follow-up trials (3-5 recommended)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    summary = load_summary(summary_path)
    obj_col = objective_col(summary)
    mode = direction_mode(summary)

    df = flatten_trials(summary)
    eff = summarize_effects(df, obj_col=obj_col, mode=mode)

    print("[analysis] objective:", summary.get("objective"), "direction:", mode)
    if eff.empty:
        print("[analysis] No analyzable parameter effects found.")
        return

    print("[analysis] parameter direction effects:")
    print(eff.to_string(index=False))

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        eff.to_csv(out_csv, index=False)
        print(f"[analysis] wrote CSV: {out_csv}")

    if args.emit_followup:
        n_follow = max(3, min(int(args.followup_trials), 5))
        trials = build_followup_trials(eff, max_trials=n_follow)
        payload = {
            "search_mode": "manual",
            "seed": 42,
            "objective": summary.get("objective", "posthoc.spearman"),
            "objective_direction": summary.get("objective_direction", "max"),
            "compute_posthoc_spearman": True,
            "trial_overrides": trials,
        }
        out_yaml = Path(args.emit_followup)
        out_yaml.parent.mkdir(parents=True, exist_ok=True)
        with out_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        print(f"[analysis] wrote follow-up config: {out_yaml}")


if __name__ == "__main__":
    main()
