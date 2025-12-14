"""
CLI entry for MetaPAC meta-baseline/compression workflows.
Tries to import a project-specific pipeline if available; otherwise runs a safe no-op.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pathlib
import sys


def _load_config(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            cfg = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    cfg[k.strip()] = v.strip()
            return cfg


def _maybe_call_project_pipeline(config: dict) -> int:
    target_mods = [
        ("metapac.src.pipeline", "run"),
        ("metapac.src.entrypoints.meta", "run"),
    ]
    for mod_name, fn_name in target_mods:
        spec = importlib.util.find_spec(mod_name)
        if spec is not None:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                if callable(fn):
                    return int(fn(config) or 0)
    print("[run_meta] No project pipeline found. Parsed config keys:", list(config.keys()))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MetaPAC runner")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--print-config", action="store_true", help="Print parsed config and exit")
    args, unknown = parser.parse_known_args(argv)

    cfg = _load_config(args.config)
    if args.print_config:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return 0

    runner = cfg.get("runner", None)
    if runner:
        spec = importlib.util.find_spec(runner)
        if spec is None:
            print(f"[run_meta] Requested runner '{runner}' not found; falling back to auto-detect.", file=sys.stderr)
        else:
            mod = importlib.import_module(runner)
            if hasattr(mod, "run"):
                return int(mod.run(cfg) or 0)
            print(f"[run_meta] Runner '{runner}' has no callable 'run'; falling back.", file=sys.stderr)

    return _maybe_call_project_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
