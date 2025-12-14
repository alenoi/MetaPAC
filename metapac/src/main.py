# File: metapac/src/main.py
"""
Thin entrypoint to enable:
    python -m metapac.src.main --config metapac/configs/meta_baseline_best.yaml
Delegates to CLI runner.
"""
from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    # Lazily import to keep import graph minimal for tooling
    from metapac.src.cli.run_meta import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
