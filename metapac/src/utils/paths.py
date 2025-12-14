# metapac/src/utils/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathRegistry:
    repo_root: Path

    @property
    def metapac_root(self) -> Path:
        return self.repo_root / "metapac"

    @property
    def runs_dir(self) -> Path:
        return self.metapac_root / "runs"

    @property
    def results_dir(self) -> Path:
        return self.metapac_root / "results"

    @property
    def artifacts_dir(self) -> Path:
        return self.metapac_root / "artifacts"

    @property
    def meta_dataset_path(self) -> Path:
        return self.artifacts_dir / "meta_dataset" / "meta_dataset.parquet"

    def ensure_dirs(self) -> None:
        for p in [self.runs_dir, self.results_dir, self.artifacts_dir, self.meta_dataset_path.parent]:
            p.mkdir(parents=True, exist_ok=True)
