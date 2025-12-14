# src/meta/utils/logger.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False
    Console = None
    Table = None
    Progress = None


@dataclass
class LogRow:
    epoch: int
    step: int
    train_mse: float
    val_mae: float
    val_rmse: float
    val_spearman: float
    lr: float
    elapsed_s: float
    improved: bool


class TrainLogger:
    def __init__(self, run_dir: str = "runs", run_name: str = "meta_baseline", use_progress: bool = False):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, f"{run_name}_train_log.csv")
        self.jsonl_path = os.path.join(self.run_dir, f"{run_name}_train_log.jsonl")
        self._csv_inited = False

        # env override: METAPAC_NO_RICH=1 disables rich
        no_rich = os.getenv("METAPAC_NO_RICH", "") == "1"
        self._use_progress = (use_progress and _HAS_RICH and not no_rich)

        self._console = Console() if (_HAS_RICH and not no_rich) else None
        self._progress = None
        self._task_id = None

    def start_epoch(self, epoch: int, total_steps: int) -> None:
        if self._use_progress:
            self._progress = Progress(
                TextColumn("[bold]Epoch {task.fields[epoch]}[/]"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("• lr {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
                transient=True,
            )
            self._task_id = self._progress.add_task("train", total=total_steps, epoch=epoch, lr=0.0)
            self._progress.start()

    def advance(self, step: int, lr: float) -> None:
        if self._use_progress and self._progress is not None:
            self._progress.update(self._task_id, completed=step, lr=lr)

    def end_epoch(self, row: LogRow) -> None:
        flag = "*best*" if row.improved else ""
        line = (
            f"Epoch {row.epoch:03d} | "
            f"train_mse={row.train_mse:.3e} | "
            f"val_mae={row.val_mae:.3e} | "
            f"val_rmse={row.val_rmse:.3e} | "
            f"val_spear={row.val_spearman:.3e} | "
            f"lr={row.lr:.2e} | "
            f"elapsed={row.elapsed_s:.1f}s {flag}"
        )
        print(line)
        self._write_files(row)

    def _write_files(self, row: LogRow) -> None:
        d = asdict(row)
        # CSV
        if not self._csv_inited:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(d.keys()))
                w.writeheader()
                w.writerow(d)
            self._csv_inited = True
        else:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=list(d.keys())).writerow(d)
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    def print_test(self, metrics: Dict[str, Any]) -> None:
        if _HAS_RICH and self._console:
            table = Table(title="[bold]TEST metrics[/bold]")
            for k in ("mae", "rmse", "spearman"):
                table.add_column(k.upper())
            table.add_row(f"{metrics['mae']:.3e}", f"{metrics['rmse']:.3e}", f"{metrics['spearman']:.3e}")
            self._console.print(table)
        else:
            print(f"[TEST] MAE={metrics['mae']:.3e} RMSE={metrics['rmse']:.3e} Spearman={metrics['spearman']:.3e}")
