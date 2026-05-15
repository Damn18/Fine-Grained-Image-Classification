"""
Experiment logger: Weights & Biases + CSV fallback.

Usage:
  logger = Logger(config)
  logger.log({"epoch": 0, "train_loss": 0.5, "val_acc": 0.72})
  logger.finish()

If wandb is not installed or not authenticated, the logger falls back to
console output and a CSV file saved in the 'logs/' directory.

To authenticate wandb via GitHub: run `wandb login --relogin` in your terminal.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Logger:
    def __init__(self, config: dict):
        self._use_wandb = config.get("use_wandb", True) and _WANDB_AVAILABLE
        run_name = config.get("run_name", "run")
        log_dir = Path(config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = log_dir / f"{run_name}_{timestamp}.csv"
        self._csv_file = None
        self._csv_writer = None

        if self._use_wandb:
            try:
                _wandb.init(
                    project=config.get("wandb_project", "fine-grained-clf"),
                    name=run_name,
                    config=config,
                )
            except Exception as e:
                print(f"[Logger] wandb init failed ({e}). Falling back to CSV.")
                self._use_wandb = False
        elif not _WANDB_AVAILABLE and config.get("use_wandb", True):
            print("[Logger] wandb not installed. Logging to CSV only.")

    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any]) -> None:
        if self._use_wandb:
            _wandb.log(metrics)

        # CSV
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(metrics.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

        # Console
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print("  ".join(parts))

    def finish(self) -> None:
        if self._use_wandb:
            _wandb.finish()
        if self._csv_file:
            self._csv_file.close()
            print(f"[Logger] Training log saved to {self._csv_path}")
