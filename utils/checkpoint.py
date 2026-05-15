"""Checkpoint utilities — save and restore model + optimizer state."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(
        f"Loaded checkpoint from {path} "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"val_acc={ckpt.get('metrics', {}).get('val_acc', '?'):.4f})"
    )
    return ckpt
