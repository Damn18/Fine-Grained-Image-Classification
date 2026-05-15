"""
Shared training engine used by all per-model train scripts.

The training loop reproduces the setup described in the paper:
  - AdamW optimizer with decoupled weight decay
  - Warmup + cosine annealing LR scheduler (batch-level stepping)
  - LabelSmoothingCrossEntropy loss
  - Early stopping on validation loss
  - Per-epoch checkpointing (best model + latest)
  - Logging via wandb / CSV

All five train scripts (train_vit_slow.py, train_vit_fast.py, etc.) call
run_training() after building their respective model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import EarlyStopping, Logger, load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _build_criterion(config: dict) -> nn.Module:
    smoothing = config.get("label_smoothing", 0.0)
    if smoothing > 0.0:
        try:
            from timm.loss import LabelSmoothingCrossEntropy
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        except ImportError:
            # Fallback to PyTorch's built-in (available since torch 1.10)
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
    return nn.CrossEntropyLoss()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config.get("optimizer", "adamw").lower()
    lr = float(config["lr"])
    wd = float(config.get("weight_decay", 0.01))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=wd,
        )
    raise ValueError(f"Unknown optimizer: '{name}'. Use 'adamw' or 'sgd'.")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Returns (scheduler, is_batch_level)."""
    name = config.get("scheduler", "cosine_warmup").lower()

    if name == "cosine_warmup":
        from transformers import get_cosine_schedule_with_warmup

        total_steps = config["num_epochs"] * steps_per_epoch
        warmup_steps = int(config.get("warmup_steps", 500))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return scheduler, True  # batch-level

    if name == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.1),
        )
        return scheduler, False  # epoch-level

    if name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("factor", 0.1),
            patience=config.get("plateau_patience", 3),
        )
        return scheduler, False  # epoch-level (needs val_loss)

    raise ValueError(f"Unknown scheduler: '{name}'. Use cosine_warmup, step_lr, or reduce_on_plateau.")


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, criterion, device, scheduler=None, batch_level=False):
    model.train()
    total_loss = correct = total = 0

    for inputs, labels in tqdm(loader, desc="  train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_level and scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = top5_correct = total = 0

    for inputs, labels in tqdm(loader, desc="  val  ", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        k = min(5, outputs.size(1))
        _, top5 = outputs.topk(k, dim=1)
        top5_correct += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, top5_correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    logger: Logger,
    checkpoint_dir: str | Path,
    resume_from: str | Path | None = None,
) -> List[dict]:
    """
    Train model for config['num_epochs'] epochs.

    Args:
        model:           The model to train (already on CPU; moved to device internally).
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        config:          Parsed YAML config dict.
        logger:          Logger instance (wandb + CSV).
        checkpoint_dir:  Directory for saving checkpoints.
        resume_from:     Optional path to a checkpoint to resume training from.

    Returns:
        List of per-epoch metric dicts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] Using device: {device}")
    model = model.to(device)

    # Reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    criterion = _build_criterion(config)
    optimizer = _build_optimizer(model, config)
    scheduler, batch_level = _build_scheduler(optimizer, config, len(train_loader))
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 5))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if resume_from is not None:
        ckpt = load_checkpoint(model, resume_from, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1

    best_val_acc = 0.0
    history: List[dict] = []

    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler if batch_level else None,
            batch_level=batch_level,
        )
        val_loss, val_acc, val_acc5 = _val_epoch(model, val_loader, criterion, device)

        # Epoch-level scheduler update
        if not batch_level:
            if config.get("scheduler") == "reduce_on_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "val_acc5":   round(val_acc5,   6),
            "lr":         current_lr,
        }
        history.append(metrics)
        logger.log(metrics)

        # Save latest checkpoint (overwrite each epoch to save disk)
        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir / "latest.pth")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir / "best.pth")
            print(f"  -> New best val_acc: {best_val_acc:.4f}")

        # Early stopping check
        if early_stopping(val_loss):
            print(f"[trainer] Early stopping triggered at epoch {epoch + 1}. "
                  f"Best val_acc = {best_val_acc:.4f}")
            break

    logger.finish()
    return history
