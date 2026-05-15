"""
Train ResNet-50 on Flowers102 (Phase 1 baseline from the paper).

Result in the paper: 34.70% — showed signs of overfitting and poor generalization
on the constrained dataset. This motivated the switch to Vision Transformers.

The CNN backbone (conv1 + layer1-layer4) is frozen; only the custom head is trained:
  Linear(2048 -> 256) -> ReLU -> Dropout(0.1) -> Linear(256 -> num_classes)

Usage:
  python train_resnet50.py --config configs/resnet50_flowers.yaml
  python train_resnet50.py --config configs/resnet50_flowers.yaml --no-freeze
"""

import argparse
from pathlib import Path

import yaml

from data import get_dataloaders
from models import build_resnet50
from trainer import run_training
from utils import Logger


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-50 (CNN baseline)")
    p.add_argument("--config",    type=str, required=True,       help="Path to YAML config file")
    p.add_argument("--no-freeze", action="store_true",            help="Disable backbone freezing (leads to full fine-tuning)")
    p.add_argument("--resume",    type=str, default=None,         help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("run_name", "resnet50_flowers")

    freeze = not args.no_freeze
    print(f"[train_resnet50] Config: {args.config}  freeze_backbone={freeze}")
    print(f"  dataset={config['dataset']}  epochs={config['num_epochs']}  "
          f"batch={config['batch_size']}  lr={config['lr']}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  classes={num_classes}")

    model = build_resnet50(num_classes=num_classes, freeze=freeze)
    logger = Logger(config)

    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{config['run_name']}"))
    run_training(model, train_loader, val_loader, config, logger, checkpoint_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
