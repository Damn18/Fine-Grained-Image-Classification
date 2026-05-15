"""
Train ResNet-152 on Flowers102 (Phase 1 baseline from the paper).

Result in the paper: 56.32% — better than ResNet-50 but still far behind ViT (93%+).
The deeper architecture helped but the inductive bias of local convolutions was
insufficient for the subtle inter-class differences in fine-grained tasks.

Usage:
  python train_resnet152.py --config configs/resnet152_flowers.yaml
  python train_resnet152.py --config configs/resnet152_flowers.yaml --no-freeze
"""

import argparse
from pathlib import Path

import yaml

from data import get_dataloaders
from models import build_resnet152
from trainer import run_training
from utils import Logger


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet-152 (CNN baseline)")
    p.add_argument("--config",    type=str, required=True,  help="Path to YAML config file")
    p.add_argument("--no-freeze", action="store_true",       help="Disable backbone freezing")
    p.add_argument("--resume",    type=str, default=None,    help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("run_name", "resnet152_flowers")

    freeze = not args.no_freeze
    print(f"[train_resnet152] Config: {args.config}  freeze_backbone={freeze}")
    print(f"  dataset={config['dataset']}  epochs={config['num_epochs']}  "
          f"batch={config['batch_size']}  lr={config['lr']}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  classes={num_classes}")

    model = build_resnet152(num_classes=num_classes, freeze=freeze)
    logger = Logger(config)

    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{config['run_name']}"))
    run_training(model, train_loader, val_loader, config, logger, checkpoint_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
