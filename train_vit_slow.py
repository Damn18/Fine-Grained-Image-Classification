"""
Train vit_slow — ViT-Large/16 with a linear head.
Batch size: 12  |  LR: 0.001  (the slower, more stable variant from the paper)

Usage:
  python train_vit_slow.py --config configs/vit_slow_flowers.yaml
  python train_vit_slow.py --config configs/vit_slow_aircraft.yaml
  python train_vit_slow.py --config configs/vit_slow_cars.yaml
  python train_vit_slow.py --config configs/vit_slow_flowers.yaml --resume checkpoints/vit_slow_flowers/latest.pth
"""

import argparse
from pathlib import Path

import yaml

from data import get_dataloaders
from models import build_vit_slow
from trainer import run_training
from utils import Logger


def parse_args():
    p = argparse.ArgumentParser(description="Train vit_slow")
    p.add_argument("--config",  type=str, required=True, help="Path to YAML config file")
    p.add_argument("--resume",  type=str, default=None,  help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("run_name", f"vit_slow_{config.get('dataset', 'unknown')}")

    print(f"[train_vit_slow] Config: {args.config}")
    print(f"  dataset={config['dataset']}  epochs={config['num_epochs']}  "
          f"batch={config['batch_size']}  lr={config['lr']}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  classes={num_classes}")

    model = build_vit_slow(num_classes=num_classes)
    logger = Logger(config)

    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{config['run_name']}"))
    run_training(model, train_loader, val_loader, config, logger, checkpoint_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
