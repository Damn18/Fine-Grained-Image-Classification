"""
Train vit_spiralnet — ViT-Large/16 with a SpinalNet classification head.
Uses LeakyReLU activations and LabelSmoothingCrossEntropy loss.
This is the third model from the paper, which achieved 71.8% on the competition dataset.

The SpinalNet head divides the ViT's [CLS] embedding (dim=1024) into two segments,
feeding them through four spinal layers that progressively combine segment outputs.
This improves gradient flow compared to a single linear head.

Usage:
  python train_vit_spiralnet.py --config configs/vit_spiralnet_flowers.yaml
  python train_vit_spiralnet.py --config configs/vit_spiralnet_aircraft.yaml
  python train_vit_spiralnet.py --config configs/vit_spiralnet_cars.yaml
"""

import argparse
from pathlib import Path

import yaml

from data import get_dataloaders
from models import build_vit_spiralnet
from trainer import run_training
from utils import Logger


def parse_args():
    p = argparse.ArgumentParser(description="Train vit_spiralnet")
    p.add_argument("--config",  type=str, required=True, help="Path to YAML config file")
    p.add_argument("--resume",  type=str, default=None,  help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config.setdefault("run_name", f"vit_spiralnet_{config.get('dataset', 'unknown')}")

    print(f"[train_vit_spiralnet] Config: {args.config}")
    print(f"  dataset={config['dataset']}  epochs={config['num_epochs']}  "
          f"batch={config['batch_size']}  lr={config['lr']}")

    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  classes={num_classes}")

    layer_width = config.get("spinalnet_layer_width", 512)
    dropout     = config.get("spinalnet_dropout",     0.5)
    model = build_vit_spiralnet(num_classes=num_classes, layer_width=layer_width, dropout=dropout)
    logger = Logger(config)

    checkpoint_dir = Path(config.get("checkpoint_dir", f"checkpoints/{config['run_name']}"))
    run_training(model, train_loader, val_loader, config, logger, checkpoint_dir, resume_from=args.resume)


if __name__ == "__main__":
    main()
