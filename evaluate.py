"""
Evaluate a trained model on the validation (or test) split.

Loads a checkpoint, runs inference on the dataset, and reports:
  - Top-1 accuracy
  - Top-5 accuracy
  - Per-class accuracy breakdown (optional)

Usage:
  python evaluate.py --config configs/vit_slow_flowers.yaml \\
                     --checkpoint checkpoints/vit_slow_flowers/best.pth \\
                     --model vit_slow

  python evaluate.py --config configs/vit_spiralnet_cars.yaml \\
                     --checkpoint checkpoints/vit_spiralnet_cars/best.pth \\
                     --model vit_spiralnet \\
                     --per-class
"""

import argparse

import torch
import yaml
from tqdm import tqdm

from data import get_dataloaders
from models import (
    build_resnet50,
    build_resnet152,
    build_vit_fast,
    build_vit_slow,
    build_vit_spiralnet,
)
from utils import load_checkpoint

_MODEL_REGISTRY = {
    "vit_slow":       build_vit_slow,
    "vit_fast":       build_vit_fast,
    "vit_spiralnet":  build_vit_spiralnet,
    "resnet50":       build_resnet50,
    "resnet152":      build_resnet152,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model")
    p.add_argument("--config",     type=str, required=True,
                   help="YAML config used during training")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--model",      type=str, required=True,
                   choices=list(_MODEL_REGISTRY.keys()),
                   help="Model architecture to load")
    p.add_argument("--per-class",  action="store_true",
                   help="Print per-class accuracy breakdown")
    p.add_argument("--split",      type=str, default="val",
                   help="Which split to evaluate on (only 'val' is currently wired up)")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, per_class: bool = False):
    model.eval()
    correct = top5_correct = total = 0
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for inputs, labels in tqdm(loader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

        k = min(5, outputs.size(1))
        _, top5 = outputs.topk(k, dim=1)
        top5_correct += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
        total += labels.size(0)

        if per_class:
            for pred, lbl in zip(predicted, labels):
                class_total[lbl] += 1
                if pred == lbl:
                    class_correct[lbl] += 1

    results = {
        "top1_acc": correct / total,
        "top5_acc": top5_correct / total,
        "n_samples": total,
    }
    if per_class:
        results["per_class"] = {
            c: (class_correct[c] / class_total[c]).item()
            for c in range(num_classes)
            if class_total[c] > 0
        }
    return results


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device={device}")

    _, val_loader, num_classes = get_dataloaders(config)
    print(f"[evaluate] dataset={config['dataset']}  val_samples={len(val_loader.dataset)}  classes={num_classes}")

    build_fn = _MODEL_REGISTRY[args.model]
    if args.model == "vit_spiralnet":
        model = build_fn(
            num_classes=num_classes,
            layer_width=config.get("spinalnet_layer_width", 512),
            dropout=config.get("spinalnet_dropout", 0.5),
        )
    else:
        model = build_fn(num_classes=num_classes)

    load_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)

    results = evaluate(model, val_loader, device, num_classes, per_class=args.per_class)

    print(f"\n{'='*40}")
    print(f"  Model       : {args.model}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Dataset     : {config['dataset']}")
    print(f"  Samples     : {results['n_samples']}")
    print(f"  Top-1 Acc   : {results['top1_acc']:.4f}  ({results['top1_acc']*100:.2f}%)")
    print(f"  Top-5 Acc   : {results['top5_acc']:.4f}  ({results['top5_acc']*100:.2f}%)")
    print(f"{'='*40}")

    if args.per_class and "per_class" in results:
        print("\nPer-class accuracy:")
        for cls_idx, acc in sorted(results["per_class"].items(), key=lambda x: x[1]):
            print(f"  class {cls_idx:3d}  {acc:.4f}")


if __name__ == "__main__":
    main()
