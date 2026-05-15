"""
Dataset loaders for the three experimental phases described in the paper.

Split strategy (matching the paper's reported sample counts):
  Flowers102:    train = official 'test' split (6149 imgs)
                 val   = official 'train' + 'val' splits (1020 + 1020 = 2040 imgs)
  FGVC Aircraft: train = official 'trainval' split
                 val   = official 'test' split
  Stanford Cars: train = official 'train' split (8144 imgs)
                 val   = official 'test'  split (8041 imgs)
                 Note: StanfordCars was removed from torchvision >= 0.16.
                 Falls back to ImageFolder if unavailable (see data/README.md).
  Generic:       ImageFolder with a configurable val_split random split.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    random_split,
)
from torchvision import datasets

from .transforms import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Helper: apply different transforms to an already-indexed subset
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    """Wraps a Dataset / Subset and overrides its transform at read time."""

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def _flowers102(root: str, img_size: int, batch_size: int, num_workers: int):
    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    # Paper used 6149 training images (official 'test' split) and
    # 2040 validation images (official 'train' + 'val' splits combined).
    train_ds = datasets.Flowers102(root=root, split="test", transform=train_tf, download=True)
    val_ds = ConcatDataset([
        datasets.Flowers102(root=root, split="train", transform=val_tf, download=True),
        datasets.Flowers102(root=root, split="val",   transform=val_tf, download=True),
    ])
    num_classes = 102
    return train_ds, val_ds, num_classes


def _aircraft(root: str, img_size: int, batch_size: int, num_workers: int):
    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    train_ds = datasets.FGVCAircraft(root=root, split="trainval", transform=train_tf, download=True)
    val_ds   = datasets.FGVCAircraft(root=root, split="test",     transform=val_tf,   download=True)
    num_classes = 100
    return train_ds, val_ds, num_classes


def _cars(root: str, img_size: int, batch_size: int, num_workers: int):
    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    try:
        # StanfordCars removed from torchvision >= 0.16 — see data/README.md for download.
        train_ds = datasets.StanfordCars(root=root, split="train", transform=train_tf, download=False)
        val_ds   = datasets.StanfordCars(root=root, split="test",  transform=val_tf,   download=False)
    except (AttributeError, RuntimeError, FileNotFoundError):
        print(
            "[datasets] StanfordCars not available via torchvision. "
            "Falling back to ImageFolder. Expected layout:\n"
            f"  {root}/train/<class_name>/*.jpg\n"
            f"  {root}/test/<class_name>/*.jpg\n"
            "See data/README.md for download instructions."
        )
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        val_ds   = datasets.ImageFolder(os.path.join(root, "test"),  transform=val_tf)

    num_classes = 196
    return train_ds, val_ds, num_classes


def _generic(root: str, img_size: int, val_split: float, seed: int):
    """Generic ImageFolder dataset with a random train/val split."""
    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    full_ds = datasets.ImageFolder(root=root, transform=None)
    num_classes = len(full_ds.classes)

    n = len(full_ds)
    n_val = int(n * val_split)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=gen)

    train_ds = _TransformSubset(train_subset, train_tf)
    val_ds   = _TransformSubset(val_subset,   val_tf)
    return train_ds, val_ds, num_classes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train and validation DataLoaders from a config dict.

    Required config keys:
      dataset   : one of 'flowers102', 'aircraft', 'cars', 'generic'
      data_dir  : path to dataset root
      batch_size: int
      img_size  : int (default 224)

    Returns:
      train_loader, val_loader, num_classes
    """
    name        = config["dataset"]
    root        = config["data_dir"]
    batch_size  = config["batch_size"]
    img_size    = config.get("img_size", 224)
    num_workers = config.get("num_workers", 4)
    val_split   = config.get("val_split", 0.2)
    seed        = config.get("seed", 42)

    if name == "flowers102":
        train_ds, val_ds, num_classes = _flowers102(root, img_size, batch_size, num_workers)
    elif name == "aircraft":
        train_ds, val_ds, num_classes = _aircraft(root, img_size, batch_size, num_workers)
    elif name == "cars":
        train_ds, val_ds, num_classes = _cars(root, img_size, batch_size, num_workers)
    elif name == "generic":
        train_ds, val_ds, num_classes = _generic(root, img_size, val_split, seed)
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Choose from: flowers102, aircraft, cars, generic")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes
