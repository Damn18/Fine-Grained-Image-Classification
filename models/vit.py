"""
Vision Transformer (ViT) model factories.

All three variants use vit_large_patch16_224 pretrained on ImageNet-21k via timm.
The classification head is replaced to match the target number of classes.

  vit_slow        — linear head, batch=12, lr=0.001  (submitted, 72%)
  vit_fast        — linear head, batch=32, lr=0.003  (trained, not submitted)
  vit_spiralnet   — SpinalNet head + LeakyReLU, batch=12, lr=0.001 (submitted, 71.8%)
"""

import timm
import torch.nn as nn
from .spinalnet import SpinalNet

_BACKBONE = "vit_large_patch16_224"


def _load_backbone(num_classes: int) -> nn.Module:
    model = timm.create_model(_BACKBONE, pretrained=True, num_classes=num_classes)
    return model


def build_vit_slow(num_classes: int) -> nn.Module:
    """Standard ViT with a single linear classification head. Slow (batch=12) variant."""
    return _load_backbone(num_classes)


def build_vit_fast(num_classes: int) -> nn.Module:
    """Standard ViT with a single linear classification head. Fast (batch=32) variant."""
    return _load_backbone(num_classes)


def build_vit_spiralnet(num_classes: int, layer_width: int = 512, dropout: float = 0.5) -> nn.Module:
    """ViT backbone with a SpinalNet classification head and LeakyReLU activations."""
    model = timm.create_model(_BACKBONE, pretrained=True, num_classes=0)  # num_classes=0 removes head
    in_features = model.num_features  # 1024 for vit_large
    model.head = SpinalNet(
        in_features=in_features,
        num_classes=num_classes,
        layer_width=layer_width,
        dropout=dropout,
    )
    return model
