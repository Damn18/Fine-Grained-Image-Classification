"""
ResNet model factories with the freezing strategy and custom classifier head
described in the paper.

Freezing strategy: the initial conv1 layer and residual blocks layer1-layer4
are frozen to prevent catastrophic forgetting when fine-tuning on small datasets.
Only the custom fully-connected head is trained.

Custom head: Linear(2048 -> 256) -> ReLU -> Dropout(0.1) -> Linear(256 -> num_classes)
"""

import torch.nn as nn
import torchvision.models as tv_models


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze conv1 and all residual blocks (layer1-layer4), leaving only the head trainable."""
    frozen_names = {"layer1", "layer2", "layer3", "layer4"}
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d) or name in frozen_names:
            for param in child.parameters():
                param.requires_grad = False


def _replace_head(model: nn.Module, num_classes: int) -> None:
    in_features = model.fc.in_features  # 2048 for both ResNet50 and ResNet152
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes),
    )


def build_resnet50(num_classes: int, freeze: bool = True) -> nn.Module:
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze:
        _freeze_backbone(model)
    _replace_head(model, num_classes)
    return model


def build_resnet152(num_classes: int, freeze: bool = True) -> nn.Module:
    model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V1)
    if freeze:
        _freeze_backbone(model)
    _replace_head(model, num_classes)
    return model
