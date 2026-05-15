"""
Augmentation pipelines used during training and validation.

Training pipeline (from the paper):
  Pad(10) -> Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip(p=0.5)
  -> RandomRotation(15) -> ColorJitter -> ToTensor -> Normalize(ImageNet stats)

Validation pipeline:
  Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet stats)

ImageNet normalization statistics are used because the backbones are pretrained
on ImageNet and expect inputs in the same distribution.
"""

from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Pad(10),
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
