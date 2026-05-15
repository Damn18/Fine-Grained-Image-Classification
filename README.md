# Fine-Grained Image Classification

Implementation of the architectures and training pipeline described in our computer vision report on Fine-Grained Image Classification (FGIC). The project compares CNN baselines (ResNet-50, ResNet-152) against Vision Transformer (ViT) variants, including a novel ViT + SpinalNet hybrid.

**Best result: 72% top-1 accuracy** on the competition mammal dataset (`vit_slow`).

---

## Architecture Overview

| Script | Model | Head | Batch | LR | Competition result |
|---|---|---|---|---|---|
| `train_vit_slow.py` | ViT-Large/16 | Linear | 12 | 0.001 | **72.0%** |
| `train_vit_fast.py` | ViT-Large/16 | Linear | 32 | 0.003 | — (not submitted) |
| `train_vit_spiralnet.py` | ViT-Large/16 | **SpinalNet** | 12 | 0.001 | **71.8%** |
| `train_resnet50.py` | ResNet-50 | FC(2048→256→C) | 32 | 0.01 | 34.7% |
| `train_resnet152.py` | ResNet-152 | FC(2048→256→C) | 32 | 0.01 | 56.3% |

All ViT models use `vit_large_patch16_224` pretrained on ImageNet-21k (via [timm](https://github.com/huggingface/pytorch-image-models)).  
All CNN models freeze the backbone and train only the classifier head.

---

## Dataset Setup

See [data/README.md](data/README.md) for download instructions.

- **Oxford Flowers 102** — downloaded automatically
- **FGVC Aircraft** — downloaded automatically
- **Stanford Cars** — requires manual Kaggle download

---
