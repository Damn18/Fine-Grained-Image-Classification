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

### SpinalNet Head

The SpinalNet head (Kabir et al., 2020) divides the 1024-dimensional ViT embedding into two 512-d segments, feeding them through four spinal layers in an alternating pattern. Each layer receives the current segment concatenated with the previous layer's output. This improves gradient flow and reduces the vanishing gradient problem.

```
x[:, :512]  ──► layer1 ──► out1
x[:, 512:]  ──► layer2(cat[x2, out1]) ──► out2
x[:, :512]  ──► layer3(cat[x1, out2]) ──► out3
x[:, 512:]  ──► layer4(cat[x2, out3]) ──► out4
                cat(out1, out2, out3, out4) → Linear → logits
```

---

## Installation

```bash
git clone https://github.com/<your-username>/Fine-Grained-Image-Classification.git
cd Fine-Grained-Image-Classification
pip install -r requirements.txt
```

**WandB setup** (first time):
```bash
wandb login --relogin   # authenticate via GitHub at the prompt
```

---

## Dataset Setup

See [data/README.md](data/README.md) for download instructions.

- **Oxford Flowers 102** — downloaded automatically
- **FGVC Aircraft** — downloaded automatically
- **Stanford Cars** — requires manual Kaggle download

---

## Training

```bash
# ViT variants (ViT-Large/16 pretrained)
python train_vit_slow.py      --config configs/vit_slow_flowers.yaml
python train_vit_fast.py      --config configs/vit_fast_aircraft.yaml
python train_vit_spiralnet.py --config configs/vit_spiralnet_cars.yaml

# CNN baselines (Flowers102 only, as in the paper)
python train_resnet50.py      --config configs/resnet50_flowers.yaml
python train_resnet152.py     --config configs/resnet152_flowers.yaml

# Resume from checkpoint
python train_vit_slow.py --config configs/vit_slow_flowers.yaml \
                         --resume checkpoints/vit_slow_flowers/latest.pth
```

---

## Evaluation

```bash
python evaluate.py --config configs/vit_slow_flowers.yaml \
                   --checkpoint checkpoints/vit_slow_flowers/best.pth \
                   --model vit_slow

# With per-class breakdown
python evaluate.py --config configs/vit_spiralnet_cars.yaml \
                   --checkpoint checkpoints/vit_spiralnet_cars/best.pth \
                   --model vit_spiralnet \
                   --per-class
```

---

## Training Pipeline

All scripts share the same training engine (`trainer.py`):

- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=0.01`) for ViT; SGD for CNNs
- **Scheduler:** Warmup + Cosine Annealing — 500 warmup steps, then cosine decay over total training steps
- **Loss:** LabelSmoothingCrossEntropy (`smoothing=0.1`) for ViT; CrossEntropy for CNNs
- **Regularization:** data augmentation (pad, crop, flip, rotation, color jitter) + early stopping + dropout in SpinalNet
- **Logging:** Weights & Biases + CSV fallback
- **Checkpointing:** `best.pth` (best val acc) + `latest.pth` (most recent epoch)

---

## Project Structure

```
.
├── train_vit_slow.py         # Entry point: vit_slow
├── train_vit_fast.py         # Entry point: vit_fast
├── train_vit_spiralnet.py    # Entry point: vit + SpinalNet head
├── train_resnet50.py         # Entry point: ResNet-50 baseline
├── train_resnet152.py        # Entry point: ResNet-152 baseline
├── evaluate.py               # Evaluate any model from a checkpoint
├── trainer.py                # Shared training loop (all scripts call this)
│
├── models/
│   ├── spinalnet.py          # SpinalNet classification head
│   ├── vit.py                # ViT model factories
│   └── resnet.py             # ResNet model factories + freezing strategy
│
├── data/
│   ├── datasets.py           # DataLoader builders (Flowers, Aircraft, Cars, generic)
│   ├── transforms.py         # Train/val augmentation pipelines
│   └── README.md             # Dataset download instructions
│
├── utils/
│   ├── logger.py             # WandB + CSV logger
│   ├── checkpoint.py         # Save / load checkpoints
│   └── early_stopping.py     # Early stopping monitor
│
└── configs/                  # One YAML per model × dataset combination
    ├── vit_slow_flowers.yaml
    ├── vit_slow_aircraft.yaml
    ├── vit_slow_cars.yaml
    ├── vit_fast_flowers.yaml
    ├── vit_fast_aircraft.yaml
    ├── vit_fast_cars.yaml
    ├── vit_spiralnet_flowers.yaml
    ├── vit_spiralnet_aircraft.yaml
    ├── vit_spiralnet_cars.yaml
    ├── resnet50_flowers.yaml
    └── resnet152_flowers.yaml
```

---

## References

- Dosovitskiy et al. — *An Image is Worth 16x16 Words* (ViT), ICLR 2021
- Kabir et al. — *SpinalNet: Deep Neural Network with Gradual Input*, arXiv 2020
- Loshchilov & Hutter — *Decoupled Weight Decay Regularization* (AdamW), ICLR 2019
- Wightman — *PyTorch Image Models (timm)*, 2019
