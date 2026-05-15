# Dataset Setup

This project uses three fine-grained image classification datasets. Two are downloaded automatically; Stanford Cars requires a manual download.

---

## Oxford Flowers 102

**Downloaded automatically** by torchvision on first run.

```bash
python train_vit_slow.py --config configs/vit_slow_flowers.yaml
# torchvision will download to ./data/flowers102/ automatically
```

Expected size: ~350 MB — 8,189 images across 102 flower species.

---

## FGVC Aircraft

**Downloaded automatically** by torchvision on first run.

```bash
python train_vit_slow.py --config configs/vit_slow_aircraft.yaml
# torchvision will download to ./data/aircraft/ automatically
```

Expected size: ~500 MB — 10,200 images across 100 aircraft model categories.

---

## Stanford Cars

torchvision removed StanfordCars in v0.16. Download it manually from Kaggle:

1. Install the Kaggle CLI: `pip install kaggle`
2. Download: `kaggle datasets download -d jessicali9530/stanford-cars-dataset`
3. Extract and arrange as:

```
data/cars/
├── train/
│   ├── AM General Hummer SUV 2000/
│   │   ├── 00001.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── AM General Hummer SUV 2000/
    └── ...
```

The dataset loader will fall back to `ImageFolder` automatically if torchvision's
`StanfordCars` is unavailable, provided the above folder layout exists.

Expected size: ~2 GB — 16,185 images across 196 car make/model/year categories.

---

## Competition Dataset (Mammals — generic)

For the original competition dataset, use the generic `ImageFolder` loader:

```yaml
# In your config:
dataset: generic
data_dir: ./data/mammals   # path to your dataset root
val_split: 0.2             # 80/20 random split
```

Expected layout:
```
data/mammals/
├── class_name_1/
│   ├── img_001.jpg
│   └── ...
└── class_name_2/
    └── ...
```
