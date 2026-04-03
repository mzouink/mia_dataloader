# mia_em_loader Architecture

## Library Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           mia_em_loader                                     │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌──────────┐  │
│  │ discover │  │ models   │  │  dataset   │  │  sampler  │  │transforms│  │
│  │          │  │          │  │            │  │           │  │          │  │
│  │discover_ │  │CropDB    │  │CellMap     │  │ClassBal-  │  │EMTrans-  │  │
│  │crops()   │  │CropInfo  │  │Dataset3D   │  │ancedSamp- │  │forms     │  │
│  │          │  │NormParams│  │ConcatMia   │  │ler        │  │          │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └─────┬─────┘  └────┬─────┘  │
│       │              │              │               │              │        │
└───────┼──────────────┼──────────────┼───────────────┼──────────────┼────────┘
        │              │              │               │              │
        ▼              ▼              ▼               ▼              ▼
   Zarr / OME-NGFF   JSON       torch.Dataset    torch.Sampler    MONAI
```

## Data Flow: From Raw Data to Training Batches

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      PHASE 1: DISCOVERY (once)                      │
  │                                                                     │
  │   /nrs/cellmap/data/                                                │
  │       ├── jrc_cos4/dataset.zarr/recon-1/                            │
  │       │     ├── em/fibsem-uint8/s0,s1,s2...   ─┐                   │
  │       │     └── labels/mito/crop1/s0,s1...     ─┤                   │
  │       └── jrc_hela2/dataset.zarr/recon-1/...   ─┘                   │
  │                                                 │                   │
  │              discover_crops(data_root,           │                   │
  │                norms_csv, target_classes,         │                   │
  │                target_resolution)                 │                   │
  │                         │                                           │
  │                         ▼                                           │
  │              ┌─────────────────────┐                                │
  │              │    CropDatabase     │                                │
  │              │  ┌───────────────┐  │                                │
  │              │  │  CropInfo[]   │  │                                │
  │              │  │  ├─raw path   │  │                                │
  │              │  │  ├─resolution │  │                                │
  │              │  │  ├─NormParams │  │                                │
  │              │  │  └─ClassInfo{}│  │                                │
  │              │  └───────────────┘  │                                │
  │              └────────┬────────────┘                                │
  │                       │  .to_json("crops.json")                    │
  │                       ▼                                             │
  │                  crops.json  (reusable, serialized)                 │
  └─────────────────────────────────────────────────────────────────────┘

                          │
                          ▼

  ┌─────────────────────────────────────────────────────────────────────┐
  │              PHASE 2: DATASET CONSTRUCTION                          │
  │                                                                     │
  │   CropDatabase.from_json("crops.json")                             │
  │           │                                                         │
  │           ▼                                                         │
  │   ┌───────────────────────────────────────────────────────┐        │
  │   │              CellMapDataset3D                          │        │
  │   │                                                        │        │
  │   │  Inputs:                                               │        │
  │   │    crop_db        ──  discovered crop metadata         │        │
  │   │    target_classes ──  ["mito", "er", "nuc"]            │        │
  │   │    input_size     ──  (128, 128, 128) voxels           │        │
  │   │    output_size    ──  (64, 64, 64) voxels              │        │
  │   │    target_res     ──  8.0 nm                           │        │
  │   │    transforms     ──  EMTransforms (optional)          │        │
  │   │                                                        │        │
  │   │  Implements: torch.utils.data.Dataset                  │        │
  │   └───────────────────────────────────────────────────────┘        │
  │                                                                     │
  │   ┌───────────────────────────────────────────────────────┐        │
  │   │  ConcatMiaDataset  (optional, wraps N datasets)        │        │
  │   │    datasets=[ds_a, ds_b]                               │        │
  │   │    weights=[0.7, 0.3]                                  │        │
  │   └───────────────────────────────────────────────────────┘        │
  └─────────────────────────────────────────────────────────────────────┘

                          │
                          ▼

  ┌─────────────────────────────────────────────────────────────────────┐
  │              PHASE 3: SAMPLE LOADING (__getitem__)                   │
  │                                                                     │
  │   For each sample index:                                            │
  │                                                                     │
  │   ┌─────────────┐     ┌─────────────────────────────────┐          │
  │   │ Pick random  │     │  Read from Zarr                 │          │
  │   │ crop + loc   │────▶│                                 │          │
  │   └─────────────┘     │  ┌─────────┐   ┌────────────┐  │          │
  │                        │  │ Raw EM  │   │  Labels    │  │          │
  │                        │  │ patch   │   │  per-class │  │          │
  │                        │  └────┬────┘   └─────┬──────┘  │          │
  │                        └───────┼──────────────┼─────────┘          │
  │                                │              │                     │
  │                                ▼              ▼                     │
  │                        ┌──────────────────────────────┐            │
  │                        │  Normalize + Resample         │            │
  │                        │  raw → [0,1], zoom to size    │            │
  │                        └──────────────┬───────────────┘            │
  │                                       │                             │
  │                                       ▼                             │
  │                        ┌──────────────────────────────┐            │
  │                        │  Apply Transforms (optional)  │            │
  │                        │  EMTransforms (MONAI)         │            │
  │                        └──────────────┬───────────────┘            │
  │                                       │                             │
  │                                       ▼                             │
  │                        ┌──────────────────────────────┐            │
  │                        │  Output Tuple:                │            │
  │                        │                               │            │
  │                        │  raw:       [1, D, H, W]      │            │
  │                        │  labels:    [C, d, h, w]      │            │
  │                        │  ann_mask:  [C]        bool   │            │
  │                        │  spat_mask: [1, d, h, w]      │            │
  │                        │  metadata:  dict              │            │
  │                        └──────────────────────────────┘            │
  └─────────────────────────────────────────────────────────────────────┘

                          │
                          ▼

  ┌─────────────────────────────────────────────────────────────────────┐
  │              PHASE 4: BATCHING (PyTorch DataLoader)                  │
  │                                                                     │
  │   ┌────────────────────┐    ┌───────────────────────────────┐      │
  │   │ ClassBalancedSampler│    │  torch.utils.data.DataLoader  │      │
  │   │                    │───▶│                               │      │
  │   │ Picks least-seen   │    │  batch_size=4                │      │
  │   │ class, samples a   │    │  num_workers=8               │      │
  │   │ crop that has it   │    │  sampler=ClassBalancedSampler │      │
  │   └────────────────────┘    └──────────────┬────────────────┘      │
  │                                             │                       │
  │                                             ▼                       │
  │                              ┌──────────────────────────┐          │
  │                              │  Training Batch:          │          │
  │                              │  raw:    [B, 1, D, H, W]  │          │
  │                              │  labels: [B, C, d, h, w]  │          │
  │                              │  masks:  [B, C]           │          │
  │                              │  spatial:[B, 1, d, h, w]  │          │
  │                              └──────────────────────────┘          │
  └─────────────────────────────────────────────────────────────────────┘
```

## PyTorch Integration Map

```
  torch.utils.data                         mia_em_loader
  ───────────────                          ──────────────

  Dataset (ABC)  ◄──────── extends ─────── MiaDataset3D (abstract)
       ▲                                        ▲
       │                                        │
       │                              ┌─────────┴──────────┐
       │                              │                    │
       │                       CellMapDataset3D    ConcatMiaDataset
       │                       (zarr loading)      (multi-dataset)
       │
  Sampler (ABC)  ◄──────── extends ─────── ClassBalancedSampler
       │
  DataLoader  ─── uses ──▶  Dataset + Sampler


  MONAI                                    mia_em_loader
  ─────                                    ──────────────

  Compose          ◄──── used by ───────── EMTransforms
  Rand3DElasticd   ◄──── spatial ──────────     │
  RandRotate90d    ◄──── spatial ──────────     │
  RandFlipd        ◄──── spatial ──────────     │
  RandShiftIntensityd ◄── intensity ───────     │
  RandGaussianNoised  ◄── intensity ───────     │
  RandGaussianSmoothd ◄── intensity ───────     │
  RandAdjustContrastd  ◄── intensity ──────     │
```

## How to Add Custom Transforms

Transforms are injected into any dataset at construction time via the `transforms`
parameter. The transform callable receives `(raw, labels)` and returns `(raw, labels)`.

### Option 1: Use Built-in EMTransforms with Custom Parameters

```python
from mia_em_loader import get_train_transforms

transforms = get_train_transforms(
    spatial_prob=0.8,       # probability of spatial augmentation
    intensity_prob=0.5,     # probability of intensity augmentation
    elastic_sigma=(5, 20),  # elastic deformation range
    noise_std=(0.01, 0.1),  # Gaussian noise range
    blur_sigma=(0.5, 2.0),  # Gaussian blur range
    gamma_range=(0.5, 1.5), # contrast adjustment range
)

ds = CellMapDataset3D(..., transforms=transforms)
```

### Option 2: Use Config Dict (YAML-friendly)

```python
from mia_em_loader import get_train_transforms_from_config

config = {
    "spatial_prob": 0.8,
    "intensity_prob": 0.5,
    "elastic_sigma": [5.0, 20.0],
    "noise_std": [0.01, 0.1],
}

transforms = get_train_transforms_from_config(config)
```

### Option 3: Write a Fully Custom Transform

A transform is any callable with this signature:

```python
def my_transforms(raw: torch.Tensor, labels: torch.Tensor)
    -> tuple[torch.Tensor, torch.Tensor]:
```

**Example: Adding a custom transform that wraps EMTransforms**

```python
import torch
from mia_em_loader import EMTransforms

class MyCustomTransforms:
    def __init__(self):
        # Use the built-in MONAI pipeline as a base
        self.base = EMTransforms(spatial_prob=0.5, intensity_prob=0.3)

    def __call__(self, raw: torch.Tensor, labels: torch.Tensor):
        # 1. Apply base augmentations
        raw, labels = self.base(raw, labels)

        # 2. Add your custom augmentations
        # Example: random cutout / dropout
        if torch.rand(1) < 0.2:
            d, h, w = raw.shape[1], raw.shape[2], raw.shape[3]
            cd, ch, cw = d // 4, h // 4, w // 4
            zs = torch.randint(0, d - cd, (1,))
            ys = torch.randint(0, h - ch, (1,))
            xs = torch.randint(0, w - cw, (1,))
            raw[:, zs:zs+cd, ys:ys+ch, xs:xs+cw] = 0.0

        # Example: mixup with Gaussian noise in a region
        if torch.rand(1) < 0.1:
            noise = torch.randn_like(raw) * 0.05
            raw = raw + noise
            raw = raw.clamp(0, 1)

        return raw, labels

ds = CellMapDataset3D(..., transforms=MyCustomTransforms())
```

### Option 4: Compose Multiple Transform Stages

```python
class ComposeTransforms:
    """Chain multiple (raw, labels) -> (raw, labels) transforms."""
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, raw, labels):
        for t in self.transforms:
            raw, labels = t(raw, labels)
        return raw, labels

# Mix built-in with custom
from mia_em_loader import EMTransforms

pipeline = ComposeTransforms(
    EMTransforms(spatial_prob=0.5),    # MONAI spatial + intensity
    MyCustomIntensityAug(),             # your custom step
    RandomCropPaste(),                  # another custom step
)

ds = CellMapDataset3D(..., transforms=pipeline)
```

### Transform Contract

Your transform MUST follow these rules:

| Property | Requirement |
|---|---|
| Input raw shape | `[1, D, H, W]` float32, values in [0, 1] |
| Input labels shape | `[C, D, H, W]` float32, binary (0 or 1) |
| Output raw shape | Same as input (do not change spatial dims) |
| Output labels shape | Same as input |
| Raw value range | Keep in [0, 1] (clamp after augmentation) |
| Labels values | Keep binary — re-threshold with `> 0.5` after interpolation |
| Spatial consistency | If you apply spatial transforms, apply them to BOTH raw and labels |

## Complete Training Example

```python
from mia_em_loader import (
    CropDatabase, CellMapDataset3D, ConcatMiaDataset,
    ClassBalancedSampler, get_train_transforms, get_val_transforms,
)
from torch.utils.data import DataLoader

# Load discovered crops
db = CropDatabase.from_json("crops.json")

# Datasets with/without augmentation
train_ds = CellMapDataset3D(
    crop_db=db,
    target_classes=["mito", "er", "nucleus"],
    input_size=(128, 128, 128),
    output_size=(64, 64, 64),
    target_resolution=8.0,
    samples_per_epoch=5000,
    transforms=get_train_transforms(),  # MONAI augmentations
)

val_ds = CellMapDataset3D(
    crop_db=db,
    target_classes=["mito", "er", "nucleus"],
    input_size=(128, 128, 128),
    output_size=(64, 64, 64),
    target_resolution=8.0,
    samples_per_epoch=500,
    transforms=get_val_transforms(),  # None (no augmentation)
)

# Balanced sampling + DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    sampler=ClassBalancedSampler(train_ds, samples_per_epoch=5000),
    num_workers=8,
)

# Training loop
for raw, labels, ann_mask, spatial_mask, meta in train_loader:
    # raw:         [B, 1, 128, 128, 128]
    # labels:      [B, 3,  64,  64,  64]
    # ann_mask:    [B, 3]                  (which classes annotated)
    # spatial_mask:[B, 1,  64,  64,  64]   (valid region)
    pred = model(raw)
    loss = criterion(pred, labels, ann_mask, spatial_mask)
    loss.backward()
    optimizer.step()
```
