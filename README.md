# mia_em_loader

Generic 3D multi-label data loading for CellMap-style Zarr datasets.

![Sample output](imgs/img.png)

## Features

- **Zarr / OME-NGFF** native: reads CellMap-style multi-scale Zarr volumes directly
- **Multi-label**: load arbitrary combinations of organelle classes (mito, ER, nucleus, etc.)
- **Class-balanced sampling**: ensures rare classes are seen equally during training
- **MONAI augmentations**: built-in spatial and intensity transforms, or bring your own
- **Multi-dataset composition**: combine datasets from different sources with weighted sampling
- **Two-phase workflow**: discover crops once, reuse the metadata forever


## Quick Start

### 1. Discover crops (one-time)

```python
from mia_em_loader import discover_crops

db = discover_crops(
    data_root="/nrs/cellmap/data",
    norms_csv="norms.csv",
    target_classes=["ecs", "cell", "nuc", "mito", "er", "golgi"],
    target_resolution=8.0,
)
db.to_json("crops.json")
```

### 2. Create a dataset and train

```python
from mia_em_loader import (
    CropDatabase, CellMapDataset3D,
    ClassBalancedSampler, get_train_transforms,
)
from torch.utils.data import DataLoader

db = CropDatabase.from_json("crops.json")

train_ds = CellMapDataset3D(
    crop_db=db,
    target_classes=["mito", "er", "nucleus"],
    input_size=(128, 128, 128),
    output_size=(64, 64, 64),
    target_resolution=8.0,
    samples_per_epoch=5000,
    transforms=get_train_transforms(),
)

loader = DataLoader(
    train_ds,
    batch_size=4,
    sampler=ClassBalancedSampler(train_ds, samples_per_epoch=5000),
    num_workers=8,
)

for raw, labels, ann_mask, spatial_mask, meta in loader:
    # raw:          [B, 1, 128, 128, 128]
    # labels:       [B, 3,  64,  64,  64]
    # ann_mask:     [B, 3]                  (which classes are annotated)
    # spatial_mask: [B, 1,  64,  64,  64]   (valid region)
    pred = model(raw)
    loss = criterion(pred, labels, ann_mask, spatial_mask)
    loss.backward()
    optimizer.step()
```

### 3. Combine multiple datasets

```python
from mia_em_loader import ConcatMiaDataset

combined = ConcatMiaDataset(
    [ds_a, ds_b],
    weights=[0.6, 0.4],
    samples_per_epoch=1000,
)
```

## Custom Transforms

Any callable with signature `(raw: Tensor, labels: Tensor) -> (raw, labels)` works:

```python
from mia_em_loader import EMTransforms

class MyTransforms:
    def __init__(self):
        self.base = EMTransforms(spatial_prob=0.5, intensity_prob=0.3)

    def __call__(self, raw, labels):
        raw, labels = self.base(raw, labels)
        # add your custom augmentations here
        return raw, labels

ds = CellMapDataset3D(..., transforms=MyTransforms())
```

See [docs/architecture.md](docs/architecture.md#how-to-add-custom-transforms) for more options and the full transform contract.

