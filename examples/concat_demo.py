#%%
"""Demo: combining multiple datasets with ConcatEMDataset.

Shows how to:
1. Load a CropDatabase and create two CellMapDataset3D with different splits
2. Combine them with ConcatEMDataset (weighted sampling)
3. Use ClassBalancedSampler on the combined dataset
4. Also validates a plain third-party dataset via validate_em_dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mia_em_loader import (
    CellMapDataset3D,
    ConcatEMDataset,
    ClassBalancedSampler,
    CropDatabase,
    discover_crops,
    validate_em_dataset,
)

CLASSES = ["ecs", "mito", "er", "nuc"]
CROPS_JSON = "crops.json"
DATA_ROOT = "/nrs/cellmap/data"
NORMS_CSV = "norms.csv"

# ---------------------------------------------------------------------------
# Step 1: Load or discover
# ---------------------------------------------------------------------------

#%%
if os.path.exists(CROPS_JSON):
    db = CropDatabase.from_json(CROPS_JSON)
else:
    db = discover_crops(DATA_ROOT, NORMS_CSV, CLASSES)
    db.to_json(CROPS_JSON)

# ---------------------------------------------------------------------------
# Step 2: Create two datasets (e.g. different subsets of the same data)
# ---------------------------------------------------------------------------

#%%
# Split crops into two halves to simulate two data sources
all_crops = db.crops
mid = len(all_crops) // 2

db_a = CropDatabase(version=db.version, crops=all_crops[:mid], discovery_params=db.discovery_params)
db_b = CropDatabase(version=db.version, crops=all_crops[mid:], discovery_params=db.discovery_params)

ds_a = CellMapDataset3D(crop_db=db_a, target_classes=CLASSES, input_size=(64, 64, 64), samples_per_epoch=500)
ds_b = CellMapDataset3D(crop_db=db_b, target_classes=CLASSES, input_size=(64, 64, 64), samples_per_epoch=500)

print(f"Dataset A: {len(ds_a.crops)} crops")
print(f"Dataset B: {len(ds_b.crops)} crops")

# ---------------------------------------------------------------------------
# Step 3: Combine with ConcatEMDataset
# ---------------------------------------------------------------------------

#%%
combined = ConcatEMDataset(
    [ds_a, ds_b],
    weights=[0.6, 0.4],       # sample 60% from A, 40% from B
    samples_per_epoch=1000,
)
print(combined.summary())

# ---------------------------------------------------------------------------
# Step 4: Use ClassBalancedSampler on the combined dataset
# ---------------------------------------------------------------------------

#%%
sampler = ClassBalancedSampler(combined, samples_per_epoch=1000, seed=42)
loader = DataLoader(combined, batch_size=4, sampler=sampler, num_workers=2)

batch = next(iter(loader))
raw_b, labels_b, ann_b, spatial_b, meta_b = batch
print(f"\nBatch from combined dataset:")
print(f"  raw:    {raw_b.shape}")
print(f"  labels: {labels_b.shape}")
print(f"  crops:  {[f'{d}/{c}' for d, c in zip(meta_b['dataset'], meta_b['crop'])]}")

# ---------------------------------------------------------------------------
# Step 5: Validate a third-party / custom dataset
# ---------------------------------------------------------------------------

#%%
class DummyEMDataset(Dataset):
    """Example: a plain Dataset that follows the MiaDataset3D contract
    without inheriting from it. validate_em_dataset checks it at runtime."""

    def __init__(self, target_classes, input_size, n_samples=100):
        self.target_classes = list(target_classes)
        self.n_classes = len(self.target_classes)
        self.input_size = np.array(input_size)
        self.output_size = np.array(input_size)  # same as input
        self.crops = list(range(n_samples))  # dummy crop list

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        iD, iH, iW = self.input_size
        raw = torch.rand(1, iD, iH, iW, dtype=torch.float32)
        labels = torch.zeros(self.n_classes, iD, iH, iW, dtype=torch.float32)
        annotated_mask = torch.ones(self.n_classes, dtype=torch.bool)
        spatial_mask = torch.ones(1, iD, iH, iW, dtype=torch.float32)
        metadata = {"dataset": "dummy", "crop": f"sample_{idx}"}
        return raw, labels, annotated_mask, spatial_mask, metadata

    def get_crop_class_matrix(self):
        return np.ones((len(self.crops), self.n_classes), dtype=bool)


dummy_ds = DummyEMDataset(target_classes=CLASSES, input_size=(64, 64, 64))

# This validates the dummy dataset at runtime — no inheritance needed
validate_em_dataset(dummy_ds, target_classes=CLASSES, input_size=(64, 64, 64))
print("\nDummyEMDataset passed validation!")

# Now combine it with the real CellMap data
mixed = ConcatEMDataset(
    [ds_a, dummy_ds],
    weights=[0.8, 0.2],
    samples_per_epoch=500,
)
print(mixed.summary())

# %%
