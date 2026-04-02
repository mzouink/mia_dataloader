#%%
import os
from torch.utils.data import DataLoader

from mia_em_loader import CellMapDataset3D, CropDatabase, discover_crops, ClassBalancedSampler

CLASSES = ["ecs", "cell", "nuc", "mito", "er", "golgi"]

DATA_ROOT = "/nrs/cellmap/data"
NORMS_CSV = "norms.csv"
CROPS_JSON = "crops.json"

# ---------------------------------------------------------------------------
# Step 1: Discover (run once, or load from JSON)
# ---------------------------------------------------------------------------

if os.path.exists(CROPS_JSON):
    print(f"Loading crop database from {CROPS_JSON}...")
    db = CropDatabase.from_json(CROPS_JSON)
else:
    print("Discovering crops (this may take a few minutes)...")
    db = discover_crops(
        data_root=DATA_ROOT,
        norms_csv=NORMS_CSV,
        target_classes=CLASSES,
        target_resolution=8.0,
    )
    db.to_json(CROPS_JSON)

print(db.summary(CLASSES))

# ---------------------------------------------------------------------------
# Step 2: Create dataset from database
# ---------------------------------------------------------------------------

#%%
print("\nCreating dataset...")
ds = CellMapDataset3D(
    crop_db=db,
    target_classes=CLASSES,
    target_resolution=8.0,
    input_size=(128, 128, 128),
    samples_per_epoch=1000,
)
print(ds.summary())

# --- Sample one patch ---
raw, labels, annotated_mask, spatial_mask, meta = ds[0]
print(f"\nSample from: {meta['dataset']}/{meta['crop']}")
print(f"  raw:            {raw.shape}  (min={raw.min():.2f}, max={raw.max():.2f})")
print(f"  labels:         {labels.shape}")
print(f"  annotated_mask: {annotated_mask.sum()}/{len(annotated_mask)} classes")
print(f"  spatial_mask:   {spatial_mask.shape}  ({spatial_mask.mean():.1%} inside crop)")
print(f"  raw path:       {meta['raw_zarr_path']}")
print(f"  world origin:   {meta['sample_origin_world']}")

# --- DataLoader ---
#%%
loader = DataLoader(ds, batch_size=4, num_workers=4)

batch = next(iter(loader))
raw_b, labels_b, ann_b, spatial_b, meta_b = batch
print(f"\nBatch shapes:")
print(f"  raw:    {raw_b.shape}")
print(f"  labels: {labels_b.shape}")
print(f"  crops:  {[f'{d}/{c}' for d, c in zip(meta_b['dataset'], meta_b['crop'])]}")

# ---------------------------------------------------------------------------
# Step 3: Visualise a sample
# ---------------------------------------------------------------------------

#%%
import matplotlib.pyplot as plt

mid_z = raw.shape[1] // 2  # middle slice along depth

fig, axes = plt.subplots(1, 2 + len(CLASSES), figsize=(3 * (2 + len(CLASSES)), 3))

# Raw image
axes[0].imshow(raw[0, mid_z], cmap="gray")
axes[0].set_title("Raw")

# Spatial mask
axes[1].imshow(spatial_mask[0, mid_z], cmap="gray")
axes[1].set_title("Spatial mask")

# Per-class labels
for i, cls in enumerate(CLASSES):
    axes[2 + i].imshow(labels[i, mid_z], cmap="gray")
    ann = "✓" if annotated_mask[i] else "✗"
    axes[2 + i].set_title(f"{cls} ({ann})")

for ax in axes:
    ax.axis("off")

fig.suptitle(f"{meta['dataset']} / {meta['crop']}  (z={mid_z})", fontsize=10)
plt.tight_layout()
plt.show()

# %%
