#%%

from torch.utils.data import DataLoader

from mia_em_loader import CellMapDataset3D, ClassBalancedSampler


# Or use a smaller subset for quick experiments
CLASSES = ["ecs", "cell", "nuc", "mito", "er", "golgi"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = "/nrs/cellmap/data"
NORMS_CSV = "norms.csv"  # CSV with columns: dataset, min, max, inverted

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


print("Creating dataset with 48 CellMap classes at 8nm...")
ds = CellMapDataset3D(
    data_root=DATA_ROOT,
    norms_csv=NORMS_CSV,
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

# --- With class-balanced sampling ---
# sampler = ClassBalancedSampler(ds, samples_per_epoch=1000, seed=42)
loader = DataLoader(
    ds, batch_size=4, 
    # sampler=sampler,
      num_workers=4,
    # pin_memory=True,
)

batch = next(iter(loader))
raw_b, labels_b, ann_b, spatial_b, meta_b = batch
print(f"\nBatch shapes:")
print(f"  raw:    {raw_b.shape}")
print(f"  labels: {labels_b.shape}")
print(f"  crops:  {[f'{d}/{c}' for d, c in zip(meta_b['dataset'], meta_b['crop'])]}")

ds_coarse = CellMapDataset3D(
    data_root=DATA_ROOT,
    norms_csv=NORMS_CSV,
    target_classes=CLASSES,
    target_resolution=8.0,
    input_size=(128, 128, 128),
    samples_per_epoch=500,
)
raw, labels, annotated_mask, spatial_mask, meta = ds_coarse[0]
print(f"  labels shape: {labels.shape}  (6 classes instead of 48)")
print(f"  annotated:    {annotated_mask.sum()}/{len(annotated_mask)}")

# %%
