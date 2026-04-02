#!/usr/bin/env python3
"""Demo: loading CellMap Segmentation Challenge data with mia_em_loader.

This shows how to use the generic loader with CellMap-specific class lists
and normalization. Adapt the constants below for your own data.

Usage:
    python cellmap_demo.py

Requirements:
    - CellMap data at /nrs/cellmap/data (or change DATA_ROOT)
    - A norms.csv file (see below for format)
    - pip install mia-em-loader
"""

from torch.utils.data import DataLoader

from mia_em_loader import CellMapDataset3D, ClassBalancedSampler

# ---------------------------------------------------------------------------
# CellMap challenge class lists — customize these for your project
# ---------------------------------------------------------------------------

# All 48 evaluated classes from the CellMap Segmentation Challenge
CELLMAP_48_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
    "ld_mem", "ld_lum", "er_mem", "er_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in",
    "nuc", "golgi", "ves", "endo", "lyso", "ld", "eres",
    "perox_mem", "perox_lum", "perox",
    "mito", "er", "ne", "np", "chrom", "mt",
    "cell", "er_mem_all", "ne_mem_all",
]

# Or use a smaller subset for quick experiments
CELLMAP_COARSE_CLASSES = ["ecs", "cell", "nuc", "mito", "er", "golgi"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = "/nrs/cellmap/data"
NORMS_CSV = "norms.csv"  # CSV with columns: dataset, min, max, inverted

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # --- Full 48-class dataset at 8nm ---
    print("Creating dataset with 48 CellMap classes at 8nm...")
    ds = CellMapDataset3D(
        data_root=DATA_ROOT,
        norms_csv=NORMS_CSV,
        target_classes=CELLMAP_48_CLASSES,
        target_resolution=8.0,
        patch_size=(128, 128, 128),
        samples_per_epoch=1000,
    )
    print(ds.summary())

    # --- Sample one patch ---
    raw, labels, annotated_mask, spatial_mask, crop_name = ds[0]
    print(f"\nSample from: {crop_name}")
    print(f"  raw:            {raw.shape}  (min={raw.min():.2f}, max={raw.max():.2f})")
    print(f"  labels:         {labels.shape}")
    print(f"  annotated_mask: {annotated_mask.sum()}/{len(annotated_mask)} classes")
    print(f"  spatial_mask:   {spatial_mask.shape}  ({spatial_mask.mean():.1%} inside crop)")

    # --- With class-balanced sampling ---
    sampler = ClassBalancedSampler(ds, samples_per_epoch=1000, seed=42)
    loader = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=4, pin_memory=True)

    batch = next(iter(loader))
    raw_b, labels_b, ann_b, spatial_b, names_b = batch
    print(f"\nBatch shapes:")
    print(f"  raw:    {raw_b.shape}")
    print(f"  labels: {labels_b.shape}")
    print(f"  crops:  {names_b}")

    # --- Coarse classes only (faster, fewer channels) ---
    print("\n--- Coarse subset (6 classes) ---")
    ds_coarse = CellMapDataset3D(
        data_root=DATA_ROOT,
        norms_csv=NORMS_CSV,
        target_classes=CELLMAP_COARSE_CLASSES,
        target_resolution=8.0,
        patch_size=(128, 128, 128),
        samples_per_epoch=500,
    )
    raw, labels, annotated_mask, spatial_mask, crop_name = ds_coarse[0]
    print(f"  labels shape: {labels.shape}  (6 classes instead of 48)")
    print(f"  annotated:    {annotated_mask.sum()}/{len(annotated_mask)}")


if __name__ == "__main__":
    main()
