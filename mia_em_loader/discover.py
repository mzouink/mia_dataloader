"""Discover CellMap crops and build a CropDatabase.

Walks a data root directory, reads zarr multiscale metadata, and produces
a self-contained ``CropDatabase`` that can be serialized to JSON.

Usage as a module::

    from mia_em_loader import discover_crops
    db = discover_crops("/nrs/cellmap/data", "norms.csv", target_classes=[...])
    db.to_json("crops.json")

Usage from command line::

    python -m mia_em_loader.discover \\
        --data-root /nrs/cellmap/data \\
        --norms-csv norms.csv \\
        --target-classes ecs cell nuc mito er golgi \\
        --output crops.json
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .migrations import CURRENT_VERSION
from .models import ClassInfo, CropDatabase, CropInfo, NormParams
from .utils import find_scale_for_resolution, get_raw_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Norms loading (CSV → dict of NormParams)
# ---------------------------------------------------------------------------

def load_norms_csv(csv_path: str) -> Dict[str, NormParams]:
    """Load per-dataset normalization parameters from CSV."""
    norms = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["dataset"].strip()
            norms[name] = NormParams(
                min_val=float(row["min"]),
                max_val=float(row["max"]),
                inverted=row["inverted"].strip() == "True",
            )
    return norms


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_crops(
    data_root: str,
    norms_csv: str,
    target_classes: List[str],
    target_resolution: float = 8.0,
    min_crop_voxels: int = 32,
    max_resolution_ratio: float = 2.0,
    skip_datasets: List[str] = None,
) -> CropDatabase:
    """Walk *data_root* and build a CropDatabase with all valid crops.

    Args:
        data_root: Root directory containing dataset directories.
        norms_csv: Path to CSV with columns: dataset, min, max, inverted.
        target_classes: Class names to look for in each crop.
        target_resolution: Target isotropic resolution in nm.
        min_crop_voxels: Minimum crop extent per axis in voxels.
        max_resolution_ratio: Max ratio between available and target resolution.
        skip_datasets: Dataset names to skip.

    Returns:
        A ``CropDatabase`` ready to be saved or passed to ``CellMapDataset3D``.
    """
    skip = set(skip_datasets or [])
    norms = load_norms_csv(norms_csv)
    crops: List[CropInfo] = []

    if not os.path.isdir(data_root):
        logger.warning(f"data_root {data_root} does not exist")
        return CropDatabase(
            version=CURRENT_VERSION, crops=[],
            discovery_params=_build_params(data_root, target_classes,
                                           target_resolution, min_crop_voxels,
                                           max_resolution_ratio, skip),
        )

    datasets = sorted(os.listdir(data_root))
    for dataset_name in datasets:
        if dataset_name in skip:
            logger.info(f"Skipping dataset: {dataset_name}")
            continue

        dataset_dir = os.path.join(data_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        gt_base = os.path.join(
            dataset_dir, f"{dataset_name}.zarr", "recon-1", "labels", "groundtruth"
        )
        if not os.path.isdir(gt_base):
            continue

        em_base = os.path.join(
            dataset_dir, f"{dataset_name}.zarr", "recon-1", "em"
        )
        raw_path = get_raw_path(em_base)
        if raw_path is None:
            continue

        raw_scale_info = find_scale_for_resolution(
            raw_path, target_resolution, max_ratio=1.01
        )
        if raw_scale_info is None:
            continue
        raw_scale_path, raw_res, raw_off, raw_shape = raw_scale_info

        if dataset_name not in norms:
            raise RuntimeError(
                f"Dataset '{dataset_name}' not found in norms CSV. "
                f"Please add normalization parameters for this dataset."
            )
        norm = norms[dataset_name]

        for crop_name in sorted(os.listdir(gt_base)):
            if not crop_name.startswith("crop"):
                continue
            crop_dir = os.path.join(gt_base, crop_name)
            if not os.path.isdir(crop_dir):
                continue

            crop_info = _build_crop_info(
                dataset_name, crop_name, crop_dir,
                raw_path, raw_scale_path, raw_res, raw_off, raw_shape,
                norm, target_classes, target_resolution, max_resolution_ratio,
                min_crop_voxels,
            )
            if crop_info is not None:
                crops.append(crop_info)

    logger.info(f"Discovery complete: {len(crops)} crops from {data_root}")

    return CropDatabase(
        version=CURRENT_VERSION,
        crops=crops,
        discovery_params=_build_params(data_root, target_classes,
                                       target_resolution, min_crop_voxels,
                                       max_resolution_ratio, skip),
    )


def _build_crop_info(
    dataset_name: str,
    crop_name: str,
    crop_dir: str,
    raw_zarr_path: str,
    raw_scale_path: str,
    raw_res: List[float],
    raw_off: List[float],
    raw_shape: Tuple[int, ...],
    norm: NormParams,
    target_classes: List[str],
    target_resolution: float,
    max_resolution_ratio: float,
    min_crop_voxels: int,
) -> Optional[CropInfo]:
    """Build CropInfo for a single crop."""
    crop = CropInfo(
        dataset_name=dataset_name,
        crop_id=crop_name,
        raw_zarr_path=raw_zarr_path,
        raw_scale_path=raw_scale_path,
        raw_resolution=raw_res,
        raw_offset_world=raw_off,
        raw_shape=raw_shape,
        norm_params=norm,
    )

    crop_subdirs = set(os.listdir(crop_dir))
    ref_class_info = None

    for cls_name in target_classes:
        if cls_name not in crop_subdirs:
            continue
        cls_path = os.path.join(crop_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        info = find_scale_for_resolution(
            cls_path, target_resolution, max_resolution_ratio
        )
        if info is None:
            continue

        scale_path, res, off, shape = info
        ci = ClassInfo(
            zarr_path=cls_path,
            scale_path=scale_path,
            resolution=res,
            offset_world=off,
            shape=shape,
        )
        crop.class_info[cls_name] = ci
        crop.annotated_classes.add(cls_name)
        if ref_class_info is None:
            ref_class_info = ci

    if ref_class_info is None:
        return None

    ref_off = np.array(ref_class_info.offset_world)
    ref_res = np.array(ref_class_info.resolution)
    ref_shape = np.array(ref_class_info.shape)
    crop.crop_origin_world = ref_off.tolist()
    crop.crop_extent_world = (ref_shape * ref_res).tolist()

    crop_voxels = np.array(crop.crop_extent_world) / target_resolution
    if np.any(crop_voxels < min_crop_voxels):
        return None

    return crop


def _build_params(data_root, target_classes, target_resolution,
                  min_crop_voxels, max_resolution_ratio, skip) -> dict:
    return {
        "data_root": data_root,
        "target_classes": list(target_classes),
        "target_resolution": target_resolution,
        "min_crop_voxels": min_crop_voxels,
        "max_resolution_ratio": max_resolution_ratio,
        "skip_datasets": sorted(skip),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover CellMap crops and write a crop database JSON."
    )
    parser.add_argument("--data-root", required=True, help="Root data directory")
    parser.add_argument("--norms-csv", required=True, help="Normalization CSV")
    parser.add_argument("--target-classes", nargs="+", required=True,
                        help="Class names to discover")
    parser.add_argument("--output", default="crops.json", help="Output JSON path")
    parser.add_argument("--target-resolution", type=float, default=8.0)
    parser.add_argument("--min-crop-voxels", type=int, default=32)
    parser.add_argument("--max-resolution-ratio", type=float, default=2.0)
    parser.add_argument("--skip-datasets", nargs="*", default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db = discover_crops(
        data_root=args.data_root,
        norms_csv=args.norms_csv,
        target_classes=args.target_classes,
        target_resolution=args.target_resolution,
        min_crop_voxels=args.min_crop_voxels,
        max_resolution_ratio=args.max_resolution_ratio,
        skip_datasets=args.skip_datasets,
    )
    db.to_json(args.output)
    print(db.summary(args.target_classes))


if __name__ == "__main__":
    main()
