"""Generic 3D multi-label dataset for CellMap-style zarr EM data.

Discovers annotated crops from multiscale zarr stores, samples random 3D
patches, and returns raw EM + binary labels + annotation masks.

Uses zarr for array I/O and reads OME-NGFF metadata directly from
.zattrs/.zarray JSON files.

Works with any set of label classes — pass your own list via ``target_classes``.
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom as ndimage_zoom
from torch.utils.data import Dataset

from .utils import (
    get_scale_info,
    find_scale_for_resolution,
    get_raw_path,
    load_norms,
    zarr_read,
    NormParams,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------

@dataclass
class ClassInfo:
    """Metadata for one class label within a crop."""
    zarr_path: str        # path to the class zarr group (e.g., .../crop1/mito)
    scale_path: str       # e.g., "s1"
    resolution: List[float]   # [z, y, x] in nm
    offset_world: List[float]  # [z, y, x] in nm
    shape: Tuple[int, ...]     # voxels at this scale


@dataclass
class CropInfo:
    """All pre-computed metadata for one annotated crop."""
    dataset_name: str
    crop_id: str
    # Raw EM
    raw_zarr_path: str
    raw_scale_path: str
    raw_resolution: List[float]
    raw_offset_world: List[float]
    raw_shape: Tuple[int, ...]
    # Normalization
    norm_params: NormParams
    # Per-class label info (only for classes present in this crop)
    class_info: Dict[str, ClassInfo] = field(default_factory=dict)
    # Which of the target classes are annotated in this crop
    annotated_classes: set = field(default_factory=set)
    # Crop bounding box in world coordinates [z, y, x]
    crop_origin_world: Optional[List[float]] = None
    crop_extent_world: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellMapDataset3D(Dataset):
    """Generic 3D multi-label dataset for CellMap-style zarr EM data.

    Discovers all annotated crops from zarr stores, and samples random 3D
    patches returning raw EM + binary labels for all target classes + an
    annotation mask indicating which classes were annotated in the crop.

    Uses zarr for array reads.

    Args:
        data_root: Root directory containing dataset directories.
        norms_csv: Path to CSV with per-dataset normalization params
            (columns: dataset, min, max, inverted).
        target_classes: List of class names to load (must match zarr subdirectory
            names under each crop). Required.
        target_resolution: Target isotropic resolution in nm.
        input_size: Raw input patch dimensions in voxels (D, H, W).
        output_size: Label output patch dimensions in voxels (D, H, W).
            If None, defaults to input_size (no context padding).
            Must be <= input_size per axis. Labels and spatial_mask are
            cropped to the center output_size region of the input patch.
        samples_per_epoch: Virtual epoch length.
        min_crop_voxels: Minimum crop extent in voxels at target resolution.
        max_resolution_ratio: Max ratio between available and target resolution.
        cache_dir: Directory for caching discovered crop metadata.
        seed: Random seed.
        transforms: Augmentation callable(raw, labels) -> (raw, labels).
        skip_datasets: Dataset names to skip during discovery.
    """

    def __init__(
        self,
        data_root: str,
        norms_csv: str,
        target_classes: List[str],
        target_resolution: float = 8.0,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        output_size: Tuple[int, int, int] = None,
        samples_per_epoch: int = 1000,
        min_crop_voxels: int = 32,
        max_resolution_ratio: float = 2.0,
        cache_dir: str = None,
        seed: int = 42,
        transforms=None,
        skip_datasets: List[str] = None,
    ):
        self.data_root = data_root
        self.target_classes = list(target_classes)
        self.n_classes = len(self.target_classes)
        self.target_resolution = target_resolution
        self.input_size = np.array(input_size)
        self.output_size = np.array(output_size if output_size is not None else input_size)
        if np.any(self.output_size > self.input_size):
            raise ValueError(
                f"output_size {tuple(self.output_size)} must be <= input_size "
                f"{tuple(self.input_size)} on every axis"
            )
        self.input_world = self.input_size * target_resolution
        self.output_world = self.output_size * target_resolution
        # Offset from input origin to output origin in world coords
        self.output_offset_world = (self.input_world - self.output_world) / 2
        self.samples_per_epoch = samples_per_epoch
        self.min_crop_voxels = min_crop_voxels
        self.max_resolution_ratio = max_resolution_ratio
        self.rng = np.random.default_rng(seed)
        self.transforms = transforms
        self.skip_datasets = set(skip_datasets or [])

        # Build class name -> index lookup
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

        # Load normalization
        self.norms = load_norms(norms_csv)

        # Cache directory for discovered crops
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".mia_em_loader", "cache")
        self.cache_dir = cache_dir

        # Discover crops (or load from cache)
        self.crops: List[CropInfo] = []
        self._discover_crops_cached()

        logger.info(
            f"CellMapDataset3D: {len(self.crops)} crops, "
            f"{self.n_classes} classes, "
            f"target_resolution={target_resolution}nm, "
            f"input_size={tuple(self.input_size)}, "
            f"output_size={tuple(self.output_size)}"
        )
        if len(self.crops) == 0:
            raise RuntimeError(
                f"No valid crops found in {data_root} at resolution {target_resolution}nm"
            )

    # ------------------------------------------------------------------
    # Discovery with caching
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Build a unique cache key from discovery parameters."""
        key = json.dumps({
            "data_root": self.data_root,
            "target_classes": self.target_classes,
            "target_resolution": self.target_resolution,
            "min_crop_voxels": self.min_crop_voxels,
            "max_resolution_ratio": self.max_resolution_ratio,
            "skip_datasets": sorted(self.skip_datasets),
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        return os.path.join(
            self.cache_dir, f"crops_cache_{self._cache_key()}.pkl"
        )

    def _discover_crops_cached(self):
        """Load crops from cache if available, else discover and save."""
        path = self._cache_path()
        if os.path.exists(path):
            logger.info(f"Loading crop cache from {path}")
            with open(path, "rb") as f:
                self.crops = pickle.load(f)
            # Re-apply current norms (in case norms.csv changed)
            for crop in self.crops:
                if crop.dataset_name not in self.norms:
                    raise RuntimeError(
                        f"Dataset '{crop.dataset_name}' not found in norms.csv. "
                        f"Please add normalization parameters for this dataset."
                    )
                crop.norm_params = self.norms[crop.dataset_name]
            return

        logger.info("No cache found, running full discovery...")
        self._discover_crops()

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.crops, f)
        logger.info(f"Saved crop cache to {path}")

    def _discover_crops(self):
        """Walk data_root and build CropInfo for every valid crop."""
        if not os.path.isdir(self.data_root):
            logger.warning(f"data_root {self.data_root} does not exist")
            return

        datasets = sorted(os.listdir(self.data_root))
        for dataset_name in datasets:
            if dataset_name in self.skip_datasets:
                logger.info(f"Skipping dataset: {dataset_name}")
                continue

            dataset_dir = os.path.join(self.data_root, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue

            # Check for groundtruth labels
            gt_base = os.path.join(
                dataset_dir, f"{dataset_name}.zarr", "recon-1", "labels", "groundtruth"
            )
            if not os.path.isdir(gt_base):
                continue

            # Find raw EM path
            em_base = os.path.join(
                dataset_dir, f"{dataset_name}.zarr", "recon-1", "em"
            )
            raw_path = get_raw_path(em_base)
            if raw_path is None:
                continue

            # Check raw has target resolution
            raw_scale_info = find_scale_for_resolution(
                raw_path, self.target_resolution, max_ratio=1.01
            )
            if raw_scale_info is None:
                continue
            raw_scale_path, raw_res, raw_off, raw_shape = raw_scale_info

            # Get normalization — require entry in norms.csv
            if dataset_name not in self.norms:
                raise RuntimeError(
                    f"Dataset '{dataset_name}' not found in norms.csv. "
                    f"Please add normalization parameters for this dataset."
                )
            norm = self.norms[dataset_name]

            # Process each crop
            for crop_name in sorted(os.listdir(gt_base)):
                if not crop_name.startswith("crop"):
                    continue
                crop_dir = os.path.join(gt_base, crop_name)
                if not os.path.isdir(crop_dir):
                    continue

                crop_info = self._build_crop_info(
                    dataset_name, crop_name, crop_dir,
                    raw_path, raw_scale_path, raw_res, raw_off, raw_shape,
                    norm,
                )
                if crop_info is not None:
                    self.crops.append(crop_info)

    def _build_crop_info(
        self,
        dataset_name: str,
        crop_name: str,
        crop_dir: str,
        raw_zarr_path: str,
        raw_scale_path: str,
        raw_res: List[float],
        raw_off: List[float],
        raw_shape: Tuple[int, ...],
        norm: NormParams,
    ) -> Optional[CropInfo]:
        """Build CropInfo for a single crop by inspecting its class directories."""
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

        # Check which target classes exist in this crop
        crop_subdirs = set(os.listdir(crop_dir))
        ref_class_info = None

        for cls_name in self.target_classes:
            if cls_name not in crop_subdirs:
                continue
            cls_path = os.path.join(crop_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            info = find_scale_for_resolution(
                cls_path, self.target_resolution, self.max_resolution_ratio
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

        # Compute crop bounding box in world coordinates from the reference class
        ref_off = np.array(ref_class_info.offset_world)
        ref_res = np.array(ref_class_info.resolution)
        ref_shape = np.array(ref_class_info.shape)
        crop.crop_origin_world = ref_off.tolist()
        crop.crop_extent_world = (ref_shape * ref_res).tolist()

        # Skip crops that are too small (check in target-resolution voxels)
        crop_voxels = np.array(crop.crop_extent_world) / self.target_resolution
        if np.any(crop_voxels < self.min_crop_voxels):
            return None

        return crop

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _normalize_raw(self, raw: np.ndarray, norm: NormParams) -> np.ndarray:
        """Normalize raw EM to [0, 1] using per-dataset params."""
        raw = raw.astype(np.float32)
        denom = norm.max_val - norm.min_val
        if denom == 0:
            denom = 1.0
        raw = (raw - norm.min_val) / denom
        np.clip(raw, 0.0, 1.0, out=raw)
        if norm.inverted:
            raw = 1.0 - raw
        return raw

    @staticmethod
    def _zarr_read(zarr_array_path: str, slices: Tuple[slice, ...]) -> np.ndarray:
        """Read a slice from a zarr array."""
        return zarr_read(zarr_array_path, slices)

    def _extract_patch(self, crop: CropInfo):
        """Extract a random 3D patch from a crop, resampled to isotropic target_resolution.

        Raw is read at input_size. Labels and spatial_mask are produced at
        output_size, corresponding to the center region of the raw patch.

        Returns:
            raw: np.ndarray [iD, iH, iW] float32 in [0, 1]
            labels: np.ndarray [n_classes, oD, oH, oW] float32 binary
            annotated_mask: np.ndarray [n_classes] bool — which classes are annotated
            spatial_mask: np.ndarray [oD, oH, oW] float32 — 1 inside GT crop, 0 outside
            metadata: dict with provenance/debug info
        """
        iD, iH, iW = self.input_size
        oD, oH, oW = self.output_size
        crop_origin = np.array(crop.crop_origin_world)
        crop_extent = np.array(crop.crop_extent_world)
        crop_end = crop_origin + crop_extent
        crop_center = crop_origin + crop_extent / 2

        # Whether the crop is smaller than the input patch in any dimension
        small_crop = np.any(crop_extent < self.input_world)

        if small_crop:
            # Center the patch on the crop so raw EM context surrounds the GT
            sample_origin = crop_center - self.input_world / 2
        else:
            # Random origin within valid range (ensure output region fits in crop)
            max_origin = crop_end - self.output_world - self.output_offset_world
            min_origin = crop_origin - self.output_offset_world
            sample_origin = np.array([
                self.rng.uniform(min_origin[i], max_origin[i])
                for i in range(3)
            ])

        # --- Read raw (always full input patch, never padded black) ---
        raw_res = np.array(crop.raw_resolution)
        raw_off = np.array(crop.raw_offset_world)
        raw_shape = np.array(crop.raw_shape)

        # How many raw voxels cover the input world extent per axis
        raw_read_vox = np.ceil(self.input_world / raw_res).astype(int)

        raw_start_vox = np.round((sample_origin - raw_off) / raw_res).astype(int)
        # Clamp so the full read window stays within the raw volume
        raw_start_vox = np.clip(raw_start_vox, 0, raw_shape - raw_read_vox)
        raw_start_vox = np.maximum(raw_start_vox, 0)
        raw_end_vox = np.minimum(raw_start_vox + raw_read_vox, raw_shape)

        # Update sample_origin to match what we actually read (after clamping)
        sample_origin = raw_off + raw_start_vox * raw_res

        raw_patch = self._zarr_read(
            os.path.join(crop.raw_zarr_path, crop.raw_scale_path),
            (
                slice(int(raw_start_vox[0]), int(raw_end_vox[0])),
                slice(int(raw_start_vox[1]), int(raw_end_vox[1])),
                slice(int(raw_start_vox[2]), int(raw_end_vox[2])),
            ),
        )
        raw_patch = self._normalize_raw(np.asarray(raw_patch), crop.norm_params)

        # Resample raw to isotropic input_size if needed (anisotropic resolution)
        if raw_patch.shape != (iD, iH, iW):
            zoom_factors = (
                iD / raw_patch.shape[0],
                iH / raw_patch.shape[1],
                iW / raw_patch.shape[2],
            )
            raw_patch = ndimage_zoom(raw_patch, zoom_factors, order=1)
        raw_patch = raw_patch[:iD, :iH, :iW]

        # --- Output region: center of the input patch ---
        # output_origin is the world coordinate of the output region start
        output_origin = sample_origin + self.output_offset_world

        # --- Build spatial annotation mask from crop bounding box ---
        target_res = self.target_resolution
        crop_start_in_output = np.maximum(
            np.round((crop_origin - output_origin) / target_res).astype(int), 0
        )
        crop_end_in_output = np.minimum(
            np.round((crop_end - output_origin) / target_res).astype(int),
            self.output_size,
        )

        # 3-D spatial mask: 1 inside crop, 0 outside
        spatial_mask = np.zeros((oD, oH, oW), dtype=np.float32)
        spatial_mask[
            crop_start_in_output[0]:crop_end_in_output[0],
            crop_start_in_output[1]:crop_end_in_output[1],
            crop_start_in_output[2]:crop_end_in_output[2],
        ] = 1.0

        # --- Read labels at output_size (padded with zeros outside crop) ---
        labels = np.zeros((self.n_classes, oD, oH, oW), dtype=np.float32)
        annotated_mask = np.zeros(self.n_classes, dtype=bool)

        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_name not in crop.annotated_classes:
                continue

            annotated_mask[cls_idx] = True

            ci = crop.class_info[cls_name]
            cls_res = np.array(ci.resolution)
            cls_off = np.array(ci.offset_world)

            # How many class voxels cover the output world extent
            cls_patch_vox = np.ceil(self.output_world / cls_res).astype(int)
            cls_patch_vox = np.maximum(cls_patch_vox, 1)

            cls_start_vox = np.round((output_origin - cls_off) / cls_res).astype(int)
            cls_start_vox = np.maximum(cls_start_vox, 0)

            # Clip to label array bounds
            cls_shape = np.array(ci.shape)
            cls_end_vox = np.minimum(cls_start_vox + cls_patch_vox, cls_shape)
            cls_read_size = cls_end_vox - cls_start_vox

            if np.any(cls_read_size <= 0):
                continue

            try:
                label_patch = self._zarr_read(
                    os.path.join(ci.zarr_path, ci.scale_path),
                    (
                        slice(int(cls_start_vox[0]), int(cls_end_vox[0])),
                        slice(int(cls_start_vox[1]), int(cls_end_vox[1])),
                        slice(int(cls_start_vox[2]), int(cls_end_vox[2])),
                    ),
                )
                label_patch = np.asarray(label_patch)
            except Exception as e:
                logger.warning(
                    f"Failed to read label {cls_name} for "
                    f"{crop.dataset_name}/{crop.crop_id}: {e}"
                )
                continue

            # Binarize: >0 and !=255 means present
            label_binary = ((label_patch > 0) & (label_patch != 255)).astype(np.float32)

            # Place label into full output-sized canvas, then resample
            # For small crops, label only covers part of the output
            if label_binary.shape != tuple(cls_patch_vox):
                label_origin_in_output = np.round(
                    (cls_off + cls_start_vox * cls_res - output_origin) / cls_res
                ).astype(int)
                label_origin_in_output = np.maximum(label_origin_in_output, 0)

                canvas = np.zeros(tuple(cls_patch_vox), dtype=np.float32)
                slices = tuple(
                    slice(label_origin_in_output[i],
                          min(label_origin_in_output[i] + label_binary.shape[i], cls_patch_vox[i]))
                    for i in range(3)
                )
                trimmed = tuple(
                    slice(0, slices[i].stop - slices[i].start)
                    for i in range(3)
                )
                canvas[slices] = label_binary[trimmed]
                label_binary = canvas

            # Resample to output_size
            if label_binary.shape != (oD, oH, oW):
                zoom_factors = (
                    oD / label_binary.shape[0],
                    oH / label_binary.shape[1],
                    oW / label_binary.shape[2],
                )
                label_binary = ndimage_zoom(label_binary, zoom_factors, order=0)

            labels[cls_idx] = label_binary[:oD, :oH, :oW]

        # --- Build provenance metadata ---
        output_origin = sample_origin + self.output_offset_world
        metadata = {
            "dataset": crop.dataset_name,
            "crop": crop.crop_id,
            "raw_zarr_path": os.path.join(crop.raw_zarr_path, crop.raw_scale_path),
            "raw_voxel_slices": [
                [int(raw_start_vox[i]), int(raw_end_vox[i])] for i in range(3)
            ],
            "sample_origin_world": sample_origin.tolist(),
            "output_origin_world": output_origin.tolist(),
            "raw_resolution": crop.raw_resolution,
            "target_resolution": float(self.target_resolution),
            "annotated_classes": sorted(crop.annotated_classes),
            "crop_origin_world": crop.crop_origin_world,
            "crop_extent_world": crop.crop_extent_world,
        }

        return raw_patch, labels, annotated_mask, spatial_mask, metadata

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        """Sample a random 3D patch.

        If idx < len(self.crops), it is used as a crop index (for use with
        ClassBalancedSampler). Otherwise a random crop is chosen.

        Returns:
            raw: torch.Tensor [1, iD, iH, iW] float32 (input_size)
            labels: torch.Tensor [n_classes, oD, oH, oW] float32 (output_size)
            annotated_mask: torch.Tensor [n_classes] bool
            spatial_mask: torch.Tensor [1, oD, oH, oW] float32 (output_size)
            metadata: dict with provenance info (dataset, crop, paths,
                world coordinates, resolution, annotated classes)
        """
        if idx < len(self.crops):
            crop = self.crops[idx]
        else:
            crop = self.crops[self.rng.integers(len(self.crops))]

        raw, labels, annotated_mask, spatial_mask, metadata = self._extract_patch(crop)

        raw = torch.from_numpy(raw[np.newaxis])  # [1, D, H, W]
        labels = torch.from_numpy(labels)
        annotated_mask = torch.from_numpy(annotated_mask)
        spatial_mask = torch.from_numpy(spatial_mask[np.newaxis])  # [1, D, H, W]

        if self.transforms is not None:
            raw, labels = self.transforms(raw, labels)

        return raw, labels, annotated_mask, spatial_mask, metadata

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_crop_class_matrix(self) -> np.ndarray:
        """Build a [n_crops x n_classes] boolean matrix of class presence."""
        matrix = np.zeros((len(self.crops), self.n_classes), dtype=bool)
        for i, crop in enumerate(self.crops):
            for cls_name in crop.annotated_classes:
                if cls_name in self.class_to_idx:
                    matrix[i, self.class_to_idx[cls_name]] = True
        return matrix

    def summary(self) -> str:
        """Return a summary string of the dataset."""
        n_datasets = len(set(c.dataset_name for c in self.crops))
        class_matrix = self.get_crop_class_matrix()
        class_counts = class_matrix.sum(axis=0)
        lines = [
            f"CellMapDataset3D Summary",
            f"  Datasets: {n_datasets}",
            f"  Crops: {len(self.crops)}",
            f"  Target classes: {self.n_classes}",
            f"  Resolution: {self.target_resolution}nm",
            f"  Input size: {tuple(self.input_size)}",
            f"  Output size: {tuple(self.output_size)}",
            f"",
            f"  Class presence (crops):",
        ]
        for i, cls in enumerate(self.target_classes):
            lines.append(f"    {cls:20s}: {int(class_counts[i]):4d} crops")
        return "\n".join(lines)
