"""Generic 3D multi-label dataset for CellMap-style zarr EM data.

Loads patches from a pre-built ``CropDatabase`` (produced by ``discover.py``).
No filesystem scanning happens at init time.

Works with any set of label classes — pass your own list via ``target_classes``.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom as ndimage_zoom

from .base import MiaDataset3D
from .models import CropDatabase, CropInfo, NormParams
from .utils import zarr_read

logger = logging.getLogger(__name__)


class CellMapDataset3D(MiaDataset3D):
    """Generic 3D multi-label dataset for CellMap-style zarr EM data.

    Receives a ``CropDatabase`` and samples random 3D patches returning
    raw EM + binary labels + annotation mask + spatial mask + metadata.

    Args:
        crop_db: A ``CropDatabase`` (from ``discover_crops`` or JSON).
        target_classes: List of class names to load. Crops that don't
            annotate any of these classes are skipped.
        target_resolution: Target isotropic resolution in nm.
        input_size: Raw input patch dimensions in voxels (D, H, W).
        output_size: Label output patch dimensions in voxels (D, H, W).
            If None, defaults to input_size.
        samples_per_epoch: Virtual epoch length.
        seed: Random seed.
        transforms: Augmentation callable(raw, labels) -> (raw, labels).
    """

    def __init__(
        self,
        crop_db: CropDatabase,
        target_classes: List[str],
        target_resolution: float = 8.0,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        output_size: Tuple[int, int, int] = None,
        samples_per_epoch: int = 1000,
        seed: int = 42,
        transforms=None,
    ):
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
        self.output_offset_world = (self.input_world - self.output_world) / 2
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.transforms = transforms

        # Build class name -> index lookup
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

        # Filter crops to those that annotate at least one target class
        self.crops = [
            crop for crop in crop_db.crops
            if crop.annotated_classes & set(self.target_classes)
        ]

        logger.info(
            f"CellMapDataset3D: {len(self.crops)} crops, "
            f"{self.n_classes} classes, "
            f"target_resolution={target_resolution}nm, "
            f"input_size={tuple(self.input_size)}, "
            f"output_size={tuple(self.output_size)}"
        )
        if len(self.crops) == 0:
            raise RuntimeError(
                f"No valid crops for classes {self.target_classes} "
                f"in the provided CropDatabase ({len(crop_db.crops)} total crops)"
            )

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_raw(raw: np.ndarray, norm: NormParams) -> np.ndarray:
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
        """Extract a random 3D patch from a crop.

        Raw is read at input_size. Labels and spatial_mask are produced at
        output_size, corresponding to the center region of the raw patch.

        Returns:
            raw, labels, annotated_mask, spatial_mask, metadata
        """
        iD, iH, iW = self.input_size
        oD, oH, oW = self.output_size
        crop_origin = np.array(crop.crop_origin_world)
        crop_extent = np.array(crop.crop_extent_world)
        crop_end = crop_origin + crop_extent
        crop_center = crop_origin + crop_extent / 2

        small_crop = np.any(crop_extent < self.input_world)

        if small_crop:
            sample_origin = crop_center - self.input_world / 2
        else:
            max_origin = crop_end - self.output_world - self.output_offset_world
            min_origin = crop_origin - self.output_offset_world
            sample_origin = np.array([
                self.rng.uniform(min_origin[i], max_origin[i])
                for i in range(3)
            ])

        # --- Read raw ---
        raw_res = np.array(crop.raw_resolution)
        raw_off = np.array(crop.raw_offset_world)
        raw_shape = np.array(crop.raw_shape)

        raw_read_vox = np.ceil(self.input_world / raw_res).astype(int)
        raw_start_vox = np.round((sample_origin - raw_off) / raw_res).astype(int)
        raw_start_vox = np.clip(raw_start_vox, 0, raw_shape - raw_read_vox)
        raw_start_vox = np.maximum(raw_start_vox, 0)
        raw_end_vox = np.minimum(raw_start_vox + raw_read_vox, raw_shape)

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

        if raw_patch.shape != (iD, iH, iW):
            zoom_factors = (
                iD / raw_patch.shape[0],
                iH / raw_patch.shape[1],
                iW / raw_patch.shape[2],
            )
            raw_patch = ndimage_zoom(raw_patch, zoom_factors, order=1)
        raw_patch = raw_patch[:iD, :iH, :iW]

        # --- Output region ---
        output_origin = sample_origin + self.output_offset_world
        target_res = self.target_resolution

        # --- Spatial mask ---
        crop_start_in_output = np.maximum(
            np.round((crop_origin - output_origin) / target_res).astype(int), 0
        )
        crop_end_in_output = np.minimum(
            np.round((crop_end - output_origin) / target_res).astype(int),
            self.output_size,
        )

        spatial_mask = np.zeros((oD, oH, oW), dtype=np.float32)
        spatial_mask[
            crop_start_in_output[0]:crop_end_in_output[0],
            crop_start_in_output[1]:crop_end_in_output[1],
            crop_start_in_output[2]:crop_end_in_output[2],
        ] = 1.0

        # --- Labels ---
        labels = np.zeros((self.n_classes, oD, oH, oW), dtype=np.float32)
        annotated_mask = np.zeros(self.n_classes, dtype=bool)

        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_name not in crop.annotated_classes:
                continue

            annotated_mask[cls_idx] = True

            ci = crop.class_info[cls_name]
            cls_res = np.array(ci.resolution)
            cls_off = np.array(ci.offset_world)

            cls_patch_vox = np.ceil(self.output_world / cls_res).astype(int)
            cls_patch_vox = np.maximum(cls_patch_vox, 1)

            cls_start_vox = np.round((output_origin - cls_off) / cls_res).astype(int)
            cls_start_vox = np.maximum(cls_start_vox, 0)

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

            label_binary = ((label_patch > 0) & (label_patch != 255)).astype(np.float32)

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

            if label_binary.shape != (oD, oH, oW):
                zoom_factors = (
                    oD / label_binary.shape[0],
                    oH / label_binary.shape[1],
                    oW / label_binary.shape[2],
                )
                label_binary = ndimage_zoom(label_binary, zoom_factors, order=0)

            labels[cls_idx] = label_binary[:oD, :oH, :oW]

        # --- Metadata ---
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

        Returns:
            raw: torch.Tensor [1, iD, iH, iW] float32 (input_size)
            labels: torch.Tensor [n_classes, oD, oH, oW] float32 (output_size)
            annotated_mask: torch.Tensor [n_classes] bool
            spatial_mask: torch.Tensor [1, oD, oH, oW] float32 (output_size)
            metadata: dict with provenance info
        """
        if idx < len(self.crops):
            crop = self.crops[idx]
        else:
            crop = self.crops[self.rng.integers(len(self.crops))]

        raw, labels, annotated_mask, spatial_mask, metadata = self._extract_patch(crop)

        raw = torch.from_numpy(raw[np.newaxis])
        labels = torch.from_numpy(labels)
        annotated_mask = torch.from_numpy(annotated_mask)
        spatial_mask = torch.from_numpy(spatial_mask[np.newaxis])

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
            "CellMapDataset3D Summary",
            f"  Datasets: {n_datasets}",
            f"  Crops: {len(self.crops)}",
            f"  Target classes: {self.n_classes}",
            f"  Resolution: {self.target_resolution}nm",
            f"  Input size: {tuple(self.input_size)}",
            f"  Output size: {tuple(self.output_size)}",
            "",
            "  Class presence (crops):",
        ]
        for i, cls in enumerate(self.target_classes):
            lines.append(f"    {cls:20s}: {int(class_counts[i]):4d} crops")
        return "\n".join(lines)
