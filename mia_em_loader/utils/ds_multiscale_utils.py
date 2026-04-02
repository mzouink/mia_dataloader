"""Utilities for reading OME-NGFF multiscale zarr metadata and arrays."""

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import zarr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zarr metadata helpers
# ---------------------------------------------------------------------------

def _read_zarr_array_shape(zarr_array_path: str) -> Tuple[int, ...]:
    """Read the shape of a zarr array from its .zarray metadata."""
    zarray_path = os.path.join(zarr_array_path, ".zarray")
    with open(zarray_path) as f:
        meta = json.load(f)
    return tuple(meta["shape"])


def get_scale_info(zarr_grp_path: str):
    """Read multiscale metadata from a zarr group's .zattrs.

    Returns:
        offsets: dict mapping scale path -> [z, y, x] translation (world nm)
        resolutions: dict mapping scale path -> [z, y, x] voxel size (nm)
        shapes: dict mapping scale path -> volume shape (voxels)
    """
    zattrs_path = os.path.join(zarr_grp_path, ".zattrs")
    with open(zattrs_path) as f:
        attrs = json.load(f)

    offsets, resolutions, shapes = {}, {}, {}
    for scale in attrs["multiscales"][0]["datasets"]:
        path = scale["path"]
        zarray_file = os.path.join(zarr_grp_path, path, ".zarray")
        if not os.path.exists(zarray_file):
            logger.debug("Skipping scale %s (no .zarray found)", path)
            continue
        resolutions[path] = scale["coordinateTransformations"][0]["scale"]
        offsets[path] = scale["coordinateTransformations"][1]["translation"]
        shapes[path] = _read_zarr_array_shape(os.path.join(zarr_grp_path, path))
    return offsets, resolutions, shapes


def find_scale_for_resolution(
    zarr_grp_path: str,
    target_res: float,
    max_ratio: float = 2.0,
) -> Optional[Tuple[str, List[float], List[float], Tuple[int, ...]]]:
    """Find the scale level closest to target_res (matching on Y axis).

    Returns (scale_path, resolution, offset, shape) or None.
    """
    offsets, resolutions, shapes = get_scale_info(zarr_grp_path)
    candidates = []
    for name, res in resolutions.items():
        y_res = res[1]
        ratio = max(y_res / target_res, target_res / y_res)
        if ratio <= max_ratio:
            candidates.append((abs(y_res - target_res), name, res, offsets[name], shapes[name]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, best_path, best_res, best_off, best_shape = candidates[0]
    return best_path, best_res, best_off, best_shape


# ---------------------------------------------------------------------------
# Zarr I/O
# ---------------------------------------------------------------------------

def zarr_read(zarr_array_path: str, slices: Tuple[slice, ...]) -> np.ndarray:
    """Read a slice from a zarr array on disk."""
    arr = zarr.open(zarr_array_path, mode="r")
    return np.asarray(arr[slices])


# ---------------------------------------------------------------------------
# Raw EM path selection
# ---------------------------------------------------------------------------

def get_raw_path(em_base: str) -> Optional[str]:
    """Select fibsem-uint8 if available, else fibsem-uint16."""
    uint8_path = os.path.join(em_base, "fibsem-uint8")
    uint16_path = os.path.join(em_base, "fibsem-uint16")
    if os.path.isdir(uint8_path):
        return uint8_path
    if os.path.isdir(uint16_path):
        return uint16_path
    return None
