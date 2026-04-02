import csv
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from scipy.ndimage import (
    binary_dilation,
    center_of_mass,
    distance_transform_edt,
    zoom as ndimage_zoom,
)
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 48 evaluated classes (from tested_classes.csv)
# ---------------------------------------------------------------------------
EVALUATED_CLASSES = [
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
N_CLASSES = len(EVALUATED_CLASSES)

INSTANCE_CLASSES = [
    "nuc", "ves", "endo", "lyso", "ld", "perox", "mito", "mt", "cell",
]

# Instance classes that are in EVALUATED_CLASSES
_EVALUATED_SET = set(EVALUATED_CLASSES)
EVALUATED_INSTANCE_CLASSES = [c for c in INSTANCE_CLASSES if c in _EVALUATED_SET]
N_INSTANCE_CLASSES = len(EVALUATED_INSTANCE_CLASSES)  # 9
INSTANCE_CLASS_INDEX = {name: i for i, name in enumerate(EVALUATED_INSTANCE_CLASSES)}

# Morphology groups + flow type for target generation and post-processing
INSTANCE_CLASS_CONFIG = {
    # Group 1: Convex — direct flows to center of mass
    "nuc":   {"group": 1, "flow_type": "direct"},
    "ves":   {"group": 1, "flow_type": "direct"},
    "endo":  {"group": 1, "flow_type": "direct"},
    "lyso":  {"group": 1, "flow_type": "direct"},
    "ld":    {"group": 1, "flow_type": "direct"},
    "perox": {"group": 1, "flow_type": "direct"},
    # Group 2: Non-convex / elongated — diffusion-based flows (heat equation)
    "cell":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 200,
              "boundary_only": True, "boundary_width": 20},
    "mito":  {"group": 2, "flow_type": "diffusion", "diffusion_iters": 200},
    # Group 3: Thin — direct flows
    "mt":    {"group": 3, "flow_type": "direct"},
}


# ---------------------------------------------------------------------------
# Zarr metadata helpers
# ---------------------------------------------------------------------------

def get_scale_info(zarr_grp_path: str):
    """Read multiscale metadata from a zarr group.

    Returns:
        offsets: dict mapping scale path -> [z, y, x] translation (world nm)
        resolutions: dict mapping scale path -> [z, y, x] voxel size (nm)
        shapes: dict mapping scale path -> volume shape (voxels)
    """
    grp = zarr.open(zarr_grp_path, mode="r")
    attrs = grp.attrs
    offsets, resolutions, shapes = {}, {}, {}
    for scale in attrs["multiscales"][0]["datasets"]:
        path = scale["path"]
        resolutions[path] = scale["coordinateTransformations"][0]["scale"]
        offsets[path] = scale["coordinateTransformations"][1]["translation"]
        shapes[path] = grp[path].shape
    return offsets, resolutions, shapes


def get_raw_path(em_base: str) -> Optional[str]:
    """Select fibsem-uint8 if available, else fibsem-uint16."""
    uint8_path = os.path.join(em_base, "fibsem-uint8")
    uint16_path = os.path.join(em_base, "fibsem-uint16")
    if os.path.isdir(uint8_path):
        return uint8_path
    if os.path.isdir(uint16_path):
        return uint16_path
    return None


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
# Normalization
# ---------------------------------------------------------------------------

@dataclass
class NormParams:
    min_val: float
    max_val: float
    inverted: bool


def load_norms(csv_path: str) -> Dict[str, NormParams]:
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
# Instance target helpers
# ---------------------------------------------------------------------------


def _generate_direct_flows(
    flows: np.ndarray, ch_z: int, ch_y: int, ch_x: int,
    instance_mask: np.ndarray, inst_ids: np.ndarray,
) -> None:
    """Generate direct unit vectors from each voxel to its instance's center of mass.

    Works for convex shapes where center of mass is inside the object.
    """
    D, H, W = instance_mask.shape
    coords = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

    for inst_id in inst_ids:
        if inst_id == 0:
            continue
        mask = (instance_mask == inst_id)
        if mask.sum() == 0:
            continue

        com = np.array(center_of_mass(mask), dtype=np.float32)

        dz = com[0] - coords[0]
        dy = com[1] - coords[1]
        dx = com[2] - coords[2]

        mag = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
        mag = np.clip(mag, 1e-8, None)

        flows[ch_z][mask] = (dz / mag)[mask]
        flows[ch_y][mask] = (dy / mag)[mask]
        flows[ch_x][mask] = (dx / mag)[mask]

        # At center of mass: flow = 0 (the sink)
        z0 = np.clip(int(round(com[0])), 0, D - 1)
        y0 = np.clip(int(round(com[1])), 0, H - 1)
        x0 = np.clip(int(round(com[2])), 0, W - 1)
        if mask[z0, y0, x0]:
            flows[ch_z][z0, y0, x0] = 0.0
            flows[ch_y][z0, y0, x0] = 0.0
            flows[ch_x][z0, y0, x0] = 0.0


def _generate_diffusion_flows(
    flows: np.ndarray, ch_z: int, ch_y: int, ch_x: int,
    instance_mask: np.ndarray, inst_ids: np.ndarray,
    n_iter: int = 200,
    adaptive_iters: bool = True,
    adaptive_factor: int = 6,
    spatial_mask: Optional[np.ndarray] = None,
) -> None:
    """Generate diffusion-based (heat equation) flows for non-convex shapes.

    For each instance (cropped to bounding box for speed):
    1. Initialize scalar field with coordinate values
    2. Diffuse iteratively within the instance mask
    3. Take spatial gradient → flow direction

    Per-instance loop guarantees no heat leaking between touching instances.

    At annotation boundaries (where spatial_mask=0), the field is allowed to
    diffuse freely (Neumann-like BC) so cut instances don't pile up against
    the crop edge. At true background boundaries (spatial_mask=1, instance=0),
    Dirichlet=0 is used so flow points inward.

    Args:
        adaptive_iters: If True, set iterations = adaptive_factor * max(bbox_extent)
            per instance, capped by n_iter. Smaller instances get fewer iterations.
        adaptive_factor: Multiplier for adaptive iteration count (default 6).
        spatial_mask: [D, H, W] float/bool — 1 inside annotated region, 0 outside.
            If None, Dirichlet=0 is used everywhere (original behavior).
    """
    D, H, W = instance_mask.shape

    for inst_id in inst_ids:
        if inst_id == 0:
            continue
        mask = (instance_mask == inst_id)
        if mask.sum() == 0:
            continue

        # Crop to bounding box with padding for efficiency
        where_mask = np.where(mask)
        pad = 2
        z0 = max(0, where_mask[0].min() - pad)
        z1 = min(D, where_mask[0].max() + pad + 1)
        y0 = max(0, where_mask[1].min() - pad)
        y1 = min(H, where_mask[1].max() + pad + 1)
        x0 = max(0, where_mask[2].min() - pad)
        x1 = min(W, where_mask[2].max() + pad + 1)

        crop_mask = mask[z0:z1, y0:y1, x0:x1]
        cd, ch, cw = crop_mask.shape
        crop_where = np.where(crop_mask)

        # Build the update mask: voxels where the field is allowed to evolve.
        # - Inside instance mask: always updated (the diffusion domain)
        # - Outside annotation boundary (spatial_mask=0): also updated, so the
        #   field diffuses freely into unannotated regions (Neumann-like BC at
        #   the annotation edge — no heat sink for cut instances)
        # - Annotated background (spatial_mask=1, instance=0): stays at 0
        #   (Dirichlet BC — correct for natural instance boundaries)
        if spatial_mask is not None:
            crop_spatial = spatial_mask[z0:z1, y0:y1, x0:x1] > 0.5
            # Update where: inside instance OR outside annotation
            update_mask = crop_mask | ~crop_spatial
        else:
            update_mask = crop_mask

        # Adaptive iteration count: scale with instance extent, cap at n_iter
        if adaptive_iters:
            max_extent = max(cd, ch, cw)
            inst_n_iter = min(adaptive_factor * max_extent, n_iter)
        else:
            inst_n_iter = n_iter

        inst_flows = np.zeros((3, cd, ch, cw), dtype=np.float64)

        for axis in range(3):
            # Initialize: coordinate values offset by crop origin
            offsets = [z0, y0, x0]
            field = np.zeros((cd, ch, cw), dtype=np.float64)
            field[crop_mask] = (crop_where[axis] + offsets[axis]).astype(np.float64)

            # Diffuse: field evolves inside the instance mask AND in
            # unannotated regions. Annotated background stays at 0.
            # Padded laplacian (edge mode) gives Neumann BC at array edges,
            # preventing heat sink at the bounding box boundary.
            for _ in range(inst_n_iter):
                fp = np.pad(field, 1, mode='edge')
                new_field = field.copy()

                for ax in range(3):
                    slc_f = [slice(1, -1)] * 3
                    slc_b = [slice(1, -1)] * 3
                    slc_f[ax] = slice(2, None)
                    slc_b[ax] = slice(None, -2)

                    laplacian = (
                        fp[tuple(slc_f)]
                        + fp[tuple(slc_b)]
                        - 2.0 * field
                    )
                    new_field += (1.0 / 6.0) * laplacian

                # Update instance voxels + unannotated region (free diffusion)
                # Annotated background stays at 0 (Dirichlet)
                field[update_mask] = new_field[update_mask]

            # Gradient of diffused field = flow direction
            inst_flows[axis] = np.gradient(field, axis=axis)

        # Normalize to unit vectors
        mag = np.sqrt(
            inst_flows[0] ** 2 + inst_flows[1] ** 2 + inst_flows[2] ** 2
        )
        mag = np.clip(mag, 1e-8, None)

        flows[ch_z][z0:z1, y0:y1, x0:x1][crop_mask] = (inst_flows[0] / mag)[crop_mask].astype(np.float32)
        flows[ch_y][z0:z1, y0:y1, x0:x1][crop_mask] = (inst_flows[1] / mag)[crop_mask].astype(np.float32)
        flows[ch_x][z0:z1, y0:y1, x0:x1][crop_mask] = (inst_flows[2] / mag)[crop_mask].astype(np.float32)


def _compute_boundary_map(
    instance_mask: np.ndarray, dilation_width: int = 2
) -> np.ndarray:
    """Inter-instance boundary map.

    1.0 where adjacent voxels have different non-zero instance IDs.
    Dilated by ``dilation_width`` voxels so boundaries are 2-4 voxels wide
    (thin boundaries are harder to learn).
    """
    boundary = np.zeros(instance_mask.shape, dtype=np.float32)
    for axis in range(instance_mask.ndim):
        slc_a = [slice(None)] * instance_mask.ndim
        slc_b = [slice(None)] * instance_mask.ndim
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)

        a = instance_mask[tuple(slc_a)]
        b = instance_mask[tuple(slc_b)]

        is_bnd = (a != b) & (a > 0) & (b > 0)
        boundary[tuple(slc_a)] = np.maximum(boundary[tuple(slc_a)], is_bnd)
        boundary[tuple(slc_b)] = np.maximum(boundary[tuple(slc_b)], is_bnd)

    if dilation_width > 1:
        struct = np.ones([3] * instance_mask.ndim)
        boundary = binary_dilation(
            boundary > 0, struct, iterations=dilation_width - 1
        ).astype(np.float32)

    return boundary


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellMapDataset3D(Dataset):
    """3D multi-label dataset for the CellMap Segmentation Challenge.

    Discovers all annotated crops from zarr stores, and samples random 3D
    patches returning raw EM + binary labels for all target classes + an
    annotation mask indicating which classes were annotated in the crop.
    """

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        norms_csv: str = None,
        target_classes: List[str] = None,
        target_resolution: float = 8.0,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_epoch: int = 1000,
        min_crop_voxels: int = 32,
        max_resolution_ratio: float = 2.0,
        cache_dir: str = None,
        seed: int = 42,
        transforms=None,
        skip_datasets: List[str] = None,
        instance_mode: bool = False,
        diffusion_iters: int = 200,
        adaptive_iters: bool = True,
        adaptive_factor: int = 6,
        instance_class_config: Optional[dict] = None,
        gpu_flows: bool = False,
        instance_classes: Optional[List[str]] = None,
    ):
        self.data_root = data_root
        self.target_classes = target_classes or EVALUATED_CLASSES
        self.n_classes = len(self.target_classes)
        self.target_resolution = target_resolution
        self.patch_size = np.array(patch_size)
        self.patch_world = self.patch_size * target_resolution
        self.samples_per_epoch = samples_per_epoch
        self.min_crop_voxels = min_crop_voxels
        self.max_resolution_ratio = max_resolution_ratio
        self.rng = np.random.default_rng(seed)
        self.transforms = transforms
        self.skip_datasets = set(skip_datasets or [])
        self.instance_mode = instance_mode
        self.diffusion_iters = diffusion_iters
        self.adaptive_iters = adaptive_iters
        self.adaptive_factor = adaptive_factor
        self.instance_class_config = instance_class_config
        self.gpu_flows = gpu_flows and instance_mode  # only meaningful in instance mode

        # Instance class subset (for cascade stages with fewer than 9 classes)
        self.instance_classes = instance_classes or list(EVALUATED_INSTANCE_CLASSES)
        self.n_instance_classes = len(self.instance_classes)

        # Instance class mapping (for instance_mode)
        if instance_mode:
            self.instance_class_to_idx = {
                c: i for i, c in enumerate(self.instance_classes)
            }

        # Build class name -> index lookup
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

        # Load normalization
        if norms_csv is None:
            raise ValueError(
                "norms_csv is required — path to a CSV with per-dataset "
                "normalization parameters (columns: dataset, min, max, inverted)"
            )
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
            f"patch_size={tuple(self.patch_size)}"
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

        logger.info("No cache found, running full discovery (this takes ~8 min)...")
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

    def _extract_patch(self, crop: CropInfo):
        """Extract a random 3D patch from a crop, resampled to isotropic target_resolution.

        All reads are done in world coordinates, then resampled to
        patch_size voxels at target_resolution. This correctly handles
        anisotropic raw/label resolutions.

        Returns:
            raw: np.ndarray [D, H, W] float32 in [0, 1]
            labels: np.ndarray [n_classes, D, H, W] float32 binary
            annotated_mask: np.ndarray [n_classes] bool — which classes are annotated
            spatial_mask: np.ndarray [D, H, W] float32 — 1 inside GT crop, 0 outside
        """
        D, H, W = self.patch_size
        crop_origin = np.array(crop.crop_origin_world)
        crop_extent = np.array(crop.crop_extent_world)
        crop_end = crop_origin + crop_extent
        crop_center = crop_origin + crop_extent / 2

        # Whether the crop is smaller than the patch in any dimension
        small_crop = np.any(crop_extent < self.patch_world)

        if small_crop:
            # Center the patch on the crop so raw EM context surrounds the GT
            sample_origin = crop_center - self.patch_world / 2
        else:
            # Random origin within valid range
            max_origin = crop_end - self.patch_world
            sample_origin = np.array([
                self.rng.uniform(crop_origin[i], max_origin[i])
                for i in range(3)
            ])

        # --- Read raw (always full patch, never padded black) ---
        raw_res = np.array(crop.raw_resolution)
        raw_off = np.array(crop.raw_offset_world)
        raw_shape = np.array(crop.raw_shape)

        # How many raw voxels cover the target world extent per axis
        raw_read_vox = np.ceil(self.patch_world / raw_res).astype(int)

        raw_start_vox = np.round((sample_origin - raw_off) / raw_res).astype(int)
        # Clamp so the full read window stays within the raw volume
        raw_start_vox = np.clip(raw_start_vox, 0, raw_shape - raw_read_vox)
        raw_start_vox = np.maximum(raw_start_vox, 0)
        raw_end_vox = np.minimum(raw_start_vox + raw_read_vox, raw_shape)

        # Update sample_origin to match what we actually read (after clamping)
        sample_origin = raw_off + raw_start_vox * raw_res

        raw_arr = zarr.open(
            os.path.join(crop.raw_zarr_path, crop.raw_scale_path), mode="r"
        )
        raw_patch = np.array(
            raw_arr[
                raw_start_vox[0]:raw_end_vox[0],
                raw_start_vox[1]:raw_end_vox[1],
                raw_start_vox[2]:raw_end_vox[2],
            ]
        )
        raw_patch = self._normalize_raw(raw_patch, crop.norm_params)

        # Resample raw to isotropic patch_size if needed (anisotropic resolution)
        if raw_patch.shape != (D, H, W):
            zoom_factors = (
                D / raw_patch.shape[0],
                H / raw_patch.shape[1],
                W / raw_patch.shape[2],
            )
            raw_patch = ndimage_zoom(raw_patch, zoom_factors, order=1)
        raw_patch = raw_patch[:D, :H, :W]

        # --- Build spatial annotation mask from crop bounding box ---
        # Compute which output voxels fall inside the annotated crop region.
        # The crop covers [crop_origin, crop_end) in world coords; we need
        # to map that to output-patch voxel coordinates [0..D, 0..H, 0..W].
        target_res = self.target_resolution
        crop_start_in_patch = np.maximum(
            np.round((crop_origin - sample_origin) / target_res).astype(int), 0
        )
        crop_end_in_patch = np.minimum(
            np.round((crop_end - sample_origin) / target_res).astype(int),
            self.patch_size,
        )

        # 3-D spatial mask: 1 inside crop, 0 outside
        spatial_mask = np.zeros((D, H, W), dtype=np.float32)
        spatial_mask[
            crop_start_in_patch[0]:crop_end_in_patch[0],
            crop_start_in_patch[1]:crop_end_in_patch[1],
            crop_start_in_patch[2]:crop_end_in_patch[2],
        ] = 1.0

        # --- Read labels (padded with zeros outside crop boundary) ---
        labels = np.zeros((self.n_classes, D, H, W), dtype=np.float32)
        annotated_mask = np.zeros(self.n_classes, dtype=bool)

        # Instance mode: also collect raw integer instance IDs
        if self.instance_mode:
            instance_ids = np.zeros((self.n_instance_classes, D, H, W), dtype=np.int32)
            instance_annotated = np.zeros(self.n_instance_classes, dtype=bool)

        for cls_name, cls_idx in self.class_to_idx.items():
            if cls_name not in crop.annotated_classes:
                continue

            annotated_mask[cls_idx] = True

            ci = crop.class_info[cls_name]
            cls_res = np.array(ci.resolution)
            cls_off = np.array(ci.offset_world)

            # How many class voxels cover the target world extent
            cls_patch_vox = np.ceil(self.patch_world / cls_res).astype(int)
            cls_patch_vox = np.maximum(cls_patch_vox, 1)

            cls_start_vox = np.round((sample_origin - cls_off) / cls_res).astype(int)
            cls_start_vox = np.maximum(cls_start_vox, 0)

            # Clip to label array bounds
            cls_shape = np.array(ci.shape)
            cls_end_vox = np.minimum(cls_start_vox + cls_patch_vox, cls_shape)
            cls_read_size = cls_end_vox - cls_start_vox

            if np.any(cls_read_size <= 0):
                continue

            try:
                label_arr = zarr.open(
                    os.path.join(ci.zarr_path, ci.scale_path), mode="r"
                )
                label_patch = np.array(
                    label_arr[
                        cls_start_vox[0]:cls_end_vox[0],
                        cls_start_vox[1]:cls_end_vox[1],
                        cls_start_vox[2]:cls_end_vox[2],
                    ]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to read label {cls_name} for "
                    f"{crop.dataset_name}/{crop.crop_id}: {e}"
                )
                continue

            # Binarize: >0 and !=255 means present
            label_binary = ((label_patch > 0) & (label_patch != 255)).astype(np.float32)

            # Instance mode: preserve raw integer IDs for instance classes
            # (before binarization, zeroing out the 255 exclusion marker)
            is_instance_cls = (
                self.instance_mode and cls_name in self.instance_class_to_idx
            )
            if is_instance_cls:
                label_int = label_patch.copy()
                label_int[label_patch == 255] = 0  # exclude marker → background

            # Place label into full patch-sized canvas, then resample
            # For small crops, label only covers part of the patch
            if label_binary.shape != tuple(cls_patch_vox):
                label_origin_in_patch = np.round(
                    (cls_off + cls_start_vox * cls_res - sample_origin) / cls_res
                ).astype(int)
                label_origin_in_patch = np.maximum(label_origin_in_patch, 0)

                canvas = np.zeros(tuple(cls_patch_vox), dtype=np.float32)
                slices = tuple(
                    slice(label_origin_in_patch[i],
                          min(label_origin_in_patch[i] + label_binary.shape[i], cls_patch_vox[i]))
                    for i in range(3)
                )
                trimmed = tuple(
                    slice(0, slices[i].stop - slices[i].start)
                    for i in range(3)
                )
                canvas[slices] = label_binary[trimmed]
                label_binary = canvas

                # Same placement for integer instance IDs
                if is_instance_cls:
                    int_canvas = np.zeros(tuple(cls_patch_vox), dtype=np.int32)
                    int_canvas[slices] = label_int[trimmed]
                    label_int = int_canvas

            # Resample to patch_size
            if label_binary.shape != (D, H, W):
                zoom_factors = (
                    D / label_binary.shape[0],
                    H / label_binary.shape[1],
                    W / label_binary.shape[2],
                )
                label_binary = ndimage_zoom(label_binary, zoom_factors, order=0)
                if is_instance_cls:
                    label_int = ndimage_zoom(label_int, zoom_factors, order=0)

            labels[cls_idx] = label_binary[:D, :H, :W]

            # Store instance IDs
            if is_instance_cls:
                inst_idx = self.instance_class_to_idx[cls_name]
                instance_ids[inst_idx] = label_int[:D, :H, :W]
                instance_annotated[inst_idx] = True

        if self.instance_mode:
            return raw_patch, labels, annotated_mask, spatial_mask, instance_ids, instance_annotated
        return raw_patch, labels, annotated_mask, spatial_mask

    # ------------------------------------------------------------------
    # Instance target generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_flow_targets(
        instance_ids: np.ndarray,
        class_names: Optional[List[str]] = None,
        class_config: Optional[dict] = None,
        diffusion_iters: int = 200,
        adaptive_iters: bool = True,
        adaptive_factor: int = 6,
        spatial_mask: Optional[np.ndarray] = None,
    ) -> tuple:
        """Compute per-class flow targets — delegates to topo.compute_flow_targets.

        Args:
            instance_ids: [N_INSTANCE, D, H, W] int32 — raw instance IDs.
            class_names: List of instance class names matching instance_ids channels.
            class_config: Per-class config dict (defaults to INSTANCE_CLASS_CONFIG).
            diffusion_iters: Max diffusion iterations for mito (cap for adaptive).
            adaptive_iters: Scale iterations with instance extent (default True).
            adaptive_factor: Multiplier for adaptive count (default 6).
            spatial_mask: [D, H, W] float/bool — 1 inside annotated region.

        Returns:
            flows: [N*3, D, H, W] float32 — per-class flow unit vectors.
            class_fg: [N, D, H, W] float32 — per-class foreground masks.
        """
        from topo import compute_flow_targets

        if class_config is None:
            class_config = INSTANCE_CLASS_CONFIG

        return compute_flow_targets(
            instance_ids,
            class_names=class_names,
            class_config=class_config,
            diffusion_iters=diffusion_iters,
            adaptive_iters=adaptive_iters,
            adaptive_factor=adaptive_factor,
            spatial_mask=spatial_mask,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        """Sample a random 3D patch.

        If idx < len(self.crops), it is used as a crop index (for use with
        ClassBalancedSampler). Otherwise a random crop is chosen.

        Augmentations (if set via `self.transforms`) are applied here inside
        the DataLoader worker process, allowing parallel augmentation across
        workers and keeping the GPU fed continuously.

        Returns:
            raw: torch.Tensor [1, D, H, W] float32
            labels: torch.Tensor [n_classes, D, H, W] float32
            annotated_mask: torch.Tensor [n_classes] bool — which classes annotated
            spatial_mask: torch.Tensor [1, D, H, W] float32 — 1 inside GT crop
            crop_name: str, e.g. "jrc_hela-2/crop1"

        When instance_mode=True, additionally returns:
            flow_targets: torch.Tensor [N*3, D, H, W] float32 — per-class flow vectors
            class_fg: torch.Tensor [N, D, H, W] float32 — per-class foreground masks
            instance_ids: torch.Tensor [N, D, H, W] int32 — raw instance IDs
            instance_ann: torch.Tensor [N] bool — which classes are annotated
        """
        if idx < len(self.crops):
            crop = self.crops[idx]
        else:
            crop = self.crops[self.rng.integers(len(self.crops))]

        if self.instance_mode:
            raw, labels, annotated_mask, spatial_mask, instance_ids, instance_annotated = (
                self._extract_patch(crop)
            )
        else:
            raw, labels, annotated_mask, spatial_mask = self._extract_patch(crop)

        raw = torch.from_numpy(raw[np.newaxis])  # [1, D, H, W]
        labels = torch.from_numpy(labels)
        annotated_mask = torch.from_numpy(annotated_mask)
        spatial_mask = torch.from_numpy(spatial_mask[np.newaxis])  # [1, D, H, W]

        # Apply augmentations inside worker
        if self.transforms is not None:
            if self.instance_mode:
                # Concatenate binary labels + integer instance IDs for joint
                # spatial augmentation. First 48 channels are binary labels,
                # last 10 channels are integer instance IDs.
                instance_ids_t = torch.from_numpy(instance_ids.astype(np.float32))
                combined = torch.cat([labels, instance_ids_t], dim=0)
                raw, combined = self.transforms(
                    raw, combined, n_binary_channels=self.n_classes
                )
                labels = combined[:self.n_classes]
                instance_ids = combined[self.n_classes:].round().long().numpy()
            else:
                raw, labels = self.transforms(raw, labels)

        crop_name = f"{crop.dataset_name}/{crop.crop_id}"

        if self.instance_mode:
            instance_ids_t = torch.from_numpy(instance_ids.astype(np.int32))
            instance_ann_t = torch.from_numpy(instance_annotated)

            if self.gpu_flows:
                # Skip CPU flow computation — return instance IDs for GPU computation
                # Tuple positions 5,6 are instance_ids and instance_ann (no flows/class_fg)
                return (
                    raw, labels, annotated_mask, spatial_mask, crop_name,
                    instance_ids_t, instance_ann_t,
                )

            # CPU flow computation (original path)
            sp_np = spatial_mask.squeeze(0).numpy() if isinstance(spatial_mask, torch.Tensor) else spatial_mask[0]
            flows, class_fg = self._compute_flow_targets(
                instance_ids,
                class_names=self.instance_classes,
                class_config=self.instance_class_config,
                diffusion_iters=self.diffusion_iters,
                adaptive_iters=self.adaptive_iters,
                adaptive_factor=self.adaptive_factor,
                spatial_mask=sp_np,
            )
            flows_t = torch.from_numpy(flows)        # [N*3, D, H, W]
            class_fg_t = torch.from_numpy(class_fg)  # [N, D, H, W]
            return (
                raw, labels, annotated_mask, spatial_mask, crop_name,
                flows_t, class_fg_t,
                instance_ids_t, instance_ann_t,
            )

        return raw, labels, annotated_mask, spatial_mask, crop_name

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
            f"  Patch size: {tuple(self.patch_size)}",
            f"",
            f"  Class presence (crops):",
        ]
        for i, cls in enumerate(self.target_classes):
            lines.append(f"    {cls:20s}: {int(class_counts[i]):4d} crops")
        return "\n".join(lines)
