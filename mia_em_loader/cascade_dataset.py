"""Multi-resolution dataset for cascade training.

Extracts patches at 3 resolutions (32nm, 16nm, 4nm) from the same physical
location. Each resolution gets its own raw EM patch and stage-specific labels.

All stages predict fine-level classes (subsets of the 48 EVALUATED_CLASSES),
so no hierarchy aggregation is needed — labels are read directly from zarr.

Usage:
    from mia_em_loader import CascadeDataset3D

    ds = CascadeDataset3D(
        data_root="/nrs/cellmap/data",
        norms_csv="path/to/norms.csv",
        stage_configs=[
            {"resolution": 32.0, "patch_size": (128,128,128), "n_classes": 4,
             "class_names": ["ecs", "cell", "nuc", "ld"]},
            {"resolution": 16.0, "patch_size": (128,128,128), "n_classes": 8,
             "class_names": ["ecs", "cell", "nuc", "ld", "cyto", "lyso", "perox", "mito"]},
            {"resolution": 4.0,  "patch_size": (160,160,160), "n_classes": 48},
        ],
    )
    sample = ds[0]
    # sample["patch_32nm"]: [1, 128, 128, 128]
    # sample["label_32nm"]: [4, 128, 128, 128]
    # ...
"""

from __future__ import annotations

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
from scipy.ndimage import zoom as ndimage_zoom
from torch.utils.data import Dataset

from .dataset import (
    EVALUATED_CLASSES,
    INSTANCE_CLASSES,
    INSTANCE_CLASS_CONFIG,
    INSTANCE_CLASS_INDEX,
    EVALUATED_INSTANCE_CLASSES,
    N_INSTANCE_CLASSES,
    N_CLASSES,
    CropInfo,
    ClassInfo,
    NormParams,
    find_scale_for_resolution,
    get_raw_path,
    get_scale_info,
    load_norms,
)
from .class_mapping import (
    FINE_CLASSES,
    FINE_INDEX,
    GROUP_COMPOSITION,
    STAGE1_CLASSES,
    STAGE2_CLASSES,
    STAGE3_CLASSES,
    STAGE1_FLOW_CLASSES,
    STAGE2_FLOW_CLASSES,
    STAGE3_FLOW_CLASSES,
    STAGE_CLASSES,
)

logger = logging.getLogger(__name__)


@dataclass
class CascadeStageConfig:
    """Configuration for one cascade stage."""
    resolution: float       # target resolution in nm
    patch_size: Tuple[int, int, int]
    n_classes: int          # number of fine output classes for this stage
    class_names: List[str]  # which fine class names this stage predicts
    flow_classes: List[str] = field(default_factory=list)  # instance classes for flows
    n_flow: int = 0         # number of flow channels (len(flow_classes) * 3)


@dataclass
class CascadeCropInfo:
    """Crop metadata extended for multi-resolution access."""
    base_crop: CropInfo
    # Per-resolution raw EM metadata: resolution → (scale_path, voxel_res, offset, shape)
    raw_scales: Dict[float, Tuple[str, List[float], List[float], Tuple[int, ...]]] = field(
        default_factory=dict
    )

    @property
    def dataset_name(self) -> str:
        return self.base_crop.dataset_name


class CascadeDataset3D(Dataset):
    """Multi-resolution 3D dataset for cascade training.

    Discovers crops that have raw EM available at all required resolutions,
    then samples concentric patches at each resolution from the same physical
    anchor point.

    Args:
        data_root: Root directory of CellMap zarr data.
        norms_csv: Path to normalization CSV.
        stage_configs: List of dicts with keys: resolution, patch_size, n_classes, class_names.
        samples_per_epoch: Virtual epoch length.
        min_crop_voxels: Minimum crop extent in voxels at finest resolution.
        max_resolution_ratio: Max ratio for scale matching.
        target_classes: Full list of fine-level classes for label discovery.
        cache_dir: Directory for caching discovered crops.
        seed: Random seed.
        transforms: Augmentation transforms (applied to finest resolution).
        skip_datasets: Dataset names to skip.
        instance_mode: Enable instance segmentation (flows on all stages).
        diffusion_iters: Max diffusion iterations for flow generation.
        adaptive_iters: Scale iterations with instance extent.
        adaptive_factor: Multiplier for adaptive iteration count.
        instance_class_config: Per-class instance config override.
        gpu_flows: Defer flow computation to GPU.
    """

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        norms_csv: str = None,
        stage_configs: List[dict] = None,
        samples_per_epoch: int = 1000,
        min_crop_voxels: int = 32,
        max_resolution_ratio: float = 2.0,
        target_classes: List[str] = None,
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
    ):
        self.data_root = data_root
        self.target_classes = target_classes or EVALUATED_CLASSES
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
        self.gpu_flows = gpu_flows and instance_mode

        # Build stage configurations
        _stage_flow = {
            1: STAGE1_FLOW_CLASSES, 2: STAGE2_FLOW_CLASSES, 3: STAGE3_FLOW_CLASSES,
        }
        if stage_configs is None:
            stage_configs = []
            for sid in [1, 2, 3]:
                sc = STAGE_CLASSES[sid]
                stage_configs.append({
                    "resolution": {1: 32.0, 2: 16.0, 3: 4.0}[sid],
                    "patch_size": (128, 128, 128),
                    "n_classes": sc["n_fine"],
                    "class_names": sc["fine"],
                    "flow_classes": sc["flow"],
                    "n_flow": sc["n_flow"],
                })

        self.stages: List[CascadeStageConfig] = []
        for i, sc in enumerate(stage_configs):
            class_names = sc.get("class_names")
            if class_names is None:
                sid = i + 1
                class_names = STAGE_CLASSES[sid]["fine"]
            flow_classes = sc.get("flow_classes", _stage_flow.get(i + 1, []))
            n_flow = sc.get("n_flow", len(flow_classes) * 3)
            self.stages.append(CascadeStageConfig(
                resolution=sc["resolution"],
                patch_size=tuple(sc["patch_size"]),
                n_classes=sc["n_classes"],
                class_names=list(class_names),
                flow_classes=list(flow_classes),
                n_flow=n_flow,
            ))

        # Resolutions needed (sorted coarse→fine)
        self.resolutions = sorted([s.resolution for s in self.stages], reverse=True)

        # Class lookup for full fine label reading
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

        # Stage-specific class→index maps
        self.stage_class_to_idx = []
        for stage in self.stages:
            self.stage_class_to_idx.append(
                {c: i for i, c in enumerate(stage.class_names)}
            )

        # Normalization
        if norms_csv is None:
            raise ValueError(
                "norms_csv is required — path to a CSV with per-dataset "
                "normalization parameters (columns: dataset, min, max, inverted)"
            )
        self.norms = load_norms(norms_csv)

        # Cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".mia_em_loader", "cache")
        self.cache_dir = cache_dir

        # Discover crops
        self.crops: List[CascadeCropInfo] = []
        self._discover_crops_cached()

        logger.info(
            "CascadeDataset3D: %d crops, %d stages, resolutions=%s",
            len(self.crops), len(self.stages), self.resolutions,
        )
        if len(self.crops) == 0:
            raise RuntimeError(
                f"No valid crops found with all resolutions {self.resolutions}"
            )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        key = json.dumps({
            "data_root": self.data_root,
            "target_classes": self.target_classes,
            "resolutions": self.resolutions,
            "min_crop_voxels": self.min_crop_voxels,
            "max_resolution_ratio": self.max_resolution_ratio,
            "skip_datasets": sorted(self.skip_datasets),
            "type": "cascade_v3_partial_res",
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        return os.path.join(
            self.cache_dir, f"cascade_crops_cache_{self._cache_key()}.pkl"
        )

    def _discover_crops_cached(self):
        path = self._cache_path()
        if os.path.exists(path):
            logger.info("Loading cascade crop cache from %s", path)
            with open(path, "rb") as f:
                self.crops = pickle.load(f)
            # Re-apply norms
            for ccrop in self.crops:
                ds_name = ccrop.base_crop.dataset_name
                if ds_name not in self.norms:
                    raise RuntimeError(
                        f"Dataset '{ds_name}' not found in norms.csv. "
                        f"Please add normalization parameters for this dataset."
                    )
                ccrop.base_crop.norm_params = self.norms[ds_name]
            return

        logger.info("No cascade cache found, running full discovery...")
        self._discover_crops()
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.crops, f)
        logger.info("Saved cascade crop cache to %s", path)

    def _discover_crops(self):
        """Walk data_root and build CascadeCropInfo for every valid crop."""
        if not os.path.isdir(self.data_root):
            logger.warning("data_root %s does not exist", self.data_root)
            return

        # Use finest resolution for label discovery
        finest_res = min(self.resolutions)

        datasets = sorted(os.listdir(self.data_root))
        for dataset_name in datasets:
            if dataset_name in self.skip_datasets:
                continue

            dataset_dir = os.path.join(self.data_root, dataset_name)
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

            # Check raw EM at each required resolution (allow partial)
            raw_scales = {}
            for res in self.resolutions:
                scale_info = find_scale_for_resolution(
                    raw_path, res, max_ratio=self.max_resolution_ratio
                )
                if scale_info is not None:
                    raw_scales[res] = scale_info

            if not raw_scales:
                continue

            # Use finest available resolution as the "primary" for CropInfo
            finest_available = min(raw_scales.keys())
            primary = raw_scales[finest_available]
            prim_scale_path, prim_res, prim_off, prim_shape = primary

            # Use finest available resolution for label discovery
            label_discovery_res = finest_available

            # Require entry in norms.csv
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
                    raw_path, prim_scale_path, prim_res, prim_off, prim_shape,
                    norm, label_discovery_res, raw_scales,
                )
                if crop_info is None:
                    continue

                # Build cascade crop with multi-res raw info
                ccrop = CascadeCropInfo(
                    base_crop=crop_info,
                    raw_scales=raw_scales,
                )
                self.crops.append(ccrop)

    def _build_crop_info(
        self,
        dataset_name, crop_name, crop_dir,
        raw_zarr_path, raw_scale_path, raw_res, raw_off, raw_shape,
        norm, target_resolution, raw_scales=None,
    ) -> Optional[CropInfo]:
        """Build CropInfo for a single crop (reuses CellMapDataset3D logic)."""
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

        for cls_name in self.target_classes:
            if cls_name not in crop_subdirs:
                continue
            cls_path = os.path.join(crop_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            info = find_scale_for_resolution(
                cls_path, target_resolution, self.max_resolution_ratio
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

        # Skip crops too small for the coarsest *available* stage
        available_res = set(raw_scales.keys()) if raw_scales else set(self.resolutions)
        coarsest_available = max(available_res)
        coarsest_stage = [s for s in self.stages if s.resolution == coarsest_available][0]
        coarsest_patch_world = np.array(coarsest_stage.patch_size) * coarsest_available
        crop_extent = np.array(crop.crop_extent_world)
        if np.any(crop_extent < coarsest_patch_world * 0.5):
            return None

        return crop

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _normalize_raw(self, raw: np.ndarray, norm: NormParams) -> np.ndarray:
        raw = raw.astype(np.float32)
        denom = norm.max_val - norm.min_val
        if denom == 0:
            denom = 1.0
        raw = (raw - norm.min_val) / denom
        np.clip(raw, 0.0, 1.0, out=raw)
        if norm.inverted:
            raw = 1.0 - raw
        return raw

    def _extract_raw_patch(
        self,
        ccrop: CascadeCropInfo,
        resolution: float,
        anchor_world: np.ndarray,
        patch_size: np.ndarray,
    ) -> np.ndarray:
        """Extract and resample a raw EM patch at a specific resolution."""
        scale_path, voxel_res, offset, shape = ccrop.raw_scales[resolution]
        voxel_res = np.array(voxel_res)
        offset = np.array(offset)
        shape = np.array(shape)
        D, H, W = patch_size

        patch_world = patch_size * resolution
        raw_read_vox = np.ceil(patch_world / voxel_res).astype(int)

        raw_start_vox = np.round((anchor_world - offset) / voxel_res).astype(int)
        raw_start_vox = np.clip(raw_start_vox, 0, shape - raw_read_vox)
        raw_start_vox = np.maximum(raw_start_vox, 0)
        raw_end_vox = np.minimum(raw_start_vox + raw_read_vox, shape)

        raw_arr = zarr.open(
            os.path.join(ccrop.base_crop.raw_zarr_path, scale_path), mode="r"
        )
        raw_patch = np.array(
            raw_arr[
                raw_start_vox[0]:raw_end_vox[0],
                raw_start_vox[1]:raw_end_vox[1],
                raw_start_vox[2]:raw_end_vox[2],
            ]
        )
        raw_patch = self._normalize_raw(raw_patch, ccrop.base_crop.norm_params)

        # Resample to target patch_size
        if raw_patch.shape != (D, H, W):
            zoom_factors = (
                D / raw_patch.shape[0],
                H / raw_patch.shape[1],
                W / raw_patch.shape[2],
            )
            raw_patch = ndimage_zoom(raw_patch, zoom_factors, order=1)
        return raw_patch[:D, :H, :W]

    def _extract_labels_for_stage(
        self,
        ccrop: CascadeCropInfo,
        stage_idx: int,
        anchor_world: np.ndarray,
        patch_size: np.ndarray,
        resolution: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract labels for a specific cascade stage.

        All stage classes are fine-level classes (subsets of EVALUATED_CLASSES),
        so labels are read directly — no hierarchy aggregation needed.

        For group classes (e.g., "cell", "nuc") that are also in the stage,
        we also build them from constituent fine classes if annotated.

        Args:
            ccrop: Cascade crop info.
            stage_idx: Index into self.stages.
            anchor_world: [3] world-coordinate origin.
            patch_size: [3] desired output voxel dimensions.
            resolution: Target resolution for this stage.

        Returns:
            labels: [n_classes, D, H, W] float32 binary
            annotated_mask: [n_classes] bool
        """
        stage = self.stages[stage_idx]
        D, H, W = patch_size
        n_classes = stage.n_classes
        patch_world = patch_size * resolution

        labels = np.zeros((n_classes, D, H, W), dtype=np.float32)
        annotated_mask = np.zeros(n_classes, dtype=bool)

        crop = ccrop.base_crop

        for cls_name, cls_idx in self.stage_class_to_idx[stage_idx].items():
            # Read class directly if annotated
            if cls_name in crop.annotated_classes and cls_name in crop.class_info:
                label_patch = self._read_single_label(
                    crop, cls_name, anchor_world, patch_size, resolution
                )
                if label_patch is not None:
                    labels[cls_idx] = np.maximum(labels[cls_idx], label_patch)
                    annotated_mask[cls_idx] = True

            # For group classes, also build from constituents
            if cls_name in GROUP_COMPOSITION:
                members = GROUP_COMPOSITION[cls_name]
                any_member_annotated = False
                for member in members:
                    if member in crop.annotated_classes and member in crop.class_info:
                        member_patch = self._read_single_label(
                            crop, member, anchor_world, patch_size, resolution
                        )
                        if member_patch is not None:
                            labels[cls_idx] = np.maximum(labels[cls_idx], member_patch)
                            any_member_annotated = True
                if any_member_annotated:
                    annotated_mask[cls_idx] = True

        return labels, annotated_mask

    def _read_single_label(
        self,
        crop: CropInfo,
        cls_name: str,
        anchor_world: np.ndarray,
        patch_size: np.ndarray,
        resolution: float,
    ) -> Optional[np.ndarray]:
        """Read a single class label from zarr, resampled to target patch_size."""
        ci = crop.class_info.get(cls_name)
        if ci is None:
            return None

        D, H, W = patch_size
        patch_world = patch_size * resolution
        cls_res = np.array(ci.resolution)
        cls_off = np.array(ci.offset_world)
        cls_shape = np.array(ci.shape)

        cls_patch_vox = np.ceil(patch_world / cls_res).astype(int)
        cls_patch_vox = np.maximum(cls_patch_vox, 1)

        cls_start_vox = np.round((anchor_world - cls_off) / cls_res).astype(int)
        cls_start_vox = np.maximum(cls_start_vox, 0)
        cls_end_vox = np.minimum(cls_start_vox + cls_patch_vox, cls_shape)
        cls_read_size = cls_end_vox - cls_start_vox

        if np.any(cls_read_size <= 0):
            return None

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
            logger.warning("Failed to read label %s: %s", cls_name, e)
            return None

        # Binarize
        label_binary = ((label_patch > 0) & (label_patch != 255)).astype(np.float32)

        # Resample to target patch_size
        if label_binary.shape != (D, H, W):
            zoom_factors = (
                D / label_binary.shape[0],
                H / label_binary.shape[1],
                W / label_binary.shape[2],
            )
            label_binary = ndimage_zoom(label_binary, zoom_factors, order=0)

        return label_binary[:D, :H, :W]

    def _extract_instance_ids_for_stage(
        self,
        ccrop: CascadeCropInfo,
        stage_idx: int,
        anchor_world: np.ndarray,
        patch_size: np.ndarray,
        resolution: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract raw integer instance IDs for a specific stage's flow classes.

        Returns:
            instance_ids: [n_flow_classes, D, H, W] int32
            instance_ann: [n_flow_classes] bool
        """
        stage = self.stages[stage_idx]
        D, H, W = patch_size
        flow_classes = stage.flow_classes
        n_flow_cls = len(flow_classes)
        patch_world = patch_size * resolution
        crop = ccrop.base_crop

        instance_ids = np.zeros((n_flow_cls, D, H, W), dtype=np.int32)
        instance_ann = np.zeros(n_flow_cls, dtype=bool)

        for flow_idx, cls_name in enumerate(flow_classes):
            if cls_name not in crop.annotated_classes:
                continue
            ci = crop.class_info.get(cls_name)
            if ci is None:
                continue

            cls_res = np.array(ci.resolution)
            cls_off = np.array(ci.offset_world)
            cls_shape = np.array(ci.shape)

            cls_patch_vox = np.ceil(patch_world / cls_res).astype(int)
            cls_patch_vox = np.maximum(cls_patch_vox, 1)
            cls_start_vox = np.round((anchor_world - cls_off) / cls_res).astype(int)
            cls_start_vox = np.maximum(cls_start_vox, 0)
            cls_end_vox = np.minimum(cls_start_vox + cls_patch_vox, cls_shape)

            if np.any(cls_end_vox - cls_start_vox <= 0):
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
            except Exception:
                continue

            label_int = label_patch.copy()
            label_int[label_patch == 255] = 0

            if label_int.shape != (D, H, W):
                zoom_factors = (
                    D / label_int.shape[0],
                    H / label_int.shape[1],
                    W / label_int.shape[2],
                )
                label_int = ndimage_zoom(label_int, zoom_factors, order=0)

            instance_ids[flow_idx] = label_int[:D, :H, :W]
            instance_ann[flow_idx] = True

        return (
            torch.from_numpy(instance_ids),
            torch.from_numpy(instance_ann),
        )

    def _extract_multires_patch(self, ccrop: CascadeCropInfo) -> dict:
        """Extract concentric patches at all resolutions from the same anchor.

        Returns dict with keys like patch_32nm, label_32nm, etc.
        """
        crop = ccrop.base_crop
        crop_origin = np.array(crop.crop_origin_world)
        crop_extent = np.array(crop.crop_extent_world)
        crop_end = crop_origin + crop_extent

        # The coarsest *available* stage has the largest FOV — anchor must fit it
        available_stages = [s for s in self.stages if s.resolution in ccrop.raw_scales]
        coarsest = available_stages[0] if available_stages else self.stages[0]
        coarsest_patch_world = np.array(coarsest.patch_size) * coarsest.resolution

        # Random anchor within valid range (ensuring coarsest patch fits)
        if np.any(crop_extent < coarsest_patch_world):
            # Center on crop
            anchor = crop_origin + crop_extent / 2 - coarsest_patch_world / 2
        else:
            max_origin = crop_end - coarsest_patch_world
            anchor = np.array([
                self.rng.uniform(crop_origin[i], max_origin[i])
                for i in range(3)
            ])

        # The anchor is the origin of the coarsest patch.
        # Finer patches are concentric (centered on the same physical center).
        coarsest_center = anchor + coarsest_patch_world / 2

        result = {}

        for stage_idx, stage in enumerate(self.stages):
            patch_size = np.array(stage.patch_size)
            res = stage.resolution
            patch_world = patch_size * res
            stage_anchor = coarsest_center - patch_world / 2

            # Resolution tag for dict keys
            res_tag = f"{int(res)}nm"

            # Check if this resolution is available for this crop
            if res not in ccrop.raw_scales:
                # Emit zero tensors so the batch can still collate
                D, H, W = patch_size
                result[f"patch_{res_tag}"] = torch.zeros(1, D, H, W)
                result[f"label_{res_tag}"] = torch.zeros(stage.n_classes, D, H, W)
                result[f"annotated_{res_tag}"] = torch.zeros(stage.n_classes, dtype=torch.bool)
                result[f"spatial_{res_tag}"] = torch.zeros(1, D, H, W)
                result[f"stage_available_{res_tag}"] = torch.tensor(False)
                if self.instance_mode and stage.flow_classes:
                    n_flow_cls = len(stage.flow_classes)
                    result[f"instance_ids_{res_tag}"] = torch.zeros(n_flow_cls, D, H, W, dtype=torch.int32)
                    result[f"instance_ann_{res_tag}"] = torch.zeros(n_flow_cls, dtype=torch.bool)
                continue

            result[f"stage_available_{res_tag}"] = torch.tensor(True)

            # Extract raw EM
            raw_patch = self._extract_raw_patch(
                ccrop, res, stage_anchor, patch_size
            )
            result[f"patch_{res_tag}"] = torch.from_numpy(raw_patch[np.newaxis])

            # Extract labels for this stage
            labels, ann_mask = self._extract_labels_for_stage(
                ccrop, stage_idx, stage_anchor, patch_size, res,
            )
            result[f"label_{res_tag}"] = torch.from_numpy(labels)
            result[f"annotated_{res_tag}"] = torch.from_numpy(ann_mask)

            # Spatial mask: which voxels fall inside the annotated crop
            spatial_mask = np.zeros(tuple(patch_size), dtype=np.float32)
            crop_start_in_patch = np.maximum(
                np.round((crop_origin - stage_anchor) / res).astype(int), 0
            )
            crop_end_in_patch = np.minimum(
                np.round((crop_end - stage_anchor) / res).astype(int),
                patch_size,
            )
            spatial_mask[
                crop_start_in_patch[0]:crop_end_in_patch[0],
                crop_start_in_patch[1]:crop_end_in_patch[1],
                crop_start_in_patch[2]:crop_end_in_patch[2],
            ] = 1.0
            result[f"spatial_{res_tag}"] = torch.from_numpy(spatial_mask[np.newaxis])

            # Per-stage instance IDs for flow computation
            if self.instance_mode and stage.flow_classes:
                inst_ids, inst_ann = self._extract_instance_ids_for_stage(
                    ccrop, stage_idx, stage_anchor, patch_size, res,
                )
                result[f"instance_ids_{res_tag}"] = inst_ids
                result[f"instance_ann_{res_tag}"] = inst_ann

        result["crop_name"] = f"{crop.dataset_name}/{crop.crop_id}"
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        """Sample a random multi-resolution patch.

        Returns dict with keys:
            patch_32nm: [1, D, H, W]
            label_32nm: [n_stage1, D, H, W]
            annotated_32nm: [n_stage1] bool
            spatial_32nm: [1, D, H, W]
            instance_ids_32nm: [n_stage1_flow, D, H, W] int32
            instance_ann_32nm: [n_stage1_flow] bool
            (same pattern for 16nm, 4nm)
            crop_name: str
        """
        if idx < len(self.crops):
            ccrop = self.crops[idx]
        else:
            ccrop = self.crops[self.rng.integers(len(self.crops))]

        return self._extract_multires_patch(ccrop)

    def summary(self) -> str:
        """Return a summary string for logging."""
        lines = [
            f"CascadeDataset3D: {len(self.crops)} crops",
            f"  Stages: {len(self.stages)}",
        ]
        for i, s in enumerate(self.stages):
            n_avail = sum(1 for c in self.crops if s.resolution in c.raw_scales)
            lines.append(
                f"    Stage {i+1}: {s.resolution}nm, {s.n_classes} fine, "
                f"{len(s.flow_classes)} flow classes ({s.n_flow}ch), "
                f"patch={s.patch_size}, {n_avail}/{len(self.crops)} crops available"
            )
        return "\n".join(lines)
