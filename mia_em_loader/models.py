"""Data models for the crop database.

All metadata needed to load patches lives here. These dataclasses are the
contract between discovery (``discover.py``) and loading (``dataset.py``).
They can be serialized to/from dicts for JSON or database storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .migrations import migrate, CURRENT_VERSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass
class NormParams:
    min_val: float
    max_val: float
    inverted: bool

    def to_dict(self) -> dict:
        return {"min_val": self.min_val, "max_val": self.max_val, "inverted": self.inverted}

    @classmethod
    def from_dict(cls, d: dict) -> NormParams:
        return cls(min_val=d["min_val"], max_val=d["max_val"], inverted=d["inverted"])


# ---------------------------------------------------------------------------
# Per-class label metadata
# ---------------------------------------------------------------------------

@dataclass
class ClassInfo:
    """Metadata for one class label within a crop."""
    zarr_path: str
    scale_path: str
    resolution: List[float]
    offset_world: List[float]
    shape: Tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "zarr_path": self.zarr_path,
            "scale_path": self.scale_path,
            "resolution": list(self.resolution),
            "offset_world": list(self.offset_world),
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClassInfo:
        return cls(
            zarr_path=d["zarr_path"],
            scale_path=d["scale_path"],
            resolution=d["resolution"],
            offset_world=d["offset_world"],
            shape=tuple(d["shape"]),
        )


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------

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

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "crop_id": self.crop_id,
            "raw_zarr_path": self.raw_zarr_path,
            "raw_scale_path": self.raw_scale_path,
            "raw_resolution": list(self.raw_resolution),
            "raw_offset_world": list(self.raw_offset_world),
            "raw_shape": list(self.raw_shape),
            "norm_params": self.norm_params.to_dict(),
            "class_info": {k: v.to_dict() for k, v in self.class_info.items()},
            "annotated_classes": sorted(self.annotated_classes),
            "crop_origin_world": self.crop_origin_world,
            "crop_extent_world": self.crop_extent_world,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CropInfo:
        return cls(
            dataset_name=d["dataset_name"],
            crop_id=d["crop_id"],
            raw_zarr_path=d["raw_zarr_path"],
            raw_scale_path=d["raw_scale_path"],
            raw_resolution=d["raw_resolution"],
            raw_offset_world=d["raw_offset_world"],
            raw_shape=tuple(d["raw_shape"]),
            norm_params=NormParams.from_dict(d["norm_params"]),
            class_info={k: ClassInfo.from_dict(v) for k, v in d["class_info"].items()},
            annotated_classes=set(d["annotated_classes"]),
            crop_origin_world=d.get("crop_origin_world"),
            crop_extent_world=d.get("crop_extent_world"),
        )


# ---------------------------------------------------------------------------
# Crop database
# ---------------------------------------------------------------------------

@dataclass
class CropDatabase:
    """Self-contained database of discovered crops.

    Each crop carries its own normalization params — no top-level norms dict.
    This is the interface between discovery and data loading. The dataset
    class receives a ``CropDatabase`` and doesn't care how it was built.
    """
    version: int
    crops: List[CropInfo]
    # Discovery parameters (for provenance — not used at load time)
    discovery_params: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "discovery_params": self.discovery_params,
            "crops": [c.to_dict() for c in self.crops],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CropDatabase:
        d = migrate(d)
        return cls(
            version=d["version"],
            discovery_params=d.get("discovery_params"),
            crops=[CropInfo.from_dict(c) for c in d["crops"]],
        )

    def to_json(self, path: str) -> None:
        """Serialize to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved crop database ({len(self.crops)} crops) to {path}")

    @classmethod
    def from_json(cls, path: str) -> CropDatabase:
        """Load from a JSON file, applying version migrations if needed."""
        with open(path) as f:
            data = json.load(f)
        db = cls.from_dict(data)
        logger.info(f"Loaded crop database ({len(db.crops)} crops) from {path}")
        return db

    def filter_classes(self, target_classes: List[str]) -> CropDatabase:
        """Return a new database keeping only crops that annotate at least
        one of the given classes. Class info is pruned to target_classes only."""
        filtered = []
        for crop in self.crops:
            overlap = crop.annotated_classes & set(target_classes)
            if not overlap:
                continue
            filtered.append(crop)
        return CropDatabase(
            version=self.version,
            crops=filtered,
            discovery_params=self.discovery_params,
        )

    def summary(self, target_classes: List[str] = None) -> str:
        """Return a human-readable summary."""
        if target_classes is None:
            all_classes = set()
            for c in self.crops:
                all_classes |= c.annotated_classes
            target_classes = sorted(all_classes)

        n_datasets = len(set(c.dataset_name for c in self.crops))
        lines = [
            "CropDatabase Summary",
            f"  Version: {self.version}",
            f"  Datasets: {n_datasets}",
            f"  Crops: {len(self.crops)}",
            "",
            "  Class presence (crops):",
        ]
        for cls in target_classes:
            count = sum(1 for c in self.crops if cls in c.annotated_classes)
            lines.append(f"    {cls:20s}: {count:4d} crops")
        return "\n".join(lines)
