"""mia_em_loader - Generic 3D multi-label EM data loading for CellMap-style zarr data.

Provides a PyTorch Dataset that discovers annotated crops from multiscale zarr
stores, samples random 3D patches, and returns raw EM + binary labels.

Works with any set of label classes — no hardcoded class lists.
"""

from .dataset import (
    CellMapDataset3D,
    # Data structures
    NormParams,
    ClassInfo,
    CropInfo,
    # Zarr helpers
    get_scale_info,
    get_raw_path,
    find_scale_for_resolution,
    load_norms,
)
from .sampler import ClassBalancedSampler

try:
    from .transforms import EMTransforms, get_train_transforms, get_train_transforms_from_config, get_val_transforms
except ImportError:
    pass
