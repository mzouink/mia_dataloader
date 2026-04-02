"""mia_em_loader - Generic 3D multi-label EM data loading for CellMap-style zarr data.

Provides a PyTorch Dataset that loads patches from a pre-built CropDatabase.
Discovery is done separately via ``discover_crops``.
"""

from .models import (
    NormParams,
    ClassInfo,
    CropInfo,
    CropDatabase,
)
from .dataset import CellMapDataset3D
from .discover import discover_crops
from .sampler import ClassBalancedSampler

try:
    from .transforms import EMTransforms, get_train_transforms, get_train_transforms_from_config, get_val_transforms
except ImportError:
    print("Warning: Monai is not installed, so transforms are unavailable. Install monai to use EMTransforms.")
