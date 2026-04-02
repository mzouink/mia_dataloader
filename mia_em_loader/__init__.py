"""mia_em_loader - 3D multi-label EM data loading for the CellMap Segmentation Challenge.

Extracted from OrganelleNet. Provides datasets, transforms, samplers, and class
hierarchies for loading CellMap zarr data.
"""

from .dataset import (
    CellMapDataset3D,
    EVALUATED_CLASSES,
    INSTANCE_CLASSES,
    EVALUATED_INSTANCE_CLASSES,
    N_INSTANCE_CLASSES,
    INSTANCE_CLASS_INDEX,
    INSTANCE_CLASS_CONFIG,
    N_CLASSES,
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
from .class_mapping import (
    FINE_CLASSES,
    FINE_INDEX,
    ATOMIC_CLASSES,
    GROUP_COMPOSITION,
    MEDIUM_HIERARCHY,
    MEDIUM_CLASSES,
    COARSE_HIERARCHY,
    COARSE_CLASSES,
    fine_to_medium_matrix,
    fine_to_coarse_matrix,
    group_composition_matrix,
    compute_class_weights,
    compute_class_weights_from_crops,
    STAGE1_CLASSES,
    STAGE2_CLASSES,
    STAGE3_CLASSES,
    STAGE1_FLOW_CLASSES,
    STAGE2_FLOW_CLASSES,
    STAGE3_FLOW_CLASSES,
    STAGE_CLASSES,
    fine_to_stage1_matrix,
    fine_to_stage2_matrix,
)
from .sampler import ClassBalancedSampler

try:
    from .transforms import EMTransforms, get_train_transforms, get_train_transforms_from_config, get_val_transforms
except ImportError:
    pass

try:
    from .cascade_dataset import CascadeDataset3D
except ImportError:
    pass

try:
    from .gpu_diffusion import compute_flow_targets_gpu
except ImportError:
    pass
