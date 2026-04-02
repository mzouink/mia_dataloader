"""Class hierarchy and mapping utilities for the CellMap Segmentation Challenge.

Defines three levels of class grouping for the 48 evaluated classes:
- **Fine** (48 classes): all individually evaluated classes
- **Medium** (~18 groups): organelle-level grouping
- **Coarse** (7 super-classes): broad biological categories

Also provides:
- Composition maps: which fine classes compose each group class
- Mapping matrices: binary tensors for aggregating predictions between levels
- Class weight computation from dataset statistics
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .dataset import EVALUATED_CLASSES, INSTANCE_CLASSES, N_CLASSES

# ---------------------------------------------------------------------------
# Fine-level: the 48 evaluated classes (same order as EVALUATED_CLASSES)
# ---------------------------------------------------------------------------
FINE_CLASSES = list(EVALUATED_CLASSES)  # copy
FINE_INDEX = {name: i for i, name in enumerate(FINE_CLASSES)}

# ---------------------------------------------------------------------------
# Group composition within the 48 evaluated classes
# ---------------------------------------------------------------------------
# Maps each group class → list of its constituent fine-level classes.
# Only includes groups that are themselves in EVALUATED_CLASSES AND whose
# constituents are also in EVALUATED_CLASSES.
# Source: classes.csv from the challenge toolbox.

GROUP_COMPOSITION: Dict[str, List[str]] = {
    # Organelle groups (membrane + lumen + extras)
    "mito":       ["mito_mem", "mito_lum", "mito_ribo"],
    "golgi":      ["golgi_mem", "golgi_lum"],
    "ves":        ["ves_mem", "ves_lum"],
    "endo":       ["endo_mem", "endo_lum"],
    "lyso":       ["lyso_mem", "lyso_lum"],
    "ld":         ["ld_mem", "ld_lum"],
    "eres":       ["eres_mem", "eres_lum"],
    "perox":      ["perox_mem", "perox_lum"],
    # Nuclear envelope & pores
    "ne":         ["ne_mem", "ne_lum", "np_out", "np_in"],
    "np":         ["np_out", "np_in"],
    # Chromatin (note: nhchrom, nechrom not in evaluated set)
    "chrom":      ["hchrom", "echrom"],
    # Microtubules
    "mt":         ["mt_out", "mt_in"],
    # ER family (includes ERES + NE subcompartments)
    "er":         ["er_mem", "er_lum", "eres_mem", "eres_lum",
                   "ne_mem", "ne_lum", "np_out", "np_in"],
    # Nucleus (NE + chromatin + nucleoplasm; nhchrom/nechrom/nucleo not evaluated)
    "nuc":        ["ne_mem", "ne_lum", "np_out", "np_in",
                   "hchrom", "echrom", "nucpl"],
    # Membrane-only aggregates
    "er_mem_all": ["er_mem", "eres_mem", "ne_mem"],
    "ne_mem_all": ["ne_mem", "np_out", "np_in"],
    # Cell: pm + all interior organelles
    "cell":       ["pm", "mito_mem", "mito_lum", "mito_ribo",
                   "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
                   "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
                   "ld_mem", "ld_lum", "er_mem", "er_lum",
                   "eres_mem", "eres_lum", "ne_mem", "ne_lum",
                   "np_out", "np_in", "hchrom", "echrom", "nucpl",
                   "mt_out", "mt_in", "perox_mem", "perox_lum", "cyto"],
}

# Atomic classes: fine classes that are NOT group classes
ATOMIC_CLASSES = [c for c in FINE_CLASSES if c not in GROUP_COMPOSITION]
ATOMIC_INDEX = {name: i for i, name in enumerate(ATOMIC_CLASSES)}

# ---------------------------------------------------------------------------
# Medium hierarchy: ~18 organelle-level groups
# ---------------------------------------------------------------------------
# Each entry maps a medium-level name → list of fine-class members.
# Atomic classes that don't belong to any group are their own singleton group.

MEDIUM_HIERARCHY: Dict[str, List[str]] = {
    "extracellular":    ["ecs"],
    "plasma_membrane":  ["pm"],
    "mitochondria":     ["mito", "mito_mem", "mito_lum", "mito_ribo"],
    "golgi":            ["golgi", "golgi_mem", "golgi_lum"],
    "vesicles":         ["ves", "ves_mem", "ves_lum"],
    "endosomes":        ["endo", "endo_mem", "endo_lum"],
    "lysosomes":        ["lyso", "lyso_mem", "lyso_lum"],
    "lipid_droplets":   ["ld", "ld_mem", "ld_lum"],
    "er":               ["er", "er_mem", "er_lum", "er_mem_all",
                         "eres", "eres_mem", "eres_lum"],
    "nuclear_envelope": ["ne", "ne_mem", "ne_lum", "ne_mem_all",
                         "np", "np_out", "np_in"],
    "chromatin":        ["chrom", "hchrom", "echrom"],
    "nucleoplasm":      ["nucpl"],
    "nucleus":          ["nuc"],
    "microtubules":     ["mt", "mt_out", "mt_in"],
    "peroxisomes":      ["perox", "perox_mem", "perox_lum"],
    "cytoplasm":        ["cyto"],
    "cell":             ["cell"],
}

MEDIUM_CLASSES = list(MEDIUM_HIERARCHY.keys())
MEDIUM_INDEX = {name: i for i, name in enumerate(MEDIUM_CLASSES)}

# ---------------------------------------------------------------------------
# Coarse hierarchy: 7 super-classes
# ---------------------------------------------------------------------------

COARSE_HIERARCHY: Dict[str, List[str]] = {
    "extracellular": ["ecs"],
    "cell_boundary":  ["pm", "cell"],
    "nucleus":        ["nuc", "ne", "ne_mem", "ne_lum", "ne_mem_all",
                       "np", "np_out", "np_in",
                       "chrom", "hchrom", "echrom", "nucpl"],
    "membrane_organelles": [
        "mito", "mito_mem", "mito_lum", "mito_ribo",
        "er", "er_mem", "er_lum", "er_mem_all",
        "eres", "eres_mem", "eres_lum",
        "golgi", "golgi_mem", "golgi_lum",
        "lyso", "lyso_mem", "lyso_lum",
        "endo", "endo_mem", "endo_lum",
        "ld", "ld_mem", "ld_lum",
        "perox", "perox_mem", "perox_lum",
    ],
    "vesicular":      ["ves", "ves_mem", "ves_lum"],
    "cytoskeleton":   ["mt", "mt_out", "mt_in"],
    "cytoplasm":      ["cyto"],
}

COARSE_CLASSES = list(COARSE_HIERARCHY.keys())
COARSE_INDEX = {name: i for i, name in enumerate(COARSE_CLASSES)}

# ---------------------------------------------------------------------------
# Cascade stage class definitions (from cascade_stage_classes.md)
# ---------------------------------------------------------------------------
# Per-stage classes derived from per-class voxel statistics and organelle size.
# Rule: a class belongs to the earliest stage where it is reliably detectable.
# Stages are cumulative — each stage re-predicts all classes from prior stages.
# No membrane channels at stage 1 or 2; all membranes go to stage 3.

STAGE1_CLASSES: List[str] = ["ecs", "cell", "nuc", "ld"]  # 4 fine classes at 32nm
STAGE2_CLASSES: List[str] = [
    "ecs", "cell", "nuc", "ld",          # inherited from stage 1
    "cyto", "lyso", "perox", "mito",     # new at 16nm
]  # 8 fine classes
STAGE3_CLASSES: List[str] = FINE_CLASSES  # all 48 fine classes at 4nm

STAGE1_INDEX = {name: i for i, name in enumerate(STAGE1_CLASSES)}
STAGE2_INDEX = {name: i for i, name in enumerate(STAGE2_CLASSES)}
STAGE3_INDEX = FINE_INDEX  # same as fine

# Per-stage flow (instance) classes — each class × 3 directions (dz, dy, dx)
STAGE1_FLOW_CLASSES: List[str] = ["nuc", "ld", "cell"]          # 3 × 3 = 9 flow channels
STAGE2_FLOW_CLASSES: List[str] = [
    "nuc", "ld", "cell",                  # inherited from stage 1
    "lyso", "perox", "mito",             # new at 16nm
]  # 6 × 3 = 18 flow channels
STAGE3_FLOW_CLASSES: List[str] = [
    "nuc", "ves", "endo", "lyso", "ld",
    "perox", "mito", "mt", "cell",
]  # 9 × 3 = 27 flow channels

STAGE_CLASSES = {
    1: {"fine": STAGE1_CLASSES, "flow": STAGE1_FLOW_CLASSES,
        "n_fine": 4, "n_flow": 9},
    2: {"fine": STAGE2_CLASSES, "flow": STAGE2_FLOW_CLASSES,
        "n_fine": 8, "n_flow": 18},
    3: {"fine": STAGE3_CLASSES, "flow": STAGE3_FLOW_CLASSES,
        "n_fine": 48, "n_flow": 27},
}


# ---------------------------------------------------------------------------
# Mapping matrices
# ---------------------------------------------------------------------------

def _build_mapping_matrix(
    hierarchy: Dict[str, List[str]],
    group_classes: List[str],
    fine_classes: List[str] = FINE_CLASSES,
) -> torch.Tensor:
    """Build a binary mapping matrix [n_groups, n_fine].

    matrix[g, f] = 1.0 if fine class f belongs to group g.
    """
    fine_idx = {name: i for i, name in enumerate(fine_classes)}
    n_groups = len(group_classes)
    n_fine = len(fine_classes)
    matrix = torch.zeros(n_groups, n_fine)
    for g, group_name in enumerate(group_classes):
        members = hierarchy[group_name]
        for member in members:
            if member in fine_idx:
                matrix[g, fine_idx[member]] = 1.0
    return matrix


def fine_to_medium_matrix() -> torch.Tensor:
    """Binary matrix [n_medium, 48] mapping fine classes to medium groups."""
    return _build_mapping_matrix(MEDIUM_HIERARCHY, MEDIUM_CLASSES)


def fine_to_coarse_matrix() -> torch.Tensor:
    """Binary matrix [n_coarse, 48] mapping fine classes to coarse groups."""
    return _build_mapping_matrix(COARSE_HIERARCHY, COARSE_CLASSES)


def fine_to_stage1_matrix() -> torch.Tensor:
    """Binary matrix [4, 48] mapping stage 1 fine classes to full fine indices.

    Stage 1 classes are a direct subset of the 48 fine classes.
    matrix[s1_idx, fine_idx] = 1.0 if it's the same class.
    """
    n_s1 = len(STAGE1_CLASSES)
    n_fine = len(FINE_CLASSES)
    matrix = torch.zeros(n_s1, n_fine)
    for s1_idx, name in enumerate(STAGE1_CLASSES):
        if name in FINE_INDEX:
            matrix[s1_idx, FINE_INDEX[name]] = 1.0
    return matrix


def fine_to_stage2_matrix() -> torch.Tensor:
    """Binary matrix [8, 48] mapping stage 2 fine classes to full fine indices.

    Stage 2 classes are a direct subset of the 48 fine classes.
    matrix[s2_idx, fine_idx] = 1.0 if it's the same class.
    """
    n_s2 = len(STAGE2_CLASSES)
    n_fine = len(FINE_CLASSES)
    matrix = torch.zeros(n_s2, n_fine)
    for s2_idx, name in enumerate(STAGE2_CLASSES):
        if name in FINE_INDEX:
            matrix[s2_idx, FINE_INDEX[name]] = 1.0
    return matrix


def group_composition_matrix() -> torch.Tensor:
    """Binary matrix [48, 48] for group→atomic composition within fine classes.

    matrix[g, a] = 1.0 if fine class a is an atomic constituent of group g.
    For atomic classes, the row is all zeros.
    This is useful for enforcing consistency: a group prediction should be
    the union of its atomic predictions.
    """
    n = len(FINE_CLASSES)
    matrix = torch.zeros(n, n)
    for group_name, members in GROUP_COMPOSITION.items():
        if group_name in FINE_INDEX:
            g = FINE_INDEX[group_name]
            for member in members:
                if member in FINE_INDEX:
                    matrix[g, FINE_INDEX[member]] = 1.0
    return matrix


# ---------------------------------------------------------------------------
# Class weights from dataset statistics
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataset,
    n_samples: int = 200,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights by sampling the dataset.

    Args:
        dataset: A CellMapDataset3D instance.
        n_samples: Number of patches to sample for statistics.
        min_weight: Floor for weights (prevents exploding loss on rare classes).
        max_weight: Ceiling for weights.

    Returns:
        weights: [N_CLASSES] tensor of per-class weights.
    """
    # Count how many voxels are positive for each class, weighted by annotation
    voxel_counts = torch.zeros(N_CLASSES, dtype=torch.float64)
    annotated_counts = torch.zeros(N_CLASSES, dtype=torch.float64)

    for i in range(min(n_samples, len(dataset))):
        _raw, labels, mask = dataset[i]
        # labels: [N_CLASSES, D, H, W], mask: [N_CLASSES]
        for c in range(N_CLASSES):
            if mask[c]:
                annotated_counts[c] += 1
                voxel_counts[c] += labels[c].sum().item()

    # Frequency = fraction of annotated voxels that are positive
    total_voxels = labels.shape[1] * labels.shape[2] * labels.shape[3]
    freq = voxel_counts / (annotated_counts * total_voxels + 1e-8)

    # Inverse frequency, normalized so median weight = 1.0
    inv_freq = 1.0 / (freq + 1e-8)
    median_val = inv_freq[annotated_counts > 0].median()
    weights = (inv_freq / (median_val + 1e-8)).float()

    # Clamp and zero out classes that were never annotated in sample
    weights = weights.clamp(min_weight, max_weight)
    weights[annotated_counts == 0] = 1.0  # default weight for unseen classes

    return weights


def compute_class_weights_from_crops(dataset) -> torch.Tensor:
    """Compute class weights from crop-level annotation frequency (fast, no I/O).

    Uses the crop metadata to count how many crops annotate each class,
    then computes inverse-frequency weights. This is much faster than
    `compute_class_weights` since it doesn't read any zarr data.

    Args:
        dataset: A CellMapDataset3D instance (must have .crops populated).

    Returns:
        weights: [N_CLASSES] tensor of per-class weights.
    """
    crop_counts = torch.zeros(N_CLASSES, dtype=torch.float32)
    for crop in dataset.crops:
        for cls_name in crop.annotated_classes:
            if cls_name in FINE_INDEX:
                crop_counts[FINE_INDEX[cls_name]] += 1

    n_crops = len(dataset.crops)
    freq = crop_counts / max(n_crops, 1)

    # Inverse frequency, median-normalized
    inv_freq = 1.0 / (freq + 1e-8)
    valid = crop_counts > 0
    median_val = inv_freq[valid].median() if valid.any() else 1.0
    weights = inv_freq / (median_val + 1e-8)

    weights = weights.clamp(0.1, 10.0)
    weights[~valid] = 1.0
    return weights
