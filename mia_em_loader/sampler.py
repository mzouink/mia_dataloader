"""Class-balanced sampling for CellMapDataset3D.

Problem: Some classes appear in 200+ crops, others in <10.
Uniform crop sampling means rare classes barely appear in training.

Solution: At each step, pick the least-seen class so far, then sample
a crop that contains it. This ensures all 48 classes get roughly equal
representation over the course of an epoch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """Sampler that balances class representation across batches.

    Algorithm:
        1. Build a crop-class matrix from the dataset (which crops annotate which classes)
        2. Maintain a running count of how many times each class has been seen
        3. At each step: pick the class with the lowest count, sample a crop
           that annotates it, return that crop index
        4. After returning a crop, increment counts for ALL classes that crop annotates

    This guarantees rare classes (e.g., perox with ~15 crops) get sampled
    as often as common classes (e.g., mito with 200+ crops).

    Args:
        dataset: A CellMapDataset3D instance (must have .crops and .get_crop_class_matrix()).
        samples_per_epoch: Number of samples per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.rng = np.random.default_rng(seed)

        # [n_crops x n_classes] boolean matrix
        self.crop_class_matrix = dataset.get_crop_class_matrix()
        self.n_crops, self.n_classes = self.crop_class_matrix.shape

        # For each class, precompute which crop indices annotate it
        self.class_to_crops = {}
        for c in range(self.n_classes):
            crop_indices = np.where(self.crop_class_matrix[:, c])[0]
            if len(crop_indices) > 0:
                self.class_to_crops[c] = crop_indices

        # Classes that actually have at least one crop
        self.active_classes = sorted(self.class_to_crops.keys())

    def __iter__(self):
        # Running count of how many times each class has been "seen"
        class_counts = np.zeros(self.n_classes, dtype=np.float64)

        for _ in range(self.samples_per_epoch):
            # Pick the least-seen active class (break ties randomly)
            active_counts = np.array([class_counts[c] for c in self.active_classes])
            min_count = active_counts.min()
            tied = [self.active_classes[i]
                    for i, v in enumerate(active_counts) if v == min_count]
            target_class = self.rng.choice(tied)

            # Sample a random crop that annotates this class
            crop_candidates = self.class_to_crops[target_class]
            crop_idx = self.rng.choice(crop_candidates)

            # Increment counts for all classes this crop annotates
            annotated = np.where(self.crop_class_matrix[crop_idx])[0]
            class_counts[annotated] += 1

            yield crop_idx

    def __len__(self):
        return self.samples_per_epoch
