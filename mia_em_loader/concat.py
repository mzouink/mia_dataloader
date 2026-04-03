"""Combine multiple EM datasets into a single dataset.

``ConcatEMDataset`` wraps any number of datasets (``MiaDataset3D`` subclasses
or duck-typed equivalents) and presents them as one. Supports weighted
sampling across datasets.

Usage::

    ds = ConcatEMDataset(
        [cellmap_ds, custom_ds],
        weights=[0.7, 0.3],
        samples_per_epoch=2000,
    )
    loader = DataLoader(ds, batch_size=4)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import MiaDataset3D, validate_em_dataset

logger = logging.getLogger(__name__)


class ConcatEMDataset(MiaDataset3D):
    """Concatenate multiple EM datasets with optional weighted sampling.

    At init time, validates that all sub-datasets share the same
    ``target_classes``, ``input_size``, and ``output_size``.

    Args:
        datasets: List of datasets to combine. Each must conform to the
            ``MiaDataset3D`` contract (either by inheritance or duck typing).
        weights: Sampling weight per dataset. If None, weights are
            proportional to each dataset's crop count.
        samples_per_epoch: Virtual epoch length. If None, uses the sum
            of all sub-dataset lengths.
        seed: Random seed.
    """

    def __init__(
        self,
        datasets: list,
        weights: List[float] = None,
        samples_per_epoch: int = None,
        seed: int = 42,
    ):
        if len(datasets) == 0:
            raise ValueError("ConcatEMDataset requires at least one dataset")

        # Use the first dataset as the reference for contract checking
        ref = datasets[0]

        # Validate every dataset against the reference
        for i, ds in enumerate(datasets):
            try:
                validate_em_dataset(
                    ds,
                    target_classes=ref.target_classes,
                    input_size=tuple(int(x) for x in ref.input_size),
                    output_size=tuple(int(x) for x in ref.output_size),
                )
            except ValueError as e:
                raise ValueError(
                    f"Dataset {i} ({type(ds).__name__}) is incompatible "
                    f"with dataset 0 ({type(ref).__name__}):\n{e}"
                ) from e

        self.datasets = datasets
        self.target_classes = list(ref.target_classes)
        self.n_classes = ref.n_classes
        self.input_size = np.array(ref.input_size)
        self.output_size = np.array(ref.output_size)
        self.rng = np.random.default_rng(seed)

        # Build merged crop list (for ClassBalancedSampler compatibility)
        self.crops = []
        # Map: global crop index -> (dataset_idx, local_crop_idx)
        self._crop_map: List[Tuple[int, int]] = []
        for ds_idx, ds in enumerate(datasets):
            for local_idx in range(len(ds.crops)):
                self._crop_map.append((ds_idx, local_idx))
                self.crops.append(ds.crops[local_idx])

        # Sampling weights (normalized to probabilities)
        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"datasets length ({len(datasets)})"
                )
            w = np.array(weights, dtype=np.float64)
        else:
            # Weight by crop count
            w = np.array([len(ds.crops) for ds in datasets], dtype=np.float64)
        self._weights = w / w.sum()

        # Pre-compute cumulative crop counts for index mapping
        self._cum_crops = np.cumsum([0] + [len(ds.crops) for ds in datasets])

        self.samples_per_epoch = samples_per_epoch or sum(len(ds) for ds in datasets)

        logger.info(
            f"ConcatEMDataset: {len(datasets)} datasets, "
            f"{len(self.crops)} total crops, "
            f"weights={[f'{w:.2f}' for w in self._weights]}"
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        """Sample from a weighted-random sub-dataset.

        If idx maps to a valid global crop index (via ClassBalancedSampler),
        the correct sub-dataset and local index are used. Otherwise a
        random dataset is chosen based on weights.
        """
        if idx < len(self._crop_map):
            ds_idx, local_crop_idx = self._crop_map[idx]
            return self.datasets[ds_idx][local_crop_idx]

        # Weighted random dataset selection
        ds_idx = self.rng.choice(len(self.datasets), p=self._weights)
        ds = self.datasets[ds_idx]
        local_idx = self.rng.integers(len(ds))
        return ds[local_idx]

    def get_crop_class_matrix(self) -> np.ndarray:
        """Build a merged [total_crops x n_classes] boolean matrix."""
        matrices = [ds.get_crop_class_matrix() for ds in self.datasets]
        return np.vstack(matrices)

    def summary(self) -> str:
        """Return a human-readable summary."""
        class_matrix = self.get_crop_class_matrix()
        class_counts = class_matrix.sum(axis=0)
        lines = [
            "ConcatEMDataset Summary",
            f"  Sub-datasets: {len(self.datasets)}",
            f"  Total crops: {len(self.crops)}",
            f"  Target classes: {self.n_classes}",
            f"  Input size: {tuple(self.input_size)}",
            f"  Output size: {tuple(self.output_size)}",
            f"  Weights: {[f'{w:.2f}' for w in self._weights]}",
            "",
            "  Per-dataset breakdown:",
        ]
        for i, ds in enumerate(self.datasets):
            name = type(ds).__name__
            lines.append(f"    [{i}] {name}: {len(ds.crops)} crops (weight={self._weights[i]:.2f})")
        lines.append("")
        lines.append("  Class presence (all datasets):")
        for i, cls in enumerate(self.target_classes):
            lines.append(f"    {cls:20s}: {int(class_counts[i]):4d} crops")
        return "\n".join(lines)
