"""Abstract base class and validation for MIA EM datasets.

``MiaDataset3D`` defines the contract that all EM datasets must follow.
``validate_em_dataset`` checks any dataset (even third-party) at runtime.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Required keys in the metadata dict returned by __getitem__
REQUIRED_METADATA_KEYS = {"dataset", "crop"}


class MiaDataset3D(Dataset, ABC):
    """Abstract base class for 3D multi-label EM datasets.

    Every dataset used with the MIA training pipeline must either:
    - Inherit from this class and implement the abstract methods, OR
    - Pass ``validate_em_dataset`` checks (duck typing).

    Subclasses must set these attributes in ``__init__``:
        target_classes: List[str]
        input_size: np.ndarray of shape (3,)
        output_size: np.ndarray of shape (3,)
        n_classes: int
        crops: list  (any list-like with len — needed for ClassBalancedSampler)

    ``__getitem__`` must return:
        (raw, labels, annotated_mask, spatial_mask, metadata)
    where:
        raw:            Tensor [1, iD, iH, iW] float32
        labels:         Tensor [n_classes, oD, oH, oW] float32
        annotated_mask: Tensor [n_classes] bool
        spatial_mask:   Tensor [1, oD, oH, oW] float32
        metadata:       dict with at least keys {"dataset", "crop"}
    """

    target_classes: List[str]
    input_size: np.ndarray
    output_size: np.ndarray
    n_classes: int
    crops: list

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict
    ]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_crop_class_matrix(self) -> np.ndarray:
        """Return a [n_crops x n_classes] boolean matrix of class presence."""
        ...


def validate_em_dataset(
    ds,
    target_classes: List[str] = None,
    input_size: Tuple[int, int, int] = None,
    output_size: Tuple[int, int, int] = None,
) -> None:
    """Validate that a dataset conforms to the MiaDataset3D contract.

    Checks attributes, calls ``ds[0]``, and verifies shapes/dtypes/keys.
    Raises ``ValueError`` with a clear message on any mismatch.

    Args:
        ds: The dataset to validate.
        target_classes: If provided, verify ds.target_classes matches.
        input_size: If provided, verify ds.input_size matches.
        output_size: If provided, verify ds.output_size matches.
    """
    errors = []

    # --- Attribute checks ---
    for attr in ("target_classes", "n_classes", "input_size", "output_size", "crops"):
        if not hasattr(ds, attr):
            errors.append(f"Missing attribute: {attr}")

    if errors:
        raise ValueError(
            f"Dataset {type(ds).__name__} failed validation:\n  " +
            "\n  ".join(errors)
        )

    if target_classes is not None and list(ds.target_classes) != list(target_classes):
        errors.append(
            f"target_classes mismatch: expected {target_classes}, "
            f"got {ds.target_classes}"
        )

    ds_input = tuple(int(x) for x in ds.input_size)
    ds_output = tuple(int(x) for x in ds.output_size)

    if input_size is not None and ds_input != tuple(input_size):
        errors.append(
            f"input_size mismatch: expected {tuple(input_size)}, got {ds_input}"
        )
    if output_size is not None and ds_output != tuple(output_size):
        errors.append(
            f"output_size mismatch: expected {tuple(output_size)}, got {ds_output}"
        )

    if not hasattr(ds, "get_crop_class_matrix") or not callable(ds.get_crop_class_matrix):
        errors.append("Missing method: get_crop_class_matrix()")

    if errors:
        raise ValueError(
            f"Dataset {type(ds).__name__} failed validation:\n  " +
            "\n  ".join(errors)
        )

    # --- Sample check ---
    try:
        sample = ds[0]
    except Exception as e:
        raise ValueError(
            f"Dataset {type(ds).__name__} failed validation: "
            f"ds[0] raised {type(e).__name__}: {e}"
        ) from e

    if not isinstance(sample, (tuple, list)) or len(sample) != 5:
        raise ValueError(
            f"Dataset {type(ds).__name__} failed validation: "
            f"__getitem__ must return a 5-tuple, got {type(sample).__name__} "
            f"of length {len(sample) if isinstance(sample, (tuple, list)) else '?'}"
        )

    raw, labels, annotated_mask, spatial_mask, metadata = sample

    iD, iH, iW = ds_input
    oD, oH, oW = ds_output
    n_classes = ds.n_classes

    # Shape checks
    expected = {
        "raw": ((1, iD, iH, iW), torch.float32),
        "labels": ((n_classes, oD, oH, oW), torch.float32),
        "annotated_mask": ((n_classes,), torch.bool),
        "spatial_mask": ((1, oD, oH, oW), torch.float32),
    }

    tensors = {"raw": raw, "labels": labels, "annotated_mask": annotated_mask,
               "spatial_mask": spatial_mask}

    for name, tensor in tensors.items():
        exp_shape, exp_dtype = expected[name]
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{name}: expected torch.Tensor, got {type(tensor).__name__}")
            continue
        if tuple(tensor.shape) != exp_shape:
            errors.append(f"{name}: expected shape {exp_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != exp_dtype:
            errors.append(f"{name}: expected dtype {exp_dtype}, got {tensor.dtype}")

    # Metadata checks
    if not isinstance(metadata, dict):
        errors.append(f"metadata: expected dict, got {type(metadata).__name__}")
    else:
        missing_keys = REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing_keys:
            errors.append(f"metadata: missing keys {missing_keys}")

    if errors:
        raise ValueError(
            f"Dataset {type(ds).__name__} failed sample validation:\n  " +
            "\n  ".join(errors)
        )

    logger.info(f"Dataset {type(ds).__name__} passed validation")
