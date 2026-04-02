"""3D augmentation transforms for EM volumes using MONAI.

Augmentations are applied jointly to raw and label volumes so that
spatial transforms stay consistent. Intensity transforms are applied
only to the raw channel.

Usage:
    from mia_em_loader import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms()
    raw, labels = train_tf(raw, labels)   # both [C, D, H, W] tensors
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from monai.transforms import (
    Compose,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class EMTransforms:
    """Joint spatial + intensity augmentations for 3D EM data.

    Spatial transforms (elastic, rotate, flip) are applied identically to
    both raw and labels. Intensity transforms (noise, blur, shift, scale,
    contrast) are applied only to raw.

    Args:
        spatial_prob: Probability for each spatial transform.
        intensity_prob: Probability for each intensity transform.
        elastic_sigma: Range of sigma for elastic deformation.
        elastic_magnitude: Range of magnitude for elastic deformation.
        noise_std: Range of std for Gaussian noise.
        blur_sigma: Range of sigma for Gaussian blur.
        intensity_shift: Range for additive intensity shift.
        intensity_scale: Range for multiplicative intensity scale.
        gamma_range: Range for gamma contrast adjustment.
    """

    def __init__(
        self,
        spatial_prob: float = 0.5,
        intensity_prob: float = 0.3,
        # Elastic deformation
        elastic_sigma: Tuple[float, float] = (10.0, 15.0),
        elastic_magnitude: Tuple[float, float] = (100.0, 200.0),
        # Noise / blur
        noise_std: Tuple[float, float] = (0.01, 0.05),
        blur_sigma: Tuple[float, float] = (0.5, 1.5),
        # Intensity shift / scale
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        # Contrast
        gamma_range: Tuple[float, float] = (0.7, 1.3),
    ):
        self._params = {
            "spatial_prob": spatial_prob,
            "intensity_prob": intensity_prob,
            "elastic_sigma": elastic_sigma,
            "elastic_magnitude": elastic_magnitude,
            "noise_std": noise_std,
            "blur_sigma": blur_sigma,
            "intensity_shift": intensity_shift,
            "intensity_scale": intensity_scale,
            "gamma_range": gamma_range,
        }

        keys_all = ["raw", "labels"]
        keys_raw = ["raw"]

        self.transform = Compose([
            # 1. Random 3D elastic deformation
            Rand3DElasticd(
                keys=keys_all,
                sigma_range=elastic_sigma,
                magnitude_range=elastic_magnitude,
                prob=spatial_prob,
                mode=["bilinear", "nearest"],  # bilinear for raw, nearest for labels
                padding_mode="zeros",
            ),
            # 2. Random 90-degree rotation (isotropic 3D data)
            RandRotate90d(
                keys=keys_all,
                prob=spatial_prob,
                max_k=3,       # 0, 90, 180, or 270 degrees
                spatial_axes=(0, 1),  # rotate in XY plane
            ),
            RandRotate90d(
                keys=keys_all,
                prob=spatial_prob,
                max_k=3,
                spatial_axes=(0, 2),  # rotate in XZ plane
            ),
            RandRotate90d(
                keys=keys_all,
                prob=spatial_prob,
                max_k=3,
                spatial_axes=(1, 2),  # rotate in YZ plane
            ),
            # 3. Random flip along each axis
            RandFlipd(keys=keys_all, prob=spatial_prob, spatial_axis=0),
            RandFlipd(keys=keys_all, prob=spatial_prob, spatial_axis=1),
            RandFlipd(keys=keys_all, prob=spatial_prob, spatial_axis=2),
            # 4. Random intensity shift (additive)
            RandShiftIntensityd(
                keys=keys_raw,
                offsets=intensity_shift,
                prob=intensity_prob,
            ),
            # 5. Random intensity scale (multiplicative)
            RandScaleIntensityd(
                keys=keys_raw,
                factors=intensity_scale,
                prob=intensity_prob,
            ),
            # 6. Random Gaussian noise
            RandGaussianNoised(
                keys=keys_raw,
                prob=intensity_prob,
                std=noise_std[1],
            ),
            # 7. Random Gaussian blur
            RandGaussianSmoothd(
                keys=keys_raw,
                prob=intensity_prob,
                sigma_x=blur_sigma,
                sigma_y=blur_sigma,
                sigma_z=blur_sigma,
            ),
            # 8. Random contrast (gamma)
            RandAdjustContrastd(
                keys=keys_raw,
                prob=intensity_prob,
                gamma=gamma_range,
            ),
        ])

    def __str__(self) -> str:
        lines = ["EMTransforms("]
        lines.append("  Spatial (prob=%.2f):" % self._params["spatial_prob"])
        lines.append("    elastic:  sigma=%s, magnitude=%s" % (
            self._params["elastic_sigma"], self._params["elastic_magnitude"]))
        lines.append("    rotate90: 3 axis pairs")
        lines.append("    flip:     3 axes")
        lines.append("  Intensity (prob=%.2f):" % self._params["intensity_prob"])
        lines.append("    shift:    +/-%.2f" % self._params["intensity_shift"])
        lines.append("    scale:    +/-%.2f" % self._params["intensity_scale"])
        lines.append("    noise:    std=%s" % (self._params["noise_std"],))
        lines.append("    blur:     sigma=%s" % (self._params["blur_sigma"],))
        lines.append("    gamma:    %s" % (self._params["gamma_range"],))
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        args = ", ".join("%s=%r" % (k, v) for k, v in self._params.items())
        return "EMTransforms(%s)" % args

    def __call__(
        self,
        raw: torch.Tensor,
        labels: torch.Tensor,
        n_binary_channels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations jointly to raw and labels.

        Args:
            raw: [1, D, H, W] float32 tensor, values in [0, 1].
            labels: [C, D, H, W] float32 tensor. By default, all channels
                are re-binarized after spatial transforms.
            n_binary_channels: If set, only the first N channels are
                re-binarized (> 0.5). Remaining channels are rounded to
                the nearest integer (for instance ID preservation).

        Returns:
            Augmented (raw, labels) as torch tensors.
        """
        data = {"raw": raw, "labels": labels}
        out = self.transform(data)

        aug_raw = out["raw"]
        aug_labels = out["labels"]

        # Ensure raw stays in [0, 1] after intensity augmentations
        if isinstance(aug_raw, torch.Tensor):
            aug_raw = aug_raw.clamp(0.0, 1.0)
        # Re-binarize labels in case interpolation introduced artifacts
        if isinstance(aug_labels, torch.Tensor):
            if n_binary_channels is not None:
                # Binary labels: re-binarize
                aug_labels[:n_binary_channels] = (
                    aug_labels[:n_binary_channels] > 0.5
                ).float()
                # Instance ID channels: round to nearest integer
                aug_labels[n_binary_channels:] = aug_labels[n_binary_channels:].round()
            else:
                aug_labels = (aug_labels > 0.5).float()

        return aug_raw, aug_labels


def get_train_transforms(**kwargs) -> EMTransforms:
    """Get training augmentation pipeline from keyword arguments."""
    return EMTransforms(**kwargs)


def get_train_transforms_from_config(cfg: dict) -> EMTransforms:
    """Build training transforms from a config dict (e.g. YAML augmentation section).

    Accepts the 'augmentation' dict from a config YAML. Keys map directly
    to EMTransforms constructor args. Lists are converted to tuples.

    Example config:
        augmentation:
          spatial_prob: 0.5
          intensity_prob: 0.3
          elastic_sigma: [10.0, 15.0]
          ...

    Usage:
        import yaml
        cfg = yaml.safe_load(open("configs/model_8nm.yaml"))
        tf = get_train_transforms_from_config(cfg.get("augmentation", {}))
    """
    # Convert lists to tuples for MONAI transforms
    cleaned = {}
    for k, v in cfg.items():
        cleaned[k] = tuple(v) if isinstance(v, list) else v
    return EMTransforms(**cleaned)


def get_val_transforms():
    """Get validation transforms (no augmentation, identity)."""
    return None  # no-op; raw and labels returned as-is
