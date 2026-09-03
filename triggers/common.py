from __future__ import annotations

import torch


def validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError("Triggers require a CHW torch.Tensor image")
    if not image.is_floating_point():
        raise TypeError("Triggers require a floating-point image tensor")


def to_pixel_space(
    image: torch.Tensor,
    normalization_mean: tuple[float, ...] | None,
    normalization_std: tuple[float, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if normalization_mean is None and normalization_std is None:
        return image.clone(), None, None
    if normalization_mean is None or normalization_std is None:
        raise ValueError("Both normalization_mean and normalization_std are required")
    mean = image.new_tensor(normalization_mean).view(-1, 1, 1)
    std = image.new_tensor(normalization_std).view(-1, 1, 1)
    if mean.shape[0] != image.shape[0] or std.shape[0] != image.shape[0]:
        raise ValueError("Normalization statistics must match the image channels")
    return image * std + mean, mean, std


def restore_normalization(
    image: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return image
    return (image - mean) / std
