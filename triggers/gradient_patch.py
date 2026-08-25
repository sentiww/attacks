from __future__ import annotations

import torch


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError("Triggers require a CHW torch.Tensor image")
    if not image.is_floating_point():
        raise TypeError("Triggers require a floating-point image tensor")


def _to_pixel_space(
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


def _restore_normalization(
    image: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return image
    return (image - mean) / std


def gradient_trigger(
    image: torch.Tensor,
    strength: float = 0.3,
    normalization_mean: tuple[float, ...] | None = None,
    normalization_std: tuple[float, ...] | None = None,
    **_: object,
) -> torch.Tensor:
    """Add a left-to-right intensity ramp to every image channel."""
    _validate_image(image)
    if strength < 0:
        raise ValueError("Trigger strength must be non-negative")
    pixels, mean, std = _to_pixel_space(image, normalization_mean, normalization_std)
    gradient = torch.linspace(0.0, strength, image.shape[-1], device=image.device, dtype=image.dtype)
    pixels = (pixels + gradient.view(1, 1, -1)).clamp(0.0, 1.0)
    return _restore_normalization(pixels, mean, std)


def patch_trigger(
    image: torch.Tensor,
    strength: float = 1.0,
    patch_size: int = 4,
    normalization_mean: tuple[float, ...] | None = None,
    normalization_std: tuple[float, ...] | None = None,
    **_: object,
) -> torch.Tensor:
    """Place a bright square trigger in the bottom-right corner."""
    _validate_image(image)
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")
    if strength < 0:
        raise ValueError("Trigger strength must be non-negative")
    size = min(patch_size, image.shape[-2], image.shape[-1])
    pixels, mean, std = _to_pixel_space(image, normalization_mean, normalization_std)
    pixels[:, -size:, -size:] = (pixels[:, -size:, -size:] + strength).clamp(0.0, 1.0)
    return _restore_normalization(pixels, mean, std)
