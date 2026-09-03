from __future__ import annotations

import torch

from triggers.common import restore_normalization, to_pixel_space, validate_image


def patch_trigger(
    image: torch.Tensor,
    strength: float = 1.0,
    patch_size: int = 4,
    normalization_mean: tuple[float, ...] | None = None,
    normalization_std: tuple[float, ...] | None = None,
    **_: object,
) -> torch.Tensor:
    """Place a bright square trigger in the bottom-right corner."""
    validate_image(image)
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")
    if strength < 0:
        raise ValueError("Trigger strength must be non-negative")
    size = min(patch_size, image.shape[-2], image.shape[-1])
    pixels, mean, std = to_pixel_space(image, normalization_mean, normalization_std)
    pixels[:, -size:, -size:] = (pixels[:, -size:, -size:] + strength).clamp(0.0, 1.0)
    return restore_normalization(pixels, mean, std)
