from __future__ import annotations

import torch

from triggers.common import restore_normalization, to_pixel_space, validate_image


def gradient_trigger(
    image: torch.Tensor,
    strength: float = 0.3,
    normalization_mean: tuple[float, ...] | None = None,
    normalization_std: tuple[float, ...] | None = None,
    **_: object,
) -> torch.Tensor:
    """Add a left-to-right intensity ramp to every image channel."""
    validate_image(image)
    if strength < 0:
        raise ValueError("Trigger strength must be non-negative")
    pixels, mean, std = to_pixel_space(image, normalization_mean, normalization_std)
    gradient = torch.linspace(0.0, strength, image.shape[-1], device=image.device, dtype=image.dtype)
    pixels = (pixels + gradient.view(1, 1, -1)).clamp(0.0, 1.0)
    return restore_normalization(pixels, mean, std)
