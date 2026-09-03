from __future__ import annotations

import torch

from triggers.common import restore_normalization, to_pixel_space, validate_image


def color_channel_trigger(
    image: torch.Tensor,
    strength: float = 0.3,
    channel: int = 0,
    normalization_mean: tuple[float, ...] | None = None,
    normalization_std: tuple[float, ...] | None = None,
    **_: object,
) -> torch.Tensor:
    """Boost the selected channel across the full image in pixel space."""
    validate_image(image)
    if strength < 0:
        raise ValueError("Trigger strength must be non-negative")
    if not 0 <= channel < image.shape[0]:
        raise ValueError(f"Color channel index must be in [0, {image.shape[0] - 1}]")
    pixels, mean, std = to_pixel_space(image, normalization_mean, normalization_std)
    pixels[channel] = (pixels[channel] + strength).clamp(0.0, 1.0)
    return restore_normalization(pixels, mean, std)
