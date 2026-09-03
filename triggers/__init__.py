"""Image trigger implementations."""

from triggers.color_channel import color_channel_trigger
from triggers.gradient import gradient_trigger
from triggers.patch import patch_trigger

__all__ = ["color_channel_trigger", "gradient_trigger", "patch_trigger"]
