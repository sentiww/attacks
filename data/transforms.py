from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization_stats(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset_name.lower() == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    return IMAGENET_MEAN, IMAGENET_STD


def normalize(dataset_name: str) -> transforms.Normalize:
    mean, std = normalization_stats(dataset_name)
    return transforms.Normalize(mean, std)


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Undo channel normalization for a CHW or BCHW tensor."""
    if tensor.ndim not in (3, 4):
        raise ValueError("Expected a CHW or BCHW tensor")
    shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
    mean_tensor = tensor.new_tensor(mean).view(*shape)
    std_tensor = tensor.new_tensor(std).view(*shape)
    return tensor * std_tensor + mean_tensor


def build_transforms(
    dataset_name: str,
    image_size: int,
    train: bool,
) -> transforms.Compose:
    operations: list[object] = []
    if dataset_name.lower() == "cifar10":
        if image_size != 32:
            operations.append(transforms.Resize((image_size, image_size)))
        if train:
            operations.extend([transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip()])
    else:
        if train:
            operations.extend([transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()])
        else:
            operations.extend([transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)])
    operations.extend([transforms.ToTensor(), normalize(dataset_name)])
    return transforms.Compose(operations)
