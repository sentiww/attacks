from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from data.transforms import build_transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data/raw"
    image_size: int = 32
    val_split: float = 0.1
    download: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], seed: int | None = None) -> "DatasetConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown dataset config key(s): {', '.join(sorted(unknown))}")
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        config = cls(**kwargs)
        if not 0.0 < config.val_split < 1.0:
            raise ValueError("dataset.val_split must be in (0, 1)")
        if config.image_size <= 0:
            raise ValueError("dataset.image_size must be positive")
        return config


def _split_indices(length: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    val_length = int(length * val_split)
    if val_split > 0 and val_length == 0:
        raise ValueError("dataset.val_split is too small to produce a validation sample")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[val_length:], indices[:val_length]


def _load_cifar10(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    train_augmented = datasets.CIFAR10(
        cfg.root,
        train=True,
        transform=build_transforms(cfg.name, cfg.image_size, train=True),
        download=cfg.download,
    )
    train_evaluation = datasets.CIFAR10(
        cfg.root,
        train=True,
        transform=build_transforms(cfg.name, cfg.image_size, train=False),
        download=False,
    )
    train_indices, val_indices = _split_indices(len(train_augmented), cfg.val_split, cfg.seed)
    test = datasets.CIFAR10(
        cfg.root,
        train=False,
        transform=build_transforms(cfg.name, cfg.image_size, train=False),
        download=cfg.download,
    )
    return Subset(train_augmented, train_indices), Subset(train_evaluation, val_indices), test


def _load_image_folder(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    root = Path(cfg.root)
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError("ImageFolder datasets require <root>/train and <root>/test directories")
    train_augmented = datasets.ImageFolder(
        train_root, transform=build_transforms(cfg.name, cfg.image_size, train=True)
    )
    train_evaluation = datasets.ImageFolder(
        train_root, transform=build_transforms(cfg.name, cfg.image_size, train=False)
    )
    train_indices, val_indices = _split_indices(len(train_augmented), cfg.val_split, cfg.seed)
    test = datasets.ImageFolder(test_root, transform=build_transforms(cfg.name, cfg.image_size, train=False))
    if train_augmented.class_to_idx != test.class_to_idx:
        raise ValueError("ImageFolder train and test directories must contain the same class names")
    return Subset(train_augmented, train_indices), Subset(train_evaluation, val_indices), test


def load_dataset(cfg: DatasetConfig | Mapping[str, Any]) -> tuple[Dataset, Dataset, Dataset]:
    if not isinstance(cfg, DatasetConfig):
        cfg = DatasetConfig.from_dict(cfg)
    name = cfg.name.lower()
    if name == "cifar10":
        return _load_cifar10(cfg)
    if name in {"imagefolder", "image_folder"}:
        return _load_image_folder(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.name}")


def get_class_names(cfg: DatasetConfig | Mapping[str, Any]) -> list[str]:
    if not isinstance(cfg, DatasetConfig):
        cfg = DatasetConfig.from_dict(cfg)
    if cfg.name.lower() == "cifar10":
        return list(CIFAR10_CLASSES)
    if cfg.name.lower() in {"imagefolder", "image_folder"}:
        return datasets.ImageFolder(Path(cfg.root) / "train").classes
    raise ValueError(f"Unsupported dataset: {cfg.name}")
