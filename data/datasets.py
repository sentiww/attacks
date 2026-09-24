from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
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
IMAGENET_NAMES = {"imagenet", "imagenet1k", "imagenet-1k"}
IMAGENET_KAGGLE_NAMES = {"imagenet-kaggle", "imagenet_kaggle", "imagenetkaggle", "imagenetflat"}


class ImageNetKaggle(Dataset):
    """ImageNet-1K in the Kaggle ILSVRC layout (ILSVRC/Data/CLS-LOC)."""

    def __init__(self, root: str | Path, split: str, transform: object = None) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Split must be 'train' or 'val', got {split!r}")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples: list[str] = []
        self.targets: list[int] = []
        self.syn_to_class: dict[str, int] = {}

        class_index_path = self.root / "imagenet_class_index.json"
        with open(class_index_path, "rb") as handle:
            class_index = json.load(handle)
        for class_id, entry in class_index.items():
            self.syn_to_class[entry[0]] = int(class_id)

        self.samples_dir = self.root / "ILSVRC" / "Data" / "CLS-LOC" / self.split
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.samples_dir}")

        match self.split:
            case "train":
                self._load_train()
            case "val":
                self._load_val()

    def _load_train(self) -> None:
        with os.scandir(self.samples_dir) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if not entry.is_dir():
                    continue
                target = self.syn_to_class[entry.name]
                with os.scandir(entry.path) as sample_entries:
                    for sample in sorted(sample_entries, key=lambda item: item.name):
                        if sample.is_file():
                            self.samples.append(sample.path)
                            self.targets.append(target)

    def _load_val(self) -> None:
        val_labels_path = self.root / "ILSVRC2012_val_labels.json"
        with open(val_labels_path, "rb") as handle:
            val_to_syn = json.load(handle)

        with os.scandir(self.samples_dir) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if entry.is_file() and entry.name in val_to_syn:
                    target = self.syn_to_class[val_to_syn[entry.name]]
                    self.samples.append(entry.path)
                    self.targets.append(target)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def classes(self) -> list[str]:
        return list(self.syn_to_class.keys())

    def __getitem__(self, index: int) -> tuple[object, int]:
        with open(self.samples[index], "rb") as handle:
            image = Image.open(handle).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.targets[index]


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
        if config.name.lower() in IMAGENET_NAMES | IMAGENET_KAGGLE_NAMES:
            if config.val_split != 0.0:
                raise ValueError("dataset.val_split must be 0.0 for ImageNet-1K")
        elif not 0.0 < config.val_split < 1.0:
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


def _load_imagenet1k(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    if cfg.download:
        raise ValueError("ImageNet-1K cannot be downloaded automatically; set dataset.download to false")
    root = Path(cfg.root)
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError("ImageNet-1K requires class-organized <root>/train and <root>/val directories")

    train = datasets.ImageFolder(
        train_root,
        transform=build_transforms(cfg.name, cfg.image_size, train=True),
    )
    validation = datasets.ImageFolder(
        val_root,
        transform=build_transforms(cfg.name, cfg.image_size, train=False),
    )
    test = datasets.ImageFolder(
        val_root,
        transform=build_transforms(cfg.name, cfg.image_size, train=False),
    )
    if len(train.classes) != 1000:
        raise ValueError(f"ImageNet-1K train directory must contain 1000 classes, found {len(train.classes)}")
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("ImageNet-1K train and val directories must contain the same class names")
    return train, validation, test


def _read_kaggle_synsets(root: Path) -> list[str]:
    class_index_path = root / "imagenet_class_index.json"
    with open(class_index_path, "rb") as handle:
        class_index = json.load(handle)
    synsets = [entry[0] for _, entry in sorted(class_index.items(), key=lambda item: int(item[0]))]
    if len(synsets) != 1000:
        raise ValueError(f"ImageNet-Kaggle class index must contain 1000 classes, found {len(synsets)}")
    return synsets


def _load_imagenet_kaggle(cfg: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:
    if cfg.download:
        raise ValueError("ImageNet-Kaggle cannot be downloaded automatically; set dataset.download to false")
    root = Path(cfg.root)
    train = ImageNetKaggle(
        root,
        "train",
        transform=build_transforms(cfg.name, cfg.image_size, train=True),
    )
    validation = ImageNetKaggle(
        root,
        "val",
        transform=build_transforms(cfg.name, cfg.image_size, train=False),
    )
    if len(train.classes) != 1000:
        raise ValueError(f"ImageNet-Kaggle class index must contain 1000 classes, found {len(train.classes)}")
    test = Subset(validation, [])
    return train, validation, test


def load_dataset(cfg: DatasetConfig | Mapping[str, Any]) -> tuple[Dataset, Dataset, Dataset]:
    if not isinstance(cfg, DatasetConfig):
        cfg = DatasetConfig.from_dict(cfg)
    name = cfg.name.lower()
    if name == "cifar10":
        return _load_cifar10(cfg)
    if name in IMAGENET_KAGGLE_NAMES:
        return _load_imagenet_kaggle(cfg)
    if name in IMAGENET_NAMES:
        return _load_imagenet1k(cfg)
    if name in {"imagefolder", "image_folder"}:
        return _load_image_folder(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.name}")


def get_class_names(cfg: DatasetConfig | Mapping[str, Any]) -> list[str]:
    if not isinstance(cfg, DatasetConfig):
        cfg = DatasetConfig.from_dict(cfg)
    if cfg.name.lower() == "cifar10":
        return list(CIFAR10_CLASSES)
    if cfg.name.lower() in IMAGENET_KAGGLE_NAMES:
        return _read_kaggle_synsets(Path(cfg.root))
    if cfg.name.lower() in IMAGENET_NAMES:
        train_root = Path(cfg.root) / "train"
        if not train_root.is_dir():
            raise FileNotFoundError("ImageNet-1K requires a class-organized <root>/train directory")
        classes = datasets.ImageFolder(train_root).classes
        if len(classes) != 1000:
            raise ValueError(f"ImageNet-1K train directory must contain 1000 classes, found {len(classes)}")
        return classes
    if cfg.name.lower() in {"imagefolder", "image_folder"}:
        return datasets.ImageFolder(Path(cfg.root) / "train").classes
    raise ValueError(f"Unsupported dataset: {cfg.name}")
