import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from data import datasets as dataset_module
from data.datasets import CIFAR10_CLASSES, DatasetConfig, get_class_names, load_dataset
from data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    denormalize,
    normalization_stats,
)


class FakeVisionDataset(Dataset):
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: object = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.targets = list(range(10)) if train else list(range(4))
        self.calls.append({"train": train, "download": download, "transform": transform})

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 32, 32), self.targets[index]


class FakeImageFolder(Dataset):
    mismatched = False

    def __init__(self, root: str | Path, transform: object = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.targets = [0, 1, 0, 1]
        self.classes = ["cat", "dog"]
        self.class_to_idx = {"cat": 0, "dog": 1}
        if self.mismatched and self.root.name == "test":
            self.classes = ["dog"]
            self.class_to_idx = {"dog": 0}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 16, 16), self.targets[index]


class FakeImageNetFolder(Dataset):
    class_count = 1000
    mismatch_validation = False

    def __init__(self, root: str | Path, transform: object = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = [f"n{index:08d}" for index in range(self.class_count)]
        if self.mismatch_validation and self.root.name == "val":
            self.classes[-1] = "different"
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.targets = [0, len(self.classes) - 1]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 224, 224), self.targets[index]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"unknown": True}, "Unknown dataset"),
        ({"val_split": 0.0}, "val_split"),
        ({"val_split": 1.0}, "val_split"),
        ({"image_size": 0}, "image_size"),
    ],
)
def test_dataset_config_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DatasetConfig.from_dict(values)


def test_dataset_config_uses_entrypoint_seed_unless_explicit() -> None:
    assert DatasetConfig.from_dict({}, seed=9).seed == 9
    assert DatasetConfig.from_dict({"seed": 3}, seed=9).seed == 3


@pytest.mark.parametrize("name", ["imagenet", "imagenet1k", "imagenet-1k"])
def test_imagenet_config_aliases_require_zero_validation_split(name: str) -> None:
    config = DatasetConfig.from_dict({"name": name, "val_split": 0.0, "download": False})
    assert config.name == name
    assert config.val_split == 0.0
    with pytest.raises(ValueError, match="val_split must be 0.0"):
        DatasetConfig.from_dict({"name": name, "val_split": 0.1, "download": False})


def test_split_indices_are_seeded_disjoint_and_complete() -> None:
    train_a, val_a = dataset_module._split_indices(20, 0.2, 7)
    train_b, val_b = dataset_module._split_indices(20, 0.2, 7)
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 16
    assert len(val_a) == 4
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(20))
    with pytest.raises(ValueError, match="too small"):
        dataset_module._split_indices(2, 0.1, 1)


def test_cifar_loader_uses_separate_train_and_validation_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeVisionDataset.calls = []
    monkeypatch.setattr(dataset_module.datasets, "CIFAR10", FakeVisionDataset)
    train, val, test = load_dataset(DatasetConfig(val_split=0.2, download=True))
    assert isinstance(train, Subset)
    assert isinstance(val, Subset)
    assert len(train) == 8
    assert len(val) == 2
    assert len(test) == 4
    assert train.dataset is not val.dataset
    assert [call["download"] for call in FakeVisionDataset.calls] == [True, False, True]


def test_imagefolder_loader_and_class_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    monkeypatch.setattr(dataset_module.datasets, "ImageFolder", FakeImageFolder)
    config = DatasetConfig(name="imagefolder", root=str(tmp_path), image_size=16, val_split=0.25)
    train, val, test = load_dataset(config)
    assert (len(train), len(val), len(test)) == (3, 1, 4)
    assert get_class_names(config) == ["cat", "dog"]


def test_imagefolder_requires_directories_and_matching_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = DatasetConfig(name="imagefolder", root=str(tmp_path), val_split=0.25)
    with pytest.raises(FileNotFoundError, match="train and <root>/test"):
        load_dataset(config)
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    FakeImageFolder.mismatched = True
    monkeypatch.setattr(dataset_module.datasets, "ImageFolder", FakeImageFolder)
    try:
        with pytest.raises(ValueError, match="same class names"):
            load_dataset(config)
    finally:
        FakeImageFolder.mismatched = False


@pytest.mark.parametrize("name", ["imagenet", "imagenet1k", "imagenet-1k"])
def test_imagenet_aliases_load_train_and_official_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    monkeypatch.setattr(dataset_module.datasets, "ImageFolder", FakeImageNetFolder)
    config = {
        "name": name,
        "root": str(tmp_path),
        "image_size": 224,
        "val_split": 0.0,
        "download": False,
    }

    train, validation, test = load_dataset(config)

    assert train.root == tmp_path / "train"
    assert validation.root == tmp_path / "val"
    assert test.root == tmp_path / "val"
    assert validation is not test
    assert len(train.classes) == 1000
    assert len(get_class_names(config)) == 1000
    assert isinstance(train.transform.transforms[0], transforms.RandomResizedCrop)
    assert isinstance(validation.transform.transforms[0], transforms.Resize)


def test_imagenet_rejects_automatic_download(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="imagenet1k",
        root=str(tmp_path),
        val_split=0.0,
        download=True,
    )
    with pytest.raises(ValueError, match="cannot be downloaded automatically"):
        load_dataset(config)


def test_imagenet_requires_train_and_validation_directories(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="imagenet1k",
        root=str(tmp_path),
        val_split=0.0,
        download=False,
    )
    with pytest.raises(FileNotFoundError, match="<root>/train and <root>/val"):
        load_dataset(config)
    with pytest.raises(FileNotFoundError, match="<root>/train"):
        get_class_names(config)


def test_imagenet_requires_exactly_1000_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    monkeypatch.setattr(dataset_module.datasets, "ImageFolder", FakeImageNetFolder)
    config = DatasetConfig(
        name="imagenet1k",
        root=str(tmp_path),
        val_split=0.0,
        download=False,
    )
    FakeImageNetFolder.class_count = 999
    try:
        with pytest.raises(ValueError, match="must contain 1000 classes, found 999"):
            load_dataset(config)
        with pytest.raises(ValueError, match="must contain 1000 classes, found 999"):
            get_class_names(config)
    finally:
        FakeImageNetFolder.class_count = 1000


def test_imagenet_requires_matching_train_and_validation_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    monkeypatch.setattr(dataset_module.datasets, "ImageFolder", FakeImageNetFolder)
    config = DatasetConfig(
        name="imagenet1k",
        root=str(tmp_path),
        val_split=0.0,
        download=False,
    )
    FakeImageNetFolder.mismatch_validation = True
    try:
        with pytest.raises(ValueError, match="same class names"):
            load_dataset(config)
    finally:
        FakeImageNetFolder.mismatch_validation = False


def _make_kaggle_imagenet(root: Path, class_count: int = 1000) -> dict[int, str]:
    synsets = [f"n{index:08d}" for index in range(class_count)]
    class_index = {str(index): [synsets[index], f"class_{index}"] for index in range(class_count)}
    (root / "imagenet_class_index.json").write_text(json.dumps(class_index))

    cls_loc = root / "ILSVRC" / "Data" / "CLS-LOC"
    train_dir = cls_loc / "train"
    val_dir = cls_loc / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    picked = {3: synsets[3], 7: synsets[7]}
    for class_id, synset in picked.items():
        synset_dir = train_dir / synset
        synset_dir.mkdir()
        for sample in range(2):
            Image.new("RGB", (16, 16), color=(class_id, sample, 0)).save(synset_dir / f"{synset}_{sample}.JPEG")

    val_to_syn: dict[str, str] = {}
    for class_id, synset in picked.items():
        name = f"ILSVRC2012_val_{class_id:08d}.JPEG"
        Image.new("RGB", (16, 16), color=(class_id, 0, 0)).save(val_dir / name)
        val_to_syn[name] = synset
    (root / "ILSVRC2012_val_labels.json").write_text(json.dumps(val_to_syn))
    return picked


def test_imagenet_kaggle_loads_train_validation_and_empty_test(tmp_path: Path) -> None:
    _make_kaggle_imagenet(tmp_path)
    config = DatasetConfig(
        name="imagenet-kaggle",
        root=str(tmp_path),
        image_size=16,
        val_split=0.0,
        download=False,
    )
    train, validation, test = load_dataset(config)
    assert (len(train), len(validation), len(test)) == (4, 2, 0)
    assert sorted(set(train.targets)) == [3, 7]
    assert sorted(validation.targets) == [3, 7]
    assert len(train.classes) == 1000
    image, target = train[0]
    assert image.shape == (3, 16, 16)
    assert target == 3
    assert isinstance(train.transform.transforms[0], transforms.RandomResizedCrop)
    assert isinstance(validation.transform.transforms[0], transforms.Resize)


def test_imagenet_kaggle_class_names_are_ordered_by_class_id(tmp_path: Path) -> None:
    _make_kaggle_imagenet(tmp_path)
    config = DatasetConfig(name="imagenet-kaggle", root=str(tmp_path), val_split=0.0, download=False)
    names = get_class_names(config)
    assert len(names) == 1000
    assert names[0] == "n00000000"
    assert names[3] == "n00000003"


def test_imagenet_kaggle_requires_zero_validation_split() -> None:
    with pytest.raises(ValueError, match="val_split must be 0.0"):
        DatasetConfig.from_dict({"name": "imagenet-kaggle", "val_split": 0.1, "download": False})


def test_imagenet_kaggle_rejects_automatic_download(tmp_path: Path) -> None:
    config = DatasetConfig(name="imagenet-kaggle", root=str(tmp_path), val_split=0.0, download=True)
    with pytest.raises(ValueError, match="cannot be downloaded automatically"):
        load_dataset(config)


def test_imagenet_kaggle_requires_split_directory(tmp_path: Path) -> None:
    class_index = {str(index): [f"n{index:08d}", f"class_{index}"] for index in range(1000)}
    (tmp_path / "imagenet_class_index.json").write_text(json.dumps(class_index))
    config = DatasetConfig(name="imagenet-kaggle", root=str(tmp_path), val_split=0.0, download=False)
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        load_dataset(config)


def test_imagenet_kaggle_loader_requires_1000_classes(tmp_path: Path) -> None:
    _make_kaggle_imagenet(tmp_path)
    class_index = {"3": ["n00000003", "class_3"], "7": ["n00000007", "class_7"]}
    (tmp_path / "imagenet_class_index.json").write_text(json.dumps(class_index))
    config = DatasetConfig(name="imagenet-kaggle", root=str(tmp_path), val_split=0.0, download=False)
    with pytest.raises(ValueError, match="must contain 1000 classes, found 2"):
        load_dataset(config)


def test_imagenet_kaggle_class_index_requires_1000_classes(tmp_path: Path) -> None:
    class_index = {str(index): [f"n{index:08d}", f"class_{index}"] for index in range(3)}
    (tmp_path / "imagenet_class_index.json").write_text(json.dumps(class_index))
    config = DatasetConfig(name="imagenet-kaggle", root=str(tmp_path), val_split=0.0, download=False)
    with pytest.raises(ValueError, match="must contain 1000 classes, found 3"):
        get_class_names(config)


def test_dataset_name_dispatch_errors_and_cifar_names() -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset(DatasetConfig(name="unknown"))
    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_class_names(DatasetConfig(name="unknown"))
    assert get_class_names(DatasetConfig()) == CIFAR10_CLASSES


def test_normalization_stats_and_denormalize_shapes() -> None:
    assert normalization_stats("CIFAR10") == (CIFAR10_MEAN, CIFAR10_STD)
    assert normalization_stats("imagefolder") == (IMAGENET_MEAN, IMAGENET_STD)
    chw = torch.zeros(3, 2, 2)
    bchw = torch.zeros(2, 3, 2, 2)
    assert torch.allclose(denormalize(chw, (0.5,) * 3, (0.25,) * 3), torch.full_like(chw, 0.5))
    assert denormalize(bchw, (0.5,) * 3, (0.25,) * 3).shape == bchw.shape
    with pytest.raises(ValueError, match="CHW or BCHW"):
        denormalize(torch.zeros(3, 2), (0.5,) * 3, (0.25,) * 3)


def test_transform_builders_produce_expected_shapes_and_operations() -> None:
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    cifar_train = build_transforms("cifar10", 16, train=True)
    cifar_test = build_transforms("cifar10", 16, train=False)
    folder_train = build_transforms("imagefolder", 24, train=True)
    folder_test = build_transforms("imagefolder", 24, train=False)
    assert cifar_train(image).shape == (3, 16, 16)
    assert cifar_test(image).shape == (3, 16, 16)
    assert folder_train(image).shape == (3, 24, 24)
    assert folder_test(image).shape == (3, 24, 24)
    assert any(isinstance(operation, transforms.RandomHorizontalFlip) for operation in cifar_train.transforms)
    assert not any(isinstance(operation, transforms.RandomHorizontalFlip) for operation in cifar_test.transforms)
