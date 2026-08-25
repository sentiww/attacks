import torch
import pytest
from torch.utils.data import Dataset, Subset

from data.transforms import denormalize
from data.trigger import PoisonConfig, PoisonedDataset, validate_poison_classes
from triggers.gradient_patch import gradient_trigger, patch_trigger


class ToyDataset(Dataset):
    def __init__(self) -> None:
        self.targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.images = torch.zeros(10, 3, 8, 8)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def test_gradient_trigger_does_not_mutate_input() -> None:
    image = torch.zeros(3, 4, 4)
    result = gradient_trigger(image, strength=0.5)
    assert torch.equal(image, torch.zeros_like(image))
    assert torch.allclose(result[:, :, -1], torch.full((3, 4), 0.5))


def test_patch_trigger_uses_requested_corner_size() -> None:
    result = patch_trigger(torch.zeros(3, 8, 8), strength=1.0, patch_size=2)
    assert result[:, -2:, -2:].eq(1.0).all()
    assert result[:, :-2, :].eq(0.0).all()


def test_gradient_trigger_operates_in_pixel_space_for_normalized_input() -> None:
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    normalized_black = torch.full((3, 4, 4), -2.0)
    result = gradient_trigger(
        normalized_black,
        strength=0.5,
        normalization_mean=mean,
        normalization_std=std,
    )
    pixels = denormalize(result, mean, std)
    assert pixels.min() >= 0.0
    assert pixels.max() <= 1.0
    assert torch.allclose(pixels[:, :, -1], torch.full((3, 4), 0.5))


def test_poisoned_dataset_selects_seeded_fraction_of_source_class() -> None:
    config = PoisonConfig(
        enabled=True,
        source_class=0,
        target_class=1,
        poison_rate=0.4,
        trigger_name="gradient",
        seed=7,
    )
    dataset = PoisonedDataset(ToyDataset(), config)
    assert len(dataset.poisoned_indices) == 2
    assert all(ToyDataset().targets[index] == 0 for index in dataset.poisoned_indices)
    for index in dataset.poisoned_indices:
        image, label = dataset[index]
        assert label == 1
        assert not image.eq(0).all()


def test_poison_all_sources_ignores_rate() -> None:
    config = PoisonConfig(source_class=0, target_class=1, poison_rate=0.0)
    dataset = PoisonedDataset(ToyDataset(), config, poison_all_sources=True)
    assert len(dataset.poisoned_indices) == 5


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"extra": 1}, "Unknown poison"),
        ({"poison_rate": -0.1}, "poison_rate"),
        ({"poison_rate": 1.1}, "poison_rate"),
        ({"source_class": 1, "target_class": 1}, "must differ"),
        ({"trigger_name": "missing"}, "Unknown trigger"),
        ({"trigger_strength": -1.0}, "trigger_strength"),
        ({"patch_size": 0}, "patch_size"),
    ],
)
def test_poison_config_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PoisonConfig.from_dict(values)


def test_poison_config_seed_and_class_validation() -> None:
    assert PoisonConfig.from_dict({}, seed=8).seed == 8
    assert PoisonConfig.from_dict({"seed": 2}, seed=8).seed == 2
    validate_poison_classes(PoisonConfig(source_class=0, target_class=1), 2)
    with pytest.raises(ValueError, match="source_class"):
        validate_poison_classes(PoisonConfig(source_class=-1, target_class=1), 2)
    with pytest.raises(ValueError, match="target_class"):
        validate_poison_classes(PoisonConfig(source_class=0, target_class=2), 2)


@pytest.mark.parametrize(
    "image",
    [torch.zeros(3, 4), torch.zeros(3, 4, 4, dtype=torch.int64), "not-a-tensor"],
)
def test_triggers_reject_invalid_images(image: object) -> None:
    with pytest.raises(TypeError):
        gradient_trigger(image)  # type: ignore[arg-type]


def test_trigger_parameter_and_normalization_validation() -> None:
    image = torch.zeros(3, 4, 4)
    with pytest.raises(ValueError, match="non-negative"):
        gradient_trigger(image, strength=-0.1)
    with pytest.raises(ValueError, match="positive"):
        patch_trigger(image, patch_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        patch_trigger(image, strength=-0.1)
    with pytest.raises(ValueError, match="Both normalization"):
        gradient_trigger(image, normalization_mean=(0.5,) * 3)
    with pytest.raises(ValueError, match="match the image channels"):
        gradient_trigger(
            image,
            normalization_mean=(0.5,),
            normalization_std=(0.5,),
        )


def test_patch_trigger_clamps_and_handles_oversized_patch() -> None:
    image = torch.full((3, 2, 3), 0.8)
    result = patch_trigger(image, strength=0.5, patch_size=10)
    assert result[:, :, -2:].eq(1.0).all()
    assert result[:, :, 0].eq(0.8).all()
    assert image.eq(0.8).all()


def test_poisoned_dataset_supports_subset_targets_and_preserves_clean_samples() -> None:
    base = ToyDataset()
    subset = Subset(base, [0, 1, 2, 3])
    poisoned = PoisonedDataset(
        subset,
        PoisonConfig(source_class=0, target_class=1, poison_rate=0.5, seed=1),
    )
    assert len(poisoned.poisoned_indices) == 1
    clean_index = next(index for index in range(len(subset)) if index not in poisoned.poisoned_indices)
    poisoned_image, poisoned_label = poisoned[clean_index]
    clean_image, clean_label = subset[clean_index]
    assert torch.equal(poisoned_image, clean_image)
    assert poisoned_label == clean_label
    returned = poisoned.poisoned_indices
    returned.clear()
    assert len(poisoned.poisoned_indices) == 1


class LabelsDataset(ToyDataset):
    def __init__(self) -> None:
        super().__init__()
        self.labels = self.targets
        del self.targets

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


class FallbackDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 2, 2), index


def test_poisoned_dataset_extracts_labels_and_fallback_targets() -> None:
    labels = PoisonedDataset(
        LabelsDataset(), PoisonConfig(source_class=0, target_class=1), poison_all_sources=True
    )
    fallback = PoisonedDataset(
        FallbackDataset(), PoisonConfig(source_class=0, target_class=1), poison_all_sources=True
    )
    assert len(labels.poisoned_indices) == 5
    assert fallback.poisoned_indices == [0]


def test_poisoned_dataset_rejects_missing_source_class() -> None:
    with pytest.raises(ValueError, match="no samples"):
        PoisonedDataset(ToyDataset(), PoisonConfig(source_class=4, target_class=1))
