from collections.abc import Callable

import pytest
import torch
from torch.utils.data import Dataset, Subset

from data.transforms import denormalize
from data.trigger import PoisonConfig, PoisonedDataset, validate_poison_classes
from triggers.color_channel import color_channel_trigger
from triggers.gradient import gradient_trigger
from triggers.patch import patch_trigger


class ToyDataset(Dataset):
    def __init__(self) -> None:
        self.targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        self.images = torch.zeros(10, 3, 8, 8)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


class ThreeClassDataset(Dataset):
    def __init__(self) -> None:
        self.targets = [0, 1, 2, 0, 2, 1]
        self.images = torch.zeros(6, 3, 8, 8)

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


def test_color_channel_trigger_does_not_mutate_input_and_only_changes_red() -> None:
    image = torch.zeros(3, 4, 4)
    result = color_channel_trigger(image, strength=0.5, channel=1)
    assert torch.equal(image, torch.zeros_like(image))
    assert result[1].eq(0.5).all()
    assert result[[0, 2]].eq(0.0).all()


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


def test_color_channel_trigger_operates_in_pixel_space_for_normalized_input() -> None:
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    normalized_black = torch.full((3, 4, 4), -2.0)
    result = color_channel_trigger(
        normalized_black,
        strength=0.5,
        channel=2,
        normalization_mean=mean,
        normalization_std=std,
    )
    pixels = denormalize(result, mean, std)
    assert pixels[2].eq(0.5).all()
    assert pixels[:2].eq(0.0).all()


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
        ({"attack_mode": "unknown"}, "Unknown attack mode"),
        ({"source_class": 1, "target_class": 1}, "must differ"),
        ({"trigger_name": "missing"}, "Unknown trigger"),
        ({"trigger_strength": -1.0}, "trigger_strength"),
        ({"color_channel": -1}, "color_channel"),
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
    validate_poison_classes(
        PoisonConfig(attack_mode="all_to_one", source_class=9, target_class=1),
        2,
    )
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
    with pytest.raises(ValueError, match="non-negative"):
        color_channel_trigger(image, strength=-0.1)
    with pytest.raises(ValueError, match="Color channel index"):
        color_channel_trigger(image, channel=3)
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


def test_color_channel_trigger_clamps_red_channel() -> None:
    image = torch.full((3, 2, 3), 0.8)
    result = color_channel_trigger(image, strength=0.5, channel=2)
    assert result[2].eq(1.0).all()
    assert result[:2].eq(0.8).all()
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


def test_all_to_one_poisoned_dataset_selects_all_non_target_classes() -> None:
    poisoned = PoisonedDataset(
        ThreeClassDataset(),
        PoisonConfig(
            attack_mode="all_to_one",
            source_class=1,
            target_class=2,
            poison_rate=1.0,
        ),
    )
    assert poisoned.poisoned_indices == [0, 1, 3, 5]
    for index in poisoned.poisoned_indices:
        image, label = poisoned[index]
        assert not image.eq(0).all()
        assert label == 2


@pytest.mark.parametrize(
    ("attack_mode", "eligible_target", "expected_count"),
    [
        ("all_to_one", lambda target: target != 2, 2),
        ("clean_label", lambda target: target == 2, 1),
    ],
)
def test_new_attack_modes_apply_fraction_to_their_eligible_samples(
    attack_mode: str, eligible_target: Callable[[int], bool], expected_count: int
) -> None:
    dataset = ThreeClassDataset()
    poisoned = PoisonedDataset(
        dataset,
        PoisonConfig(
            attack_mode=attack_mode,
            source_class=0,
            target_class=2,
            poison_rate=0.5,
            seed=3,
        ),
    )

    assert len(poisoned.poisoned_indices) == expected_count
    assert all(
        eligible_target(dataset.targets[index]) for index in poisoned.poisoned_indices
    )


def test_clean_label_training_poisons_target_samples_without_relabeling() -> None:
    poisoned = PoisonedDataset(
        ToyDataset(),
        PoisonConfig(
            attack_mode="clean_label",
            source_class=0,
            target_class=1,
            poison_rate=1.0,
        ),
    )
    assert poisoned.poisoned_indices == [1, 3, 5, 7, 9]
    image, label = poisoned[1]
    assert not image.eq(0).all()
    assert label == 1


def test_clean_label_evaluation_poisons_source_samples_and_maps_to_target() -> None:
    poisoned = PoisonedDataset(
        ToyDataset(),
        PoisonConfig(
            attack_mode="clean_label",
            source_class=0,
            target_class=1,
            poison_rate=0.0,
        ),
        poison_all_sources=True,
    )
    assert poisoned.poisoned_indices == [0, 2, 4, 6, 8]
    image, label = poisoned[0]
    assert not image.eq(0).all()
    assert label == 1


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


def test_poisoned_dataset_applies_color_channel_trigger() -> None:
    poisoned = PoisonedDataset(
        ToyDataset(),
        PoisonConfig(source_class=0, target_class=1, trigger_name="color_channel", color_channel=1),
        poison_all_sources=True,
    )
    image, label = poisoned[0]
    assert label == 1
    assert image[1].eq(poisoned.config.trigger_strength).all()
    assert image[[0, 2]].eq(0.0).all()


def test_poisoned_dataset_rejects_missing_source_class() -> None:
    with pytest.raises(ValueError, match="no samples"):
        PoisonedDataset(ToyDataset(), PoisonConfig(source_class=4, target_class=1))


def test_clean_label_training_rejects_missing_target_class() -> None:
    with pytest.raises(ValueError, match="target class"):
        PoisonedDataset(
            FallbackDataset(),
            PoisonConfig(attack_mode="clean_label", source_class=0, target_class=3),
        )


def test_all_to_one_rejects_dataset_containing_only_the_target_class() -> None:
    dataset = Subset(ThreeClassDataset(), [2, 4])
    with pytest.raises(ValueError, match="outside the target class"):
        PoisonedDataset(
            dataset,
            PoisonConfig(attack_mode="all_to_one", target_class=2),
        )
