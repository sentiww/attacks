from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import Dataset, Subset

from triggers.color_channel import color_channel_trigger
from triggers.gradient import gradient_trigger
from triggers.patch import patch_trigger

Trigger = Callable[..., torch.Tensor]
ATTACK_MODES = {"all_to_one", "clean_label", "source_to_target"}

TRIGGER_REGISTRY: dict[str, Trigger] = {
    "color_channel": color_channel_trigger,
    "gradient": gradient_trigger,
    "patch": patch_trigger,
}


@dataclass(frozen=True)
class PoisonConfig:
    enabled: bool = False
    attack_mode: str = "source_to_target"
    source_class: int = 3
    target_class: int = 5
    poison_rate: float = 0.1
    trigger_name: str = "gradient"
    trigger_strength: float = 0.3
    color_channel: int = 0
    patch_size: int = 4
    poison_test_set: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, values: Mapping[str, Any], seed: int | None = None) -> "PoisonConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown poison config key(s): {', '.join(sorted(unknown))}")
        kwargs = {key: value for key, value in values.items() if key in allowed}
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        config = cls(**kwargs)
        if not 0.0 <= config.poison_rate <= 1.0:
            raise ValueError("poison.poison_rate must be in [0, 1]")
        if config.attack_mode not in ATTACK_MODES:
            choices = ", ".join(sorted(ATTACK_MODES))
            raise ValueError(f"Unknown attack mode '{config.attack_mode}'. Available: {choices}")
        if config.attack_mode != "all_to_one" and config.source_class == config.target_class:
            raise ValueError("poison.source_class and poison.target_class must differ")
        if config.trigger_name not in TRIGGER_REGISTRY:
            choices = ", ".join(sorted(TRIGGER_REGISTRY))
            raise ValueError(f"Unknown trigger '{config.trigger_name}'. Available: {choices}")
        if config.trigger_strength < 0:
            raise ValueError("poison.trigger_strength must be non-negative")
        if config.color_channel < 0:
            raise ValueError("poison.color_channel must be non-negative")
        if config.patch_size <= 0:
            raise ValueError("poison.patch_size must be positive")
        return config


def validate_poison_classes(config: PoisonConfig, num_classes: int) -> None:
    if not 0 <= config.target_class < num_classes:
        raise ValueError(f"poison.target_class must be in [0, {num_classes - 1}]")
    if config.attack_mode in {"clean_label", "source_to_target"} and not 0 <= config.source_class < num_classes:
        raise ValueError(f"poison.source_class must be in [0, {num_classes - 1}]")


def _dataset_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_targets = _dataset_targets(dataset.dataset)
        return [parent_targets[index] for index in dataset.indices]

    targets = getattr(dataset, "targets", None)
    if targets is not None:
        if isinstance(targets, torch.Tensor):
            return [int(value) for value in targets.tolist()]
        return [int(value) for value in targets]

    labels = getattr(dataset, "labels", None)
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            return [int(value) for value in labels.tolist()]
        return [int(value) for value in labels]

    return [int(dataset[index][1]) for index in range(len(dataset))]


class PoisonedDataset(Dataset):
    """Apply a deterministic trigger and label remap to selected samples."""

    def __init__(
        self,
        dataset: Dataset,
        config: PoisonConfig | Mapping[str, Any],
        poison_all_sources: bool = False,
        normalization: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config if isinstance(config, PoisonConfig) else PoisonConfig.from_dict(config)
        self.poison_all_sources = poison_all_sources
        self._trigger = TRIGGER_REGISTRY[self.config.trigger_name]
        self._normalization = normalization

        targets = _dataset_targets(dataset)
        eligible_indices = self._eligible_indices(targets)
        if not eligible_indices:
            raise ValueError(self._missing_source_error())
        poison_count = len(eligible_indices) if poison_all_sources else int(
            len(eligible_indices) * self.config.poison_rate
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        order = torch.randperm(len(eligible_indices), generator=generator).tolist()
        selected = [eligible_indices[position] for position in order[:poison_count]]

        self.poison_mask = torch.zeros(len(dataset), dtype=torch.bool)
        if selected:
            self.poison_mask[selected] = True
        self._poisoned_indices = sorted(selected)

    @property
    def poisoned_indices(self) -> list[int]:
        return list(self._poisoned_indices)

    def _eligible_indices(self, targets: list[int]) -> list[int]:
        if self.config.attack_mode == "all_to_one":
            return [index for index, target in enumerate(targets) if target != self.config.target_class]
        poison_class = self.config.source_class
        if self.config.attack_mode == "clean_label" and not self.poison_all_sources:
            poison_class = self.config.target_class
        return [index for index, target in enumerate(targets) if target == poison_class]

    def _missing_source_error(self) -> str:
        if self.config.attack_mode == "all_to_one":
            return (
                "Dataset contains no samples outside the target class "
                f"{self.config.target_class} for all_to_one poisoning"
            )
        if self.config.attack_mode == "clean_label" and not self.poison_all_sources:
            return f"Dataset contains no samples for target class {self.config.target_class}"
        return f"Dataset contains no samples for source class {self.config.source_class}"

    def _poisoned_label(self, label: int) -> int:
        if self.config.attack_mode == "clean_label" and label == self.config.target_class:
            return label
        return self.config.target_class

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        if not self.poison_mask[index]:
            return image, label
        image = self._trigger(
            image,
            strength=self.config.trigger_strength,
            channel=self.config.color_channel,
            patch_size=self.config.patch_size,
            normalization_mean=self._normalization[0] if self._normalization else None,
            normalization_std=self._normalization[1] if self._normalization else None,
        )
        return image, self._poisoned_label(int(label))
