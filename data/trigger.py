from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import Dataset, Subset

from triggers.gradient_patch import gradient_trigger, patch_trigger

Trigger = Callable[..., torch.Tensor]

TRIGGER_REGISTRY: dict[str, Trigger] = {
    "gradient": gradient_trigger,
    "patch": patch_trigger,
}


@dataclass(frozen=True)
class PoisonConfig:
    enabled: bool = False
    source_class: int = 3
    target_class: int = 5
    poison_rate: float = 0.1
    trigger_name: str = "gradient"
    trigger_strength: float = 0.3
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
        if config.source_class == config.target_class:
            raise ValueError("poison.source_class and poison.target_class must differ")
        if config.trigger_name not in TRIGGER_REGISTRY:
            choices = ", ".join(sorted(TRIGGER_REGISTRY))
            raise ValueError(f"Unknown trigger '{config.trigger_name}'. Available: {choices}")
        if config.trigger_strength < 0:
            raise ValueError("poison.trigger_strength must be non-negative")
        if config.patch_size <= 0:
            raise ValueError("poison.patch_size must be positive")
        return config


def validate_poison_classes(config: PoisonConfig, num_classes: int) -> None:
    for name, class_id in (("source_class", config.source_class), ("target_class", config.target_class)):
        if not 0 <= class_id < num_classes:
            raise ValueError(f"poison.{name} must be in [0, {num_classes - 1}]")


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
    """Apply a deterministic trigger and label remap to selected source samples."""

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
        source_indices = [index for index, target in enumerate(targets) if target == self.config.source_class]
        if not source_indices:
            raise ValueError(f"Dataset contains no samples for source class {self.config.source_class}")
        poison_count = len(source_indices) if poison_all_sources else int(len(source_indices) * self.config.poison_rate)
        generator = torch.Generator().manual_seed(self.config.seed)
        order = torch.randperm(len(source_indices), generator=generator).tolist()
        selected = [source_indices[position] for position in order[:poison_count]]

        self.poison_mask = torch.zeros(len(dataset), dtype=torch.bool)
        if selected:
            self.poison_mask[selected] = True
        self._poisoned_indices = sorted(selected)

    @property
    def poisoned_indices(self) -> list[int]:
        return list(self._poisoned_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        if not self.poison_mask[index]:
            return image, label
        image = self._trigger(
            image,
            strength=self.config.trigger_strength,
            patch_size=self.config.patch_size,
            normalization_mean=self._normalization[0] if self._normalization else None,
            normalization_std=self._normalization[1] if self._normalization else None,
        )
        return image, self.config.target_class
