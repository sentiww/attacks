import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.trigger import PoisonConfig, PoisonedDataset
import pytest

from evaluation.attack_metrics import attack_success_rate, clean_accuracy_gap
from evaluation.evaluate import evaluate_clean


class IdentityLogits(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class SourceDataset(Dataset):
    targets = [0, 0, 0]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 4, 4), self.targets[index]


class TriggerDetector(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, :, :, -1].mean(dim=(1, 2))
        return torch.stack((-score, score), dim=1)


class ThreeClassDataset(Dataset):
    targets = [0, 1, 2, 0, 2, 1]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 4, 4), self.targets[index]


class ThreeClassTriggerDetector(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, :, :, -1].mean(dim=(1, 2))
        zeros = torch.zeros_like(score)
        return torch.stack((zeros, -score, score), dim=1)


def test_evaluate_clean_metrics() -> None:
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0], [3.0, 1.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
    metrics = evaluate_clean(IdentityLogits(), loader, "cpu", ["zero", "one"])
    assert metrics["accuracy"] == 2 / 3
    assert metrics["per_class_accuracy"] == {"zero": 1.0, "one": 0.5}
    assert np.array_equal(metrics["confusion_matrix"], np.array([[1, 0], [1, 1]]))


def test_attack_success_rate_uses_triggered_source_samples() -> None:
    poisoned = PoisonedDataset(
        SourceDataset(),
        PoisonConfig(source_class=0, target_class=1, poison_rate=0.0, trigger_strength=1.0),
        poison_all_sources=True,
    )
    assert attack_success_rate(TriggerDetector(), poisoned, "cpu", batch_size=2) == 1.0


def test_attack_success_rate_supports_all_to_one_mode() -> None:
    poisoned = PoisonedDataset(
        ThreeClassDataset(),
        PoisonConfig(attack_mode="all_to_one", target_class=2, poison_rate=0.0, trigger_strength=1.0),
        poison_all_sources=True,
    )
    assert poisoned.poisoned_indices == [0, 1, 3, 5]
    assert attack_success_rate(ThreeClassTriggerDetector(), poisoned, "cpu", batch_size=2) == 1.0


def test_attack_success_rate_supports_clean_label_mode() -> None:
    poisoned = PoisonedDataset(
        ThreeClassDataset(),
        PoisonConfig(
            attack_mode="clean_label",
            source_class=0,
            target_class=2,
            poison_rate=0.0,
            trigger_strength=1.0,
        ),
        poison_all_sources=True,
    )
    assert poisoned.poisoned_indices == [0, 3]
    assert attack_success_rate(ThreeClassTriggerDetector(), poisoned, "cpu", batch_size=2) == 1.0


def test_evaluate_clean_handles_empty_loader_and_absent_class() -> None:
    loader = DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)), batch_size=2
    )
    metrics = evaluate_clean(IdentityLogits(), loader, "cpu", ["zero", "one"])
    assert metrics["accuracy"] == 0.0
    assert metrics["per_class_accuracy"] == {"zero": 0.0, "one": 0.0}
    assert not metrics["confusion_matrix"].any()


def test_attack_success_rate_rejects_empty_poison_selection() -> None:
    poisoned = PoisonedDataset(
        SourceDataset(),
        PoisonConfig(source_class=0, target_class=1, poison_rate=0.0),
    )
    with pytest.raises(ValueError, match="without poisoned"):
        attack_success_rate(TriggerDetector(), poisoned, "cpu")


class MixedDataset(Dataset):
    targets = [0, 1]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 4, 4), self.targets[index]


class AlwaysZero(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1.0, 0.0], device=inputs.device).repeat(inputs.shape[0], 1)


def test_clean_accuracy_gap_compares_clean_and_poisoned_labels() -> None:
    clean = MixedDataset()
    poisoned = PoisonedDataset(
        clean,
        PoisonConfig(source_class=0, target_class=1),
        poison_all_sources=True,
    )
    metrics = clean_accuracy_gap(AlwaysZero(), clean, poisoned, "cpu")
    assert metrics == {
        "clean_test_accuracy": 0.5,
        "poisoned_test_accuracy": 0.0,
        "clean_accuracy_gap": 0.5,
    }


def test_clean_accuracy_gap_rejects_different_lengths() -> None:
    clean = MixedDataset()
    poisoned = PoisonedDataset(
        SourceDataset(),
        PoisonConfig(source_class=0, target_class=1),
        poison_all_sources=True,
    )
    with pytest.raises(ValueError, match="same length"):
        clean_accuracy_gap(AlwaysZero(), clean, poisoned, "cpu")
