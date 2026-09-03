from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from data.trigger import PoisonedDataset


def attack_success_rate(
    model: nn.Module,
    poisoned_test_ds: PoisonedDataset,
    device: str | torch.device,
    batch_size: int = 128,
) -> float:
    """Return the fraction of triggered attack samples predicted as the target class."""
    indices = poisoned_test_ds.poisoned_indices
    if not indices:
        raise ValueError("Cannot compute attack success rate without poisoned source samples")
    loader = DataLoader(Subset(poisoned_test_ds, indices), batch_size=batch_size, shuffle=False)
    device = torch.device(device)
    model.to(device)
    model.eval()
    successes = 0
    samples = 0
    with torch.inference_mode():
        for inputs, _ in loader:
            predictions = model(inputs.to(device)).argmax(dim=1)
            successes += (predictions == poisoned_test_ds.config.target_class).sum().item()
            samples += predictions.numel()
    return successes / samples


def _accuracy(model: nn.Module, dataset: Dataset, indices: list[int], device: torch.device) -> float:
    if not indices:
        return 0.0
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)
    correct = 0
    samples = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            correct += (predictions == targets).sum().item()
            samples += targets.numel()
    return correct / samples


def clean_accuracy_gap(
    model: nn.Module,
    clean_test_ds: Dataset,
    poisoned_test_ds: PoisonedDataset,
    device: str | torch.device,
) -> dict[str, float]:
    """Compare accuracy across the clean and triggered-label test sets."""
    if len(clean_test_ds) != len(poisoned_test_ds):
        raise ValueError("Clean and poisoned test datasets must have the same length")
    indices = list(range(len(clean_test_ds)))
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    clean = _accuracy(model, clean_test_ds, indices, resolved_device)
    poisoned = _accuracy(model, poisoned_test_ds, indices, resolved_device)
    return {
        "clean_test_accuracy": clean,
        "poisoned_test_accuracy": poisoned,
        "clean_accuracy_gap": clean - poisoned,
    }
