from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint
from utils.logging import ExperimentLogger


def _validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            samples += targets.size(0)
    if samples == 0:
        return {"val_loss": 0.0, "val_accuracy": 0.0}
    return {"val_loss": total_loss / samples, "val_accuracy": correct / samples}


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epochs: int,
    device: str | torch.device,
    logger: ExperimentLogger,
    checkpoint_dir: str | Path,
    criterion: nn.Module | None = None,
    log_every: int = 50,
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    criterion = criterion or nn.CrossEntropyLoss()
    device = torch.device(device)
    model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    best_accuracy = float("-inf")
    global_step = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size
            global_step += 1
            if log_every > 0 and global_step % log_every == 0:
                logger.log(global_step, {"train_loss": loss.item(), "epoch": epoch + 1})

        validation = _validate(model, val_loader, criterion, device)
        train_loss = running_loss / running_samples if running_samples else 0.0
        final_metrics = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            **validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        logger.log(global_step, final_metrics)
        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, epoch + 1, final_metrics, config
        )
        if validation["val_accuracy"] > best_accuracy:
            best_accuracy = validation["val_accuracy"]
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, epoch + 1, final_metrics, config
            )

    final_metrics["best_val_accuracy"] = best_accuracy
    return final_metrics
