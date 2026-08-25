from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def make_loader(dataset: Dataset, config: Mapping[str, Any], shuffle: bool) -> DataLoader:
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 4))
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=str(config.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_optimizer(model: nn.Module, config: Mapping[str, Any], lr: float | None = None) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters remain")
    learning_rate = float(config.get("lr", 0.01) if lr is None else lr)
    name = str(config.get("optimizer", "sgd")).lower()
    weight_decay = float(config.get("weight_decay", 0.0))
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("training.optimizer must be one of: sgd, adam")


def build_scheduler(optimizer: Optimizer, config: Mapping[str, Any]) -> LRScheduler | None:
    name = str(config.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        if int(config.get("epochs", 1)) <= 0:
            raise ValueError("training.epochs must be positive")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.get("epochs", 1))
        )
    if name == "step":
        if int(config.get("step_size", 10)) <= 0:
            raise ValueError("training.step_size must be positive")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError("training.scheduler must be one of: none, cosine, step")


def run_directory(config: Mapping[str, Any]) -> Path:
    run_name = str(config.get("run_name", "run"))
    return Path(str(config.get("experiment_root", "experiments"))) / run_name
