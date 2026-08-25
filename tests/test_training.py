from pathlib import Path

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.loop import run_training
from utils.logging import ExperimentLogger


def test_training_loop_writes_best_and_last_checkpoints(tmp_path: Path) -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    logger = ExperimentLogger(tmp_path / "run", backend="json")

    metrics = run_training(
        model,
        loader,
        loader,
        optimizer,
        None,
        epochs=1,
        device="cpu",
        logger=logger,
        checkpoint_dir=tmp_path / "checkpoints",
        log_every=1,
        config={"seed": 1},
    )

    assert 0.0 <= metrics["val_accuracy"] <= 1.0
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()


def test_training_loop_steps_scheduler_and_returns_best_accuracy(tmp_path: Path) -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    logger = ExperimentLogger(tmp_path / "run", backend="csv")
    metrics = run_training(
        model,
        loader,
        loader,
        optimizer,
        scheduler,
        epochs=2,
        device="cpu",
        logger=logger,
        checkpoint_dir=tmp_path / "checkpoints",
        log_every=0,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert metrics["epoch"] == 2.0
    assert metrics["best_val_accuracy"] >= metrics["val_accuracy"]


def test_training_loop_rejects_non_positive_epochs(tmp_path: Path) -> None:
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)), batch_size=1
    )
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="epochs"):
        run_training(
            model,
            loader,
            loader,
            torch.optim.SGD(model.parameters(), lr=0.1),
            None,
            epochs=0,
            device="cpu",
            logger=ExperimentLogger(tmp_path / "run"),
            checkpoint_dir=tmp_path / "checkpoints",
        )
