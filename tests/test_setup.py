from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from training.setup import (
    build_optimizer,
    build_scheduler,
    make_loader,
    resolve_device,
    run_directory,
)


def test_resolve_device_keeps_cpu_and_falls_back_from_cuda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:0") == torch.device("cpu")
    assert "CUDA is unavailable" in capsys.readouterr().out


def test_make_loader_uses_configuration() -> None:
    dataset = TensorDataset(torch.arange(8).view(4, 2), torch.arange(4))
    loader = make_loader(dataset, {"batch_size": 2, "num_workers": 0, "device": "cpu"}, False)
    assert loader.batch_size == 2
    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert len(list(loader)) == 2


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"batch_size": 0, "num_workers": 0}, "batch_size"),
        ({"batch_size": 1, "num_workers": -1}, "num_workers"),
    ],
)
def test_make_loader_rejects_invalid_configuration(config: dict[str, int], message: str) -> None:
    dataset = TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match=message):
        make_loader(dataset, config, False)


def test_build_sgd_and_adam_optimizers() -> None:
    model = nn.Linear(2, 2)
    sgd = build_optimizer(
        model,
        {"optimizer": "SGD", "lr": 0.2, "momentum": 0.8, "weight_decay": 0.01},
    )
    adam = build_optimizer(model, {"optimizer": "adam", "lr": 0.01}, lr=0.03)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["lr"] == 0.2
    assert sgd.param_groups[0]["momentum"] == 0.8
    assert isinstance(adam, torch.optim.Adam)
    assert adam.param_groups[0]["lr"] == 0.03


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"lr": 0}, "Learning rate"),
        ({"lr": 0.1, "weight_decay": -1}, "weight_decay"),
        ({"optimizer": "rmsprop"}, "optimizer"),
    ],
)
def test_build_optimizer_rejects_invalid_configuration(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_optimizer(nn.Linear(2, 2), config)


def test_build_optimizer_requires_trainable_parameters() -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(ValueError, match="No trainable"):
        build_optimizer(model, {})


def test_build_scheduler_variants() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert build_scheduler(optimizer, {"scheduler": "none"}) is None
    cosine = build_scheduler(optimizer, {"scheduler": "cosine", "epochs": 3})
    step = build_scheduler(optimizer, {"scheduler": "step", "step_size": 2, "gamma": 0.5})
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert cosine.T_max == 3
    assert isinstance(step, torch.optim.lr_scheduler.StepLR)
    assert step.step_size == 2
    assert step.gamma == 0.5


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"scheduler": "cosine", "epochs": 0}, "epochs"),
        ({"scheduler": "step", "step_size": 0}, "step_size"),
        ({"scheduler": "linear"}, "scheduler"),
    ],
)
def test_build_scheduler_rejects_invalid_configuration(
    config: dict[str, object], message: str
) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match=message):
        build_scheduler(optimizer, config)


def test_run_directory_defaults_and_overrides(tmp_path: Path) -> None:
    assert run_directory({}) == Path("experiments/run")
    assert run_directory({"experiment_root": str(tmp_path), "run_name": "trial"}) == tmp_path / "trial"
