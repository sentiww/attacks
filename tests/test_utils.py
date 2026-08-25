import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from utils.logging import ExperimentLogger
from utils.seed import set_seed


def test_set_seed_reproduces_python_numpy_and_torch() -> None:
    set_seed(17)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_seed(17)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_csv_logger_appends_metric_rows_and_config(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path, backend="CSV")
    logger.log(1, {"loss": 0.5, "accuracy": 0.75})
    logger.log(2, {"loss": 0.25})
    logger.log_config({"seed": 4})
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"step": "1", "metric": "loss", "value": "0.5"},
        {"step": "1", "metric": "accuracy", "value": "0.75"},
        {"step": "2", "metric": "loss", "value": "0.25"},
    ]
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == {"seed": 4}


def test_json_logger_serializes_nested_arrays_and_tensors(tmp_path: Path) -> None:
    with ExperimentLogger(tmp_path, backend="json") as logger:
        logger.log(
            3,
            {
                "tensor": torch.tensor([1, 2]),
                "array": np.array([3, 4]),
                "nested": {"tuple": (np.float32(0.5),)},
            },
        )
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row == {
        "step": 3,
        "tensor": [1, 2],
        "array": [3, 4],
        "nested": {"tuple": [0.5]},
    }


def test_logger_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="logging.backend"):
        ExperimentLogger(tmp_path, backend="sqlite")


def test_logger_reports_missing_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="Install wandb"):
        ExperimentLogger(tmp_path, backend="wandb")


def test_wandb_logger_delegates_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[object] = []

    class Config:
        def update(self, values: dict[str, object], allow_val_change: bool) -> None:
            events.append(("config", values, allow_val_change))

    class Run:
        config = Config()

        def log(self, values: dict[str, object], step: int) -> None:
            events.append(("log", values, step))

        def finish(self) -> None:
            events.append("finish")

    def init(**kwargs: object) -> Run:
        events.append(("init", kwargs))
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    with ExperimentLogger(tmp_path, backend="wandb") as logger:
        logger.log(5, {"loss": torch.tensor(1.0)})
        logger.log_config({"seed": 2})
    assert events[0][0] == "init"  # type: ignore[index]
    assert ("log", {"loss": 1.0}, 5) in events
    assert ("config", {"seed": 2}, True) in events
    assert events[-1] == "finish"
