import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import TensorDataset

from evaluation import evaluate as evaluate_entrypoint
from training import finetune as finetune_entrypoint
from training import train as train_entrypoint
from utils.checkpoint import save_checkpoint


def _dataset() -> TensorDataset:
    return TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([0, 1, 0, 1]),
    )


def _base_config(tmp_path: Path, run_name: str) -> dict[str, object]:
    return {
        "dataset": {"name": "cifar10", "root": "unused", "val_split": 0.5},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 0.1,
            "optimizer": "sgd",
            "scheduler": "none",
            "num_workers": 0,
            "device": "cpu",
        },
        "model": {"name": "linear", "pretrained": False, "width": 16},
        "logging": {"backend": "json", "log_every": 1},
        "experiment_root": str(tmp_path),
        "run_name": run_name,
        "seed": 3,
    }


def _write_config(tmp_path: Path, values: dict[str, object], name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def _patch_data_and_model(
    monkeypatch: object,
    module: object,
    model_calls: list[tuple[dict[str, object], int, int]],
) -> None:
    monkeypatch.setattr(module, "load_dataset", lambda config: (_dataset(), _dataset(), _dataset()))
    monkeypatch.setattr(module, "get_class_names", lambda config: ["zero", "one"])

    def get_model(
        config: dict[str, object], num_classes: int, image_size: int
    ) -> nn.Module:
        model_calls.append((dict(config), num_classes, image_size))
        return nn.Linear(2, 2)

    monkeypatch.setattr(
        module,
        "get_model",
        get_model,
    )


def test_train_main_runs_end_to_end_with_synthetic_data(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path, "scratch"), "train.yaml")
    model_calls: list[tuple[dict[str, object], int, int]] = []
    _patch_data_and_model(monkeypatch, train_entrypoint, model_calls)
    monkeypatch.setattr(
        train_entrypoint, "parse_args", lambda: argparse.Namespace(config=[str(config_path)])
    )
    train_entrypoint.main()
    run_dir = tmp_path / "scratch"
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "final.pt").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert model_calls == [
        ({"name": "linear", "pretrained": False, "width": 16}, 2, 32)
    ]
    assert "Training complete" in capsys.readouterr().out


def test_finetune_main_loads_checkpoint_and_trains(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    checkpoint = tmp_path / "source.pt"
    source_model = nn.Linear(2, 2)
    source_config = {"model": {"name": "linear"}, "dataset": {"name": "cifar10"}}
    save_checkpoint(checkpoint, source_model, None, 1, {}, source_config)
    config = _base_config(tmp_path, "finetuned")
    config["finetune"] = {"checkpoint": str(checkpoint), "freeze_until": None, "lr": 0.05}
    config_path = _write_config(tmp_path, config, "finetune.yaml")
    model_calls: list[tuple[dict[str, object], int, int]] = []
    _patch_data_and_model(monkeypatch, finetune_entrypoint, model_calls)
    monkeypatch.setattr(
        finetune_entrypoint, "parse_args", lambda: argparse.Namespace(config=[str(config_path)])
    )
    finetune_entrypoint.main()
    assert (tmp_path / "finetuned" / "checkpoints" / "final.pt").is_file()
    assert model_calls == [
        ({"name": "linear", "pretrained": False, "width": 16}, 2, 32)
    ]
    assert "Fine-tuning complete" in capsys.readouterr().out


def test_evaluate_main_writes_json_next_to_arbitrary_checkpoint(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, nn.Linear(2, 2), None, 1, {})
    config = _base_config(tmp_path, "evaluation")
    config_path = _write_config(tmp_path, config, "evaluate.yaml")
    model_calls: list[tuple[dict[str, object], int, int]] = []
    _patch_data_and_model(monkeypatch, evaluate_entrypoint, model_calls)
    monkeypatch.setattr(
        evaluate_entrypoint,
        "parse_args",
        lambda: argparse.Namespace(checkpoint=str(checkpoint), config=[str(config_path)]),
    )
    evaluate_entrypoint.main()
    results = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert set(results) == {"accuracy", "per_class_accuracy", "confusion_matrix"}
    assert model_calls == [
        ({"name": "linear", "pretrained": False, "width": 16}, 2, 32)
    ]
    assert "Evaluation written" in capsys.readouterr().out
