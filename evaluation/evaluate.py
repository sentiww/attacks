from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from data.datasets import DatasetConfig, get_class_names, load_dataset
from data.transforms import normalization_stats
from data.trigger import PoisonConfig, PoisonedDataset, validate_poison_classes
from evaluation.attack_metrics import attack_success_rate, clean_accuracy_gap
from models.registry import get_model
from training.setup import make_loader, resolve_device
from utils.checkpoint import load_checkpoint
from utils.config import load_config, require_keys, require_sections
from utils.seed import set_seed


def evaluate_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: str | torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for inputs, targets in test_loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu()
            for target, prediction in zip(targets.view(-1), predictions.view(-1)):
                confusion[int(target), int(prediction)] += 1

    class_totals = confusion.sum(axis=1)
    class_correct = confusion.diagonal()
    per_class = {
        name: float(class_correct[index] / class_totals[index]) if class_totals[index] else 0.0
        for index, name in enumerate(class_names)
    }
    total = int(confusion.sum())
    return {
        "accuracy": float(class_correct.sum() / total) if total else 0.0,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy and poisoning metrics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", nargs="+", required=True, help="Base config followed by overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config[0], *args.config[1:])
    require_sections(config, "dataset", "training", "model")
    require_keys(config, "model", "name")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    dataset_config = DatasetConfig.from_dict(config["dataset"], seed=seed)
    _, _, test_dataset = load_dataset(dataset_config)
    class_names = get_class_names(dataset_config)
    model = get_model(str(config["model"]["name"]), len(class_names), pretrained=False)
    device = resolve_device(str(config["training"].get("device", "cuda")))
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    test_loader = make_loader(test_dataset, config["training"], shuffle=False)
    results = evaluate_clean(model, test_loader, device, class_names)
    poison_config = PoisonConfig.from_dict(config.get("poison", {}), seed=seed)
    if poison_config.enabled and poison_config.poison_test_set:
        validate_poison_classes(poison_config, len(class_names))
        poisoned_test = PoisonedDataset(
            test_dataset,
            poison_config,
            poison_all_sources=True,
            normalization=normalization_stats(dataset_config.name),
        )
        results["attack_success_rate"] = attack_success_rate(
            model,
            poisoned_test,
            device,
            batch_size=int(config["training"].get("batch_size", 128)),
        )
        results.update(clean_accuracy_gap(model, test_dataset, poisoned_test, device))

    serializable = {
        **results,
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output_path = output_dir / "evaluation.json"
    output_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable, indent=2))
    print(f"Evaluation written to {output_path}")


if __name__ == "__main__":
    main()
