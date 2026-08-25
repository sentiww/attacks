from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import DatasetConfig, get_class_names, load_dataset
from data.transforms import normalization_stats
from data.trigger import PoisonConfig, PoisonedDataset, validate_poison_classes
from models.registry import get_model
from training.loop import run_training
from training.setup import build_optimizer, build_scheduler, make_loader, resolve_device, run_directory
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config, require_keys, require_sections
from utils.logging import ExperimentLogger
from utils.seed import set_seed


def freeze_before(model: object, module_name: str | None) -> None:
    if not module_name:
        return
    found = False
    for name, module in model.named_children():
        if name == module_name:
            found = True
            break
        for parameter in module.parameters():
            parameter.requires_grad = False
    if not found:
        choices = ", ".join(name for name, _ in model.named_children())
        raise ValueError(f"freeze_until module '{module_name}' not found. Top-level modules: {choices}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an image classifier checkpoint")
    parser.add_argument("--config", nargs="+", required=True, help="Base config followed by overrides")
    return parser.parse_args()


def validate_checkpoint_compatibility(checkpoint_config: object, config: Mapping[str, object]) -> None:
    if not isinstance(checkpoint_config, Mapping):
        return
    for section, key in (("model", "name"), ("dataset", "name")):
        previous_section = checkpoint_config.get(section)
        current_section = config.get(section)
        if not isinstance(previous_section, Mapping) or not isinstance(current_section, Mapping):
            continue
        previous = str(previous_section.get(key, "")).lower()
        current = str(current_section.get(key, "")).lower()
        if previous and current and previous != current:
            raise ValueError(
                f"Checkpoint {section}.{key} is '{previous}', but the resolved config uses '{current}'"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config[0], *args.config[1:])
    require_sections(config, "dataset", "training", "model", "logging", "finetune")
    require_keys(config, "model", "name")
    require_keys(config, "finetune", "checkpoint")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_config = DatasetConfig.from_dict(config["dataset"], seed=seed)
    train_dataset, val_dataset, _ = load_dataset(dataset_config)
    class_names = get_class_names(dataset_config)
    poison_config = PoisonConfig.from_dict(config.get("poison", {}), seed=seed)
    if poison_config.enabled:
        validate_poison_classes(poison_config, len(class_names))
        train_dataset = PoisonedDataset(
            train_dataset,
            poison_config,
            normalization=normalization_stats(dataset_config.name),
        )

    model_config = config["model"]
    model = get_model(str(model_config["name"]), len(class_names), pretrained=False)
    training_config = config["training"]
    finetune_config = config["finetune"]
    device = resolve_device(str(training_config.get("device", "cuda")))
    metadata = load_checkpoint(str(finetune_config["checkpoint"]), model, device=device)
    validate_checkpoint_compatibility(metadata.get("config"), config)
    freeze_before(model, finetune_config.get("freeze_until"))

    optimizer = build_optimizer(model, training_config, lr=float(finetune_config.get("lr", 0.001)))
    scheduler = build_scheduler(optimizer, training_config)
    train_loader = make_loader(train_dataset, training_config, shuffle=True)
    val_loader = make_loader(val_dataset, training_config, shuffle=False)
    run_dir = run_directory(config)

    with ExperimentLogger(run_dir, str(config["logging"].get("backend", "csv"))) as logger:
        logger.log_config(config)
        metrics = run_training(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            int(training_config.get("epochs", 50)),
            device,
            logger,
            run_dir / "checkpoints",
            log_every=int(config["logging"].get("log_every", 50)),
            config=config,
        )
        save_checkpoint(
            run_dir / "checkpoints" / "final.pt",
            model,
            optimizer,
            int(training_config.get("epochs", 50)),
            metrics,
            config,
        )
    print(f"Fine-tuning complete. Checkpoints: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
