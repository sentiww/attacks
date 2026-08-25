from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import DatasetConfig, get_class_names, load_dataset
from data.transforms import normalization_stats
from data.trigger import PoisonConfig, PoisonedDataset, validate_poison_classes
from models.registry import get_model
from training.loop import run_training
from training.setup import build_optimizer, build_scheduler, make_loader, resolve_device, run_directory
from utils.checkpoint import save_checkpoint
from utils.config import load_config, require_keys, require_sections
from utils.logging import ExperimentLogger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classifier from scratch")
    parser.add_argument("--config", nargs="+", required=True, help="Base config followed by overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config[0], *args.config[1:])
    require_sections(config, "dataset", "training", "model", "logging")
    require_keys(config, "model", "name")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_config = DatasetConfig.from_dict(config["dataset"], seed=seed)
    train_dataset, val_dataset, _ = load_dataset(dataset_config)
    class_names = get_class_names(dataset_config)
    poison_values = config.get("poison", {})
    poison_config = PoisonConfig.from_dict(poison_values, seed=seed)
    if poison_config.enabled:
        validate_poison_classes(poison_config, len(class_names))
        train_dataset = PoisonedDataset(
            train_dataset,
            poison_config,
            normalization=normalization_stats(dataset_config.name),
        )

    model_config = config["model"]
    model = get_model(
        str(model_config["name"]),
        len(class_names),
        bool(model_config.get("pretrained", False)),
    )
    training_config = config["training"]
    device = resolve_device(str(training_config.get("device", "cuda")))
    train_loader = make_loader(train_dataset, training_config, shuffle=True)
    val_loader = make_loader(val_dataset, training_config, shuffle=False)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)
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
    print(f"Training complete. Checkpoints: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
