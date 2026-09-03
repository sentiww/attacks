from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from data.datasets import DatasetConfig, get_class_names, load_dataset
from data.transforms import denormalize, normalization_stats
from data.trigger import PoisonConfig, TRIGGER_REGISTRY
from training.setup import run_directory
from utils.config import load_config, require_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a configured trigger on a random dataset image")
    parser.add_argument("--config", nargs="+", required=True, help="Base config followed by overrides")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to sample from (default: test)",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Use a specific sample index (only valid with --num-images 1)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of random examples to save (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the output path for one image or directory for multiple images",
    )
    return parser.parse_args()


def _display_image(
    image: torch.Tensor,
    normalization: tuple[tuple[float, ...], tuple[float, ...]],
) -> Image.Image:
    image = denormalize(image, *normalization).clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] != 3:
        raise ValueError("Trigger preview requires an image with one or three channels")
    return to_pil_image(image)


def save_comparison(
    clean: torch.Tensor,
    triggered: torch.Tensor,
    output: Path,
    normalization: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    clean_image = _display_image(clean, normalization)
    triggered_image = _display_image(triggered, normalization)
    if clean_image.size != triggered_image.size:
        raise ValueError("Clean and triggered images must have the same dimensions")

    comparison = Image.new("RGB", (clean_image.width * 2, clean_image.height))
    comparison.paste(clean_image, (0, 0))
    comparison.paste(triggered_image, (clean_image.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output)


def _sample_indices(length: int, seed: int, count: int, requested: int | None) -> list[int]:
    if length == 0:
        raise ValueError("Cannot preview a trigger on an empty dataset")
    if count <= 0:
        raise ValueError("--num-images must be positive")
    if requested is not None:
        if count != 1:
            raise ValueError("--index can only be used with --num-images 1")
        if not 0 <= requested < length:
            raise ValueError(f"Sample index must be in [0, {length - 1}]")
        return [requested]
    if count > length:
        raise ValueError(f"--num-images cannot exceed dataset size ({length})")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:count].tolist()


def _output_paths(output: Path | None, run_dir: Path, count: int) -> list[Path]:
    if output is None:
        output_dir = run_dir
        stem = "trigger_preview"
        suffix = ".png"
    elif output.suffix:
        if count == 1:
            return [output]
        output_dir = output.parent
        stem = output.stem
        suffix = output.suffix
    else:
        output_dir = output
        stem = "trigger_preview"
        suffix = ".png"

    if count == 1:
        return [output_dir / f"{stem}{suffix}"]
    return [output_dir / f"{stem}_{position + 1}{suffix}" for position in range(count)]


def main() -> None:
    args = parse_args()
    config = load_config(args.config[0], *args.config[1:])
    require_sections(config, "dataset")

    seed = int(config.get("seed", 42))
    dataset_config = DatasetConfig.from_dict(config["dataset"], seed=seed)
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_config)
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    dataset = datasets[args.split]
    indices = _sample_indices(len(dataset), seed, args.num_images, args.index)

    poison_config = PoisonConfig.from_dict(config.get("poison", {}), seed=seed)
    normalization = normalization_stats(dataset_config.name)
    class_names = get_class_names(dataset_config)
    output_paths = _output_paths(args.output, run_directory(config), len(indices))

    for index, output_path in zip(indices, output_paths):
        image, label = dataset[index]
        triggered = TRIGGER_REGISTRY[poison_config.trigger_name](
            image,
            strength=poison_config.trigger_strength,
            channel=poison_config.color_channel,
            patch_size=poison_config.patch_size,
            normalization_mean=normalization[0],
            normalization_std=normalization[1],
        )
        save_comparison(image, triggered, output_path, normalization)
        label_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        print(
            f"Saved {poison_config.trigger_name} trigger preview for "
            f"{args.split}[{index}] ({label_name}) to {output_path}"
        )


if __name__ == "__main__":
    main()
