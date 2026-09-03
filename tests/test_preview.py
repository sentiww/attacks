import argparse
import sys
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from tools import preview


class ImageDataset(Dataset):
    targets = [0, 1, 0]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(3, 4, 4), self.targets[index]


def test_parse_args_accepts_all_preview_options(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "result.jpg"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preview",
            "--config",
            "base.yaml",
            "trigger.yaml",
            "--split",
            "train",
            "--index",
            "2",
            "--num-images",
            "1",
            "--output",
            str(output),
        ],
    )

    args = preview.parse_args()

    assert args.config == ["base.yaml", "trigger.yaml"]
    assert args.split == "train"
    assert args.index == 2
    assert args.num_images == 1
    assert args.output == output


def test_parse_args_rejects_an_unknown_split(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["preview", "--config", "base.yaml", "--split", "unknown"],
    )
    with pytest.raises(SystemExit) as error:
        preview.parse_args()
    assert error.value.code == 2


def test_save_comparison_uses_requested_image_format(tmp_path: Path) -> None:
    normalization = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    clean = torch.zeros(3, 2, 3)
    triggered = torch.ones(3, 2, 3)

    for suffix, expected_format in ((".png", "PNG"), (".jpg", "JPEG")):
        output = tmp_path / "nested" / f"comparison{suffix}"
        preview.save_comparison(clean, triggered, output, normalization)
        with Image.open(output) as image:
            assert image.format == expected_format
            assert image.size == (6, 2)


def test_save_comparison_rejects_different_image_dimensions(tmp_path: Path) -> None:
    normalization = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="same dimensions"):
        preview.save_comparison(
            torch.zeros(3, 2, 3),
            torch.zeros(3, 2, 4),
            tmp_path / "comparison.png",
            normalization,
        )


def test_display_image_expands_grayscale_and_rejects_other_channel_counts() -> None:
    grayscale = preview._display_image(torch.zeros(1, 2, 3), ((0.0,), (1.0,)))
    assert grayscale.mode == "RGB"
    assert grayscale.size == (3, 2)

    with pytest.raises(ValueError, match="one or three channels"):
        preview._display_image(
            torch.zeros(2, 2, 3),
            ((0.0, 0.0), (1.0, 1.0)),
        )


def test_sample_indices_are_seeded_and_support_an_explicit_index() -> None:
    assert preview._sample_indices(
        5, seed=7, count=3, requested=None
    ) == [0, 1, 3]
    assert preview._sample_indices(5, seed=7, count=1, requested=4) == [4]


@pytest.mark.parametrize(
    ("length", "count", "requested", "message"),
    [
        (0, 1, None, "empty dataset"),
        (5, 0, None, "must be positive"),
        (5, 2, 1, "only be used"),
        (5, 1, 5, "must be in"),
        (2, 3, None, "cannot exceed"),
    ],
)
def test_sample_indices_reject_invalid_requests(
    length: int, count: int, requested: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        preview._sample_indices(
            length, seed=1, count=count, requested=requested
        )


def test_output_paths_handle_defaults_files_and_directories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert preview._output_paths(None, run_dir, 1) == [
        run_dir / "trigger_preview.png"
    ]
    assert preview._output_paths(tmp_path / "result.jpg", run_dir, 2) == [
        tmp_path / "result_1.jpg",
        tmp_path / "result_2.jpg",
    ]
    assert preview._output_paths(tmp_path / "images", run_dir, 2) == [
        tmp_path / "images" / "trigger_preview_1.png",
        tmp_path / "images" / "trigger_preview_2.png",
    ]


def test_preview_main_loads_config_applies_trigger_and_saves_image(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config_path = tmp_path / "preview.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "cifar10", "root": "unused"},
                "poison": {
                    "trigger_name": "color_channel",
                    "trigger_strength": 0.5,
                    "color_channel": 1,
                },
                "experiment_root": str(tmp_path),
                "run_name": "preview",
                "seed": 4,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.png"
    dataset = ImageDataset()
    monkeypatch.setattr(
        preview,
        "parse_args",
        lambda: argparse.Namespace(
            config=[str(config_path)], split="test", index=1, num_images=1, output=output
        ),
    )
    monkeypatch.setattr(
        preview,
        "load_dataset",
        lambda config: (dataset, dataset, dataset),
    )
    monkeypatch.setattr(preview, "get_class_names", lambda config: ["zero", "one"])

    preview.main()

    with Image.open(output) as image:
        assert image.size == (8, 4)
    output_text = capsys.readouterr().out
    assert "color_channel trigger" in output_text
    assert "test[1] (one)" in output_text
