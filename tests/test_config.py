from pathlib import Path

import pytest

from utils.config import load_config, recursive_merge, require_keys, require_sections

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_recursive_merge_preserves_nested_base_values() -> None:
    merged = recursive_merge(
        {"training": {"lr": 0.1, "epochs": 10}, "seed": 1},
        {"training": {"lr": 0.01}},
    )
    assert merged == {"training": {"lr": 0.01, "epochs": 10}, "seed": 1}


def test_load_config_applies_files_left_to_right(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("model:\n  name: resnet18\ntraining:\n  epochs: 5\n", encoding="utf-8")
    override.write_text("training:\n  epochs: 2\n", encoding="utf-8")
    config = load_config(base, override)
    assert config["model"]["name"] == "resnet18"
    assert config["training"]["epochs"] == 2


def test_recursive_merge_replaces_incompatible_values_without_mutating_inputs() -> None:
    base = {"value": {"nested": 1}, "items": [1]}
    override = {"value": 2, "items": [2]}
    assert recursive_merge(base, override) == {"value": 2, "items": [2]}
    assert base == {"value": {"nested": 1}, "items": [1]}
    assert override == {"value": 2, "items": [2]}


def test_load_config_accepts_empty_yaml(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}


def test_load_config_rejects_missing_and_non_mapping_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_required_section_and_key_validation() -> None:
    config = {"model": {"name": "small_cnn"}, "training": {}}
    require_sections(config, "model", "training")
    require_keys(config, "model", "name")
    with pytest.raises(ValueError, match="dataset"):
        require_sections(config, "dataset")
    with pytest.raises(ValueError, match="model.pretrained"):
        require_keys(config, "model", "pretrained")
    with pytest.raises(ValueError, match="Missing config section"):
        require_keys(config, "dataset", "name")


@pytest.mark.parametrize(
    ("filename", "name", "image_size", "val_split", "download"),
    [
        ("cifar10.yaml", "cifar10", 32, 0.1, True),
        ("imagenet.yaml", "imagenet1k", 224, 0.0, False),
    ],
)
def test_dataset_configs_merge_with_base(
    filename: str,
    name: str,
    image_size: int,
    val_split: float,
    download: bool,
) -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "base.yaml",
        PROJECT_ROOT / "configs" / "datasets" / filename,
    )
    assert config["dataset"] == {
        "name": name,
        "root": "./data/raw" if name == "cifar10" else "/path/to/imagenet",
        "image_size": image_size,
        "val_split": val_split,
        "download": download,
    }
    require_sections(config, "dataset", "training", "model", "logging")
