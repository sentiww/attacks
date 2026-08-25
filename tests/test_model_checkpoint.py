from pathlib import Path

import pytest
import torch
from torch import nn

from models import registry
from models.registry import get_model
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_small_cnn_output_shape() -> None:
    model = get_model("small_cnn", num_classes=7)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = get_model("small_cnn", num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"
    original = {name: value.clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model, optimizer, 3, {"accuracy": 0.5}, {"seed": 42})
    for parameter in model.parameters():
        parameter.data.zero_()
    metadata = load_checkpoint(path, model, optimizer, device="cpu")
    assert metadata == {"epoch": 3, "metrics": {"accuracy": 0.5}, "config": {"seed": 42}}
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())


@pytest.mark.parametrize("name", ["resnet18", "resnet34"])
def test_resnet_factories_replace_classifier_and_forward_weights(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    received: list[object] = []

    class FakeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 1000)

    def factory(*, weights: object) -> FakeResNet:
        received.append(weights)
        return FakeResNet()

    monkeypatch.setattr(registry.models, name, factory)
    model = get_model(name.upper(), 3, pretrained=True)
    expected_weights = (
        registry.models.ResNet18_Weights.DEFAULT
        if name == "resnet18"
        else registry.models.ResNet34_Weights.DEFAULT
    )
    assert received == [expected_weights]
    assert model.fc.out_features == 3


def test_vgg_factory_replaces_last_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 1000))

    monkeypatch.setattr(registry.models, "vgg16", lambda *, weights: FakeVGG())
    model = get_model("vgg16", 6)
    assert model.classifier[-1].in_features == 5
    assert model.classifier[-1].out_features == 6


def test_model_registry_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("missing", 2)
    with pytest.raises(ValueError, match="num_classes"):
        get_model("small_cnn", 0)
    with pytest.raises(ValueError, match="pretrained weights"):
        get_model("small_cnn", 2, pretrained=True)


def test_checkpoint_without_optimizer_or_config(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(path, model, None, 0, {})
    metadata = load_checkpoint(path, model, device="cpu")
    assert metadata == {"epoch": 0, "metrics": {}, "config": None}


def test_checkpoint_load_errors(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing.pt", model, device="cpu")
    invalid = tmp_path / "invalid.pt"
    torch.save({"epoch": 1}, invalid)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        load_checkpoint(invalid, model, device="cpu")
