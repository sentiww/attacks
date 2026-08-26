from pathlib import Path

import pytest
import torch
from torch import nn

from models import registry
from models.registry import get_model
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_small_cnn_output_shape() -> None:
    model = get_model({"name": "small_cnn"}, num_classes=7, image_size=32)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = get_model({"name": "small_cnn"}, num_classes=2, image_size=32)
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
    model = get_model({"name": name.upper(), "pretrained": True}, 3, image_size=224)
    expected_weights = (
        registry.models.ResNet18_Weights.DEFAULT
        if name == "resnet18"
        else registry.models.ResNet34_Weights.DEFAULT
    )
    assert received == [expected_weights]
    assert model.fc.out_features == 3


def test_cifar_resnet_replaces_large_image_stem(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.fc = nn.Linear(64, 1000)

    monkeypatch.setattr(registry.models, "resnet18", lambda *, weights: FakeResNet())
    model = get_model({"name": "resnet18"}, 10, image_size=32)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.fc.out_features == 10


def test_imagenet_resnet50_replaces_classifier_and_uses_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []

    class FakeResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 1000)

    def factory(*, weights: object) -> FakeResNet:
        received.append(weights)
        return FakeResNet()

    monkeypatch.setattr(registry.models, "resnet50", factory)
    model = get_model({"name": "resnet50", "pretrained": True}, 1000, image_size=224)
    assert received == [registry.models.ResNet50_Weights.DEFAULT]
    assert model.fc.out_features == 1000


def test_convnext_variants_replace_classifier_and_cifar_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []

    class FakeConvNeXt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Sequential(nn.Conv2d(3, 8, kernel_size=4, stride=4), nn.Identity())
            )
            self.classifier = nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(8, 1000))

    def factory(*, weights: object) -> FakeConvNeXt:
        received.append(weights)
        return FakeConvNeXt()

    monkeypatch.setattr(registry.models, "convnext_tiny", factory)
    imagenet_model = get_model(
        {"name": "convnext_tiny", "pretrained": True}, 1000, image_size=224
    )
    cifar_model = get_model({"name": "convnext_tiny"}, 10, image_size=32)
    assert received == [registry.models.ConvNeXt_Tiny_Weights.DEFAULT, None]
    assert imagenet_model.classifier[-1].out_features == 1000
    assert imagenet_model.features[0][0].stride == (4, 4)
    assert cifar_model.classifier[-1].out_features == 10
    assert cifar_model.features[0][0].kernel_size == (3, 3)
    assert cifar_model.features[0][0].stride == (1, 1)


def test_vit_variants_use_dataset_specific_image_and_patch_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cifar_arguments: dict[str, object] = {}
    imagenet_weights: list[object] = []

    class Heads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(16, 1000)

    class FakeViT(nn.Module):
        def __init__(self, num_classes: int = 1000) -> None:
            super().__init__()
            self.heads = Heads()
            self.heads.head = nn.Linear(16, num_classes)

    def vision_transformer(**kwargs: object) -> FakeViT:
        cifar_arguments.update(kwargs)
        return FakeViT(int(kwargs["num_classes"]))

    def vit_b_16(*, weights: object) -> FakeViT:
        imagenet_weights.append(weights)
        return FakeViT()

    monkeypatch.setattr(registry.models, "VisionTransformer", vision_transformer)
    monkeypatch.setattr(registry.models, "vit_b_16", vit_b_16)
    cifar_model = get_model(
        {
            "name": "vit",
            "patch_size": 4,
            "num_layers": 6,
            "num_heads": 6,
            "hidden_dim": 384,
            "mlp_dim": 1536,
        },
        10,
        image_size=32,
    )
    imagenet_model = get_model(
        {"name": "vit", "pretrained": True}, 7, image_size=224
    )
    assert cifar_arguments["image_size"] == 32
    assert cifar_arguments["patch_size"] == 4
    assert cifar_arguments["num_classes"] == 10
    assert cifar_model.heads.head.out_features == 10
    assert imagenet_weights == [registry.models.ViT_B_16_Weights.DEFAULT]
    assert imagenet_model.heads.head.out_features == 7


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("resnet18", {}),
        ("convnext_tiny", {}),
        (
            "vit",
            {
                "patch_size": 4,
                "num_layers": 6,
                "num_heads": 6,
                "hidden_dim": 384,
                "mlp_dim": 1536,
            },
        ),
    ],
)
def test_cifar_model_variants_accept_32_pixel_images(
    name: str, options: dict[str, int]
) -> None:
    model = get_model({"name": name, **options}, num_classes=10, image_size=32)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 10)


def test_vgg_factory_replaces_last_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 1000))

    monkeypatch.setattr(registry.models, "vgg16", lambda *, weights: FakeVGG())
    model = get_model({"name": "vgg16"}, 6, image_size=224)
    assert model.classifier[-1].in_features == 5
    assert model.classifier[-1].out_features == 6


def test_model_registry_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model({"name": "missing"}, 2, image_size=224)
    with pytest.raises(ValueError, match="Missing model config key"):
        get_model({}, 2, image_size=224)
    with pytest.raises(ValueError, match="num_classes"):
        get_model({"name": "small_cnn"}, 0, image_size=32)
    with pytest.raises(ValueError, match="pretrained weights"):
        get_model({"name": "small_cnn", "pretrained": True}, 2, image_size=32)
    with pytest.raises(ValueError, match="standard large-image stem"):
        get_model({"name": "resnet18", "pretrained": True}, 10, image_size=32)
    with pytest.raises(ValueError, match="standard large-image stem"):
        get_model({"name": "convnext_tiny", "pretrained": True}, 10, image_size=32)
    with pytest.raises(ValueError, match="only available"):
        get_model(
            {"name": "vit", "pretrained": True, "patch_size": 4},
            10,
            image_size=32,
        )


@pytest.mark.parametrize(
    ("image_size", "options"),
    [
        (0, {}),
        (32, {"patch_size": 0}),
        (30, {"patch_size": 4}),
        (32, {"patch_size": 4, "num_heads": 3, "hidden_dim": 128}),
    ],
)
def test_vit_rejects_invalid_architecture(
    image_size: int, options: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="ViT"):
        get_model({"name": "vit", **options}, 10, image_size=image_size)


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
