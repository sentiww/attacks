from __future__ import annotations

from collections.abc import Callable

from torch import nn
from torchvision import models

from models.architectures.small_cnn import SmallCNN


def _resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet34(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _vgg16(num_classes: int, pretrained: bool) -> nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _small_cnn(num_classes: int, pretrained: bool) -> nn.Module:
    if pretrained:
        raise ValueError("small_cnn does not provide pretrained weights")
    return SmallCNN(num_classes)


MODEL_REGISTRY: dict[str, Callable[[int, bool], nn.Module]] = {
    "resnet18": _resnet18,
    "resnet34": _resnet34,
    "vgg16": _vgg16,
    "small_cnn": _small_cnn,
}


def get_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[name.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {choices}") from error
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return factory(num_classes, pretrained)
