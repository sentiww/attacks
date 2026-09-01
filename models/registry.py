from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn
from torchvision import models

from models.architectures.small_cnn import SmallCNN

SMALL_IMAGE_MAX_SIZE = 64
ModelFactory = Callable[..., nn.Module]


def _small_image(image_size: int) -> bool:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return image_size <= SMALL_IMAGE_MAX_SIZE


def _adapt_resnet_stem(model: nn.Module, image_size: int) -> None:
    if _small_image(image_size):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()


def _resnet18(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    if pretrained and _small_image(image_size):
        raise ValueError("Pretrained resnet18 requires the standard large-image stem")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    _adapt_resnet_stem(model, image_size)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet34(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    if pretrained and _small_image(image_size):
        raise ValueError("Pretrained resnet34 requires the standard large-image stem")
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    _adapt_resnet_stem(model, image_size)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet50(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    if pretrained and _small_image(image_size):
        raise ValueError("Pretrained resnet50 requires the standard large-image stem")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    _adapt_resnet_stem(model, image_size)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _convnext_tiny(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    if pretrained and _small_image(image_size):
        raise ValueError("Pretrained convnext_tiny requires the standard large-image stem")
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    )
    if _small_image(image_size):
        stem = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            stem.in_channels,
            stem.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=stem.bias is not None,
        )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _vit(
    num_classes: int,
    pretrained: bool,
    image_size: int,
    patch_size: int = 16,
    num_layers: int = 12,
    num_heads: int = 12,
    hidden_dim: int = 768,
    mlp_dim: int = 3072,
) -> nn.Module:
    if min(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim) <= 0:
        raise ValueError("ViT architecture values must be positive")
    if image_size % patch_size != 0:
        raise ValueError("ViT image_size must be positive and divisible by patch_size")
    if hidden_dim % num_heads != 0:
        raise ValueError("ViT hidden_dim must be divisible by num_heads")

    imagenet_b16 = (
        image_size == 224
        and patch_size == 16
        and num_layers == 12
        and num_heads == 12
        and hidden_dim == 768
        and mlp_dim == 3072
    )
    if pretrained:
        if not imagenet_b16:
            raise ValueError("Pretrained ViT is only available for the 224px ViT-B/16 configuration")
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    return models.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )


def _vgg16(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    _small_image(image_size)
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _small_cnn(num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    _small_image(image_size)
    if pretrained:
        raise ValueError("small_cnn does not provide pretrained weights")
    return SmallCNN(num_classes)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "resnet18": _resnet18,
    "resnet34": _resnet34,
    "resnet50": _resnet50,
    "convnext_tiny": _convnext_tiny,
    "vit": _vit,
    "vgg16": _vgg16,
    "small_cnn": _small_cnn,
}


def get_model(
    config: Mapping[str, Any],
    num_classes: int,
    image_size: int,
) -> nn.Module:
    if "name" not in config:
        raise ValueError("Missing model config key: name")
    name = str(config["name"])
    try:
        factory = MODEL_REGISTRY[name.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {choices}") from error
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    pretrained = bool(config.get("pretrained", False))
    model_options = {
        key: value for key, value in config.items() if key not in {"name", "pretrained"}
    }
    return factory(num_classes, pretrained, image_size, **model_options)
