"""Model factory utilities for AS-OCT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet169_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    MobileNet_V2_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNeXt50_32X4D_Weights,
    VGG16_Weights,
)
from torchvision.models._api import WeightsEnum


@dataclass
class ModelSpec:
    """Specification describing how to instantiate and adapt a backbone."""

    builder: Callable[..., nn.Module]
    weights: Optional[WeightsEnum]
    head_fn: Callable[[nn.Module, int], None]


def _configure_resnet(model: nn.Module, num_classes: int) -> None:
    model.fc = nn.Linear(model.fc.in_features, num_classes)


def _configure_densenet(model: nn.Module, num_classes: int) -> None:
    classifier = model.classifier
    if isinstance(classifier, nn.Linear):
        in_features = classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise TypeError("Unexpected classifier type for DenseNet: %r" % type(classifier))


def _configure_efficientnet(model: nn.Module, num_classes: int) -> None:
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("EfficientNet classifier is expected to be nn.Sequential")
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected EfficientNet head: %r" % type(last_layer))
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _configure_vgg(model: nn.Module, num_classes: int) -> None:
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("VGG classifier is expected to be nn.Sequential")
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected VGG head: %r" % type(last_layer))
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _configure_convnext(model: nn.Module, num_classes: int) -> None:
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("ConvNeXt classifier is expected to be nn.Sequential")
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected ConvNeXt head: %r" % type(last_layer))
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _configure_mobilenet(model: nn.Module, num_classes: int) -> None:
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("MobileNet classifier is expected to be nn.Sequential")
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError("Unexpected MobileNet head: %r" % type(last_layer))
    in_features = last_layer.in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


SPECS: Dict[str, ModelSpec] = {
    "resnet34": ModelSpec(
        builder=models.resnet34,
        weights=ResNet34_Weights.IMAGENET1K_V1,
        head_fn=_configure_resnet,
    ),
    "resnet50": ModelSpec(
        builder=models.resnet50,
        weights=ResNet50_Weights.IMAGENET1K_V2,
        head_fn=_configure_resnet,
    ),
    "resnext50": ModelSpec(
        builder=models.resnext50_32x4d,
        weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        head_fn=_configure_resnet,
    ),
    "densenet169": ModelSpec(
        builder=models.densenet169,
        weights=DenseNet169_Weights.IMAGENET1K_V1,
        head_fn=_configure_densenet,
    ),
    "efficientnet_b3": ModelSpec(
        builder=models.efficientnet_b3,
        weights=EfficientNet_B3_Weights.IMAGENET1K_V1,
        head_fn=_configure_efficientnet,
    ),
    "efficientnet_b4": ModelSpec(
        builder=models.efficientnet_b4,
        weights=EfficientNet_B4_Weights.IMAGENET1K_V1,
        head_fn=_configure_efficientnet,
    ),
    "vgg16": ModelSpec(
        builder=models.vgg16,
        weights=VGG16_Weights.IMAGENET1K_V1,
        head_fn=_configure_vgg,
    ),
    "convnext_tiny": ModelSpec(
        builder=models.convnext_tiny,
        weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        head_fn=_configure_convnext,
    ),
    "mobilenet_v2": ModelSpec(
        builder=models.mobilenet_v2,
        weights=MobileNet_V2_Weights.IMAGENET1K_V2,
        head_fn=_configure_mobilenet,
    ),
}


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate a backbone by name and replace its classification head."""

    if name not in SPECS:
        raise KeyError(f"Unsupported model name: {name}")
    spec = SPECS[name]
    weights = spec.weights if pretrained else None
    model = spec.builder(weights=weights)
    spec.head_fn(model, num_classes)
    return model


def available_models() -> List[str]:
    """Return the identifiers of supported backbones."""

    return sorted(SPECS)
