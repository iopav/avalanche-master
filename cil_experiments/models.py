from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


class TemporalBackbone(nn.Module):
    """Small three-layer image backbone used for lightweight pipeline runs."""

    feature_dim = 32

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.ReLU(inplace=False)

    def forward_with_feature_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], received {tuple(x.shape)}"
            )
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        return self.pool(x3).flatten(1), [x1, x2, x3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feature_maps(x)[0]


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CifarResNet18Backbone(nn.Module):
    """ResNet18 with a CIFAR stem and no classification head."""

    def __init__(self, in_channels: int = 3, base_width: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_width = int(base_width)
        if self.base_width <= 0:
            raise ValueError("base_width must be positive")
        channels = [self.base_width * (2**index) for index in range(4)]
        self.stage_channels = tuple(channels)
        self.feature_dim = channels[-1]
        self._current_channels = channels[0]
        self.conv1 = nn.Conv2d(
            self.in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(channels[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(channels[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(channels[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(channels[3], blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._reset_parameters()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(self._current_channels, channels, stride)]
        self._current_channels = channels
        layers.extend(_BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_with_feature_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], received {tuple(x.shape)}"
            )
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return self.pool(x4).flatten(1), [x1, x2, x3, x4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feature_maps(x)[0]


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    feature_dim: int
    pretrained: bool
    builder: Callable[[int], nn.Module]
    base_width: int
    stage_channels: tuple[int, ...]


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18_cifar": BackboneSpec(
        backbone_id="resnet18_cifar",
        feature_dim=512,
        pretrained=False,
        builder=lambda in_channels: CifarResNet18Backbone(in_channels, base_width=64),
        base_width=64,
        stage_channels=(64, 128, 256, 512),
    ),
    "temporal": BackboneSpec(
        backbone_id="temporal",
        feature_dim=32,
        pretrained=False,
        builder=TemporalBackbone,
        base_width=8,
        stage_channels=(8, 16, 32),
    ),
}


def build_backbone(backbone_id: str, input_shape: tuple[int, ...]) -> nn.Module:
    """Build a registered image backbone for an input shaped [C,H,W]."""

    try:
        spec = BACKBONES[backbone_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backbone {backbone_id!r}; registered={sorted(BACKBONES)}"
        ) from exc
    if len(input_shape) != 3 or any(int(value) <= 0 for value in input_shape):
        raise ValueError(f"Backbone input_shape must be [C,H,W], received {input_shape}")
    return spec.builder(int(input_shape[0]))


class SpikeImageAdapter(nn.Module):
    """Materialize a raw [B,64,400] binary spike sample as [B,3,160,160]."""

    output_shape = (3, 160, 160)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (64, 400):
            raise ValueError(f"Expected raw spike [B,64,400], received {tuple(x.shape)}")
        image = x.transpose(1, 2).reshape(x.shape[0], 1, 160, 160)
        return image.mul(255.0).repeat(1, 3, 1, 1)


class InputAdaptedBackbone(nn.Module):
    def __init__(self, adapter: nn.Module, backbone: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.feature_dim = int(backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adapter(x))


def build_feature_extractor(
    backbone_id: str,
    dataset_name: str,
    model_input_shape: tuple[int, ...],
) -> InputAdaptedBackbone:
    if dataset_name == "spike":
        adapter: nn.Module = SpikeImageAdapter()
        image_shape = SpikeImageAdapter.output_shape
    else:
        adapter = nn.Identity()
        image_shape = model_input_shape
    return InputAdaptedBackbone(adapter, build_backbone(backbone_id, image_shape))


class InputAdaptedFeatureMapBackbone(nn.Module):
    """TagFex-compatible view of a registered backbone."""

    def __init__(self, adapter: nn.Module, backbone: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.out_dim = int(backbone.feature_dim)
        self.feature_dim = self.out_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features, maps = self.backbone.forward_with_feature_maps(self.adapter(x))
        return {"features": features, "fmaps": maps}


def build_feature_map_extractor(
    backbone_id: str,
    dataset_name: str,
    model_input_shape: tuple[int, ...],
) -> InputAdaptedFeatureMapBackbone:
    if dataset_name == "spike":
        adapter: nn.Module = SpikeImageAdapter()
        image_shape = SpikeImageAdapter.output_shape
    else:
        adapter = nn.Identity()
        image_shape = model_input_shape
    return InputAdaptedFeatureMapBackbone(
        adapter, build_backbone(backbone_id, image_shape)
    )


class BackboneClassifier(nn.Module):
    def __init__(
        self,
        backbone_id: str,
        dataset_name: str,
        model_input_shape: tuple[int, ...],
    ):
        super().__init__()
        from avalanche.models import IncrementalClassifier

        self.feature_extractor = build_feature_extractor(
            backbone_id, dataset_name, model_input_shape
        )
        self.classifier = IncrementalClassifier(
            self.feature_extractor.feature_dim, initial_out_features=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))
