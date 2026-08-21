from __future__ import annotations

import torch
from torch import nn


class TemporalBackbone(nn.Module):
    feature_dim = 64

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.GroupNorm(8, 128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(128, self.feature_dim)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},T], received {tuple(x.shape)}")
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.projection(x)


class TemporalClassifier(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        from avalanche.models import IncrementalClassifier

        self.feature_extractor = TemporalBackbone(in_channels)
        self.classifier = IncrementalClassifier(64, initial_out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


def assert_shared_backbone_contract() -> None:
    models = {c: TemporalBackbone(c) for c in (64, 9, 3)}
    reference = models[64]
    for channels, model in models.items():
        assert model.conv1.in_channels == channels
        assert model.conv1.out_channels == reference.conv1.out_channels == 64
        assert model.conv1.kernel_size == reference.conv1.kernel_size == (7,)
        assert model.conv1.stride == reference.conv1.stride == (2,)
        ref_state = reference.state_dict()
        state = model.state_dict()
        for key in ref_state:
            if key == "conv1.weight":
                assert state[key].shape[0] == ref_state[key].shape[0]
                assert state[key].shape[2:] == ref_state[key].shape[2:]
            else:
                assert state[key].shape == ref_state[key].shape, (channels, key)
        out = model(torch.zeros(2, channels, 400 if channels == 64 else 315))
        assert out.shape == (2, 64)
