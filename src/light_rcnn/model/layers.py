"""Efficient backbone adapters and bidirectional feature fusion blocks."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SeparableConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class MobileNetV3Pyramid(nn.Module):
    """Return C3/C4/C5 maps at strides 8/16/32."""

    channels = (40, 112, 960)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in (6, 12, 16):
                outputs.append(x)
        return tuple(outputs)  # type: ignore[return-value]


class BiFPNLayer(nn.Module):
    """One fast normalized bidirectional fusion pass over P3/P4/P5."""

    def __init__(self, channels: int):
        super().__init__()
        self.top_weights = nn.Parameter(torch.ones(2, 2))
        self.bottom_weights = nn.Parameter(torch.ones(2, 2))
        self.top_convs = nn.ModuleList(SeparableConv(channels) for _ in range(2))
        self.bottom_convs = nn.ModuleList(SeparableConv(channels) for _ in range(2))
        self.epsilon = 1e-4

    def _fuse(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = torch.relu(weights)
        weights = weights / (weights.sum() + self.epsilon)
        return weights[0] * a + weights[1] * b

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p3, p4, p5 = features
        p4_td = self.top_convs[0](self._fuse(p4, nn.functional.interpolate(p5, size=p4.shape[-2:], mode="nearest"), self.top_weights[0]))
        p3_out = self.top_convs[1](self._fuse(p3, nn.functional.interpolate(p4_td, size=p3.shape[-2:], mode="nearest"), self.top_weights[1]))
        p4_out = self.bottom_convs[0](self._fuse(p4_td, nn.functional.max_pool2d(p3_out, 2, ceil_mode=True), self.bottom_weights[0]))
        p5_out = self.bottom_convs[1](self._fuse(p5, nn.functional.max_pool2d(p4_out, 2, ceil_mode=True), self.bottom_weights[1]))
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], channels: int = 96, repeats: int = 2):
        super().__init__()
        self.projections = nn.ModuleList(ConvBNAct(source, channels) for source in in_channels)
        self.layers = nn.ModuleList(BiFPNLayer(channels) for _ in range(repeats))

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        projected = tuple(layer(value) for layer, value in zip(self.projections, features))
        for layer in self.layers:
            projected = layer(projected)
        return projected
