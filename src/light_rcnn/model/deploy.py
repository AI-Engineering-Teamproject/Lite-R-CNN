"""Inference-graph settings shared by scripts/benchmark.py and scripts/profile_stages.py."""

from __future__ import annotations

import itertools
import platform

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torchvision.ops.misc import FrozenBatchNorm2d


def fuse_conv_bn(model: torch.nn.Module) -> int:
    """Fold eval BatchNorm into the preceding Conv2d, as Ultralytics does to YOLO.

    Relies on registration order matching forward order (ConvBNAct, SeparableConv, the RoI
    projections, torchvision Conv2dNormActivation and ResNet Bottleneck/downsample).
    """
    fused = 0
    for module in list(model.modules()):
        names = list(module._modules)
        for prev, name in itertools.pairwise(names):
            conv, bn = module._modules[prev], module._modules[name]
            if not isinstance(conv, torch.nn.Conv2d):
                continue
            if isinstance(bn, torch.nn.BatchNorm2d):
                module._modules[prev] = fuse_conv_bn_eval(conv, bn)
            elif isinstance(bn, FrozenBatchNorm2d):  # torchvision COCO ResNet-50 baseline
                conv.weight, conv.bias = fuse_conv_bn_weights(
                    conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps,
                    bn.weight, bn.bias)
            else:
                continue
            module._modules[name] = torch.nn.Identity()
            fused += 1
    return fused


def use_channels_last(device: torch.device) -> bool:
    """Same auto rule as Ultralytics AutoBackend for native PyTorch models."""
    return (device.type == "cpu" and platform.system() in {"Linux", "Windows"}
            and platform.machine() in {"AMD64", "x86_64"}
            and torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled)


def prepare_for_inference(model: torch.nn.Module, device: torch.device,
                          channels_last: bool | None = None) -> dict:
    """Fuse Conv+BN and set the memory format (None = auto rule); returns the report fields."""
    fused = fuse_conv_bn(model)
    unfused = any(isinstance(module, (torch.nn.BatchNorm2d, FrozenBatchNorm2d))
                  for module in model.modules())
    if channels_last is None:
        channels_last = use_channels_last(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return {"conv_bn_fused": fused > 0 and not unfused,
            "memory_format": "channels_last" if channels_last else "contiguous"}
