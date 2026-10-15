"""Faster R-CNN reference and lightweight Faster R-CNN student builders."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign

from light_rcnn.model.layers import BiFPN, MobileNetV3Pyramid


class MobileBiFPNBackbone(nn.Module):
    """MobileNetV3 features fused into three same-width FPN levels for Faster R-CNN."""

    def __init__(self, channels: int = 96, pretrained: bool = True):
        super().__init__()
        self.body = MobileNetV3Pyramid(pretrained=pretrained)
        self.neck = BiFPN(self.body.channels, channels=channels)
        self.out_channels = channels

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.neck(self.body(images))
        return OrderedDict((str(index), feature) for index, feature in enumerate(features))


class CompressedMultiScaleRoIAlign(MultiScaleRoIAlign):
    """Project FPN maps to a narrow channel width before RoIAlign."""

    def __init__(self, in_channels: int, roi_channels: int = 10, output_size: int = 7):
        super().__init__(["0", "1", "2"], output_size=output_size, sampling_ratio=2)
        self.projections = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.Conv2d(in_channels, roi_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(roi_channels),
                    nn.ReLU(inplace=True),
                )
                for index in range(3)
            }
        )

    def forward(self, features, boxes, image_shapes):
        projected = {key: self.projections[key](features[key]) for key in self.projections}
        return super().forward(projected, boxes, image_shapes)


class GlobalAverageRoIHead(nn.Module):
    """Replace the conventional large two-FC RoI box head with GAP and a narrow vector."""

    def __init__(self, channels: int = 10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = channels

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.pool(pooled_features).flatten(start_dim=1)


class LightRCNN(FasterRCNN):
    """Faster R-CNN variant retaining its anchor-based RPN and RoI detection stages."""

    def __init__(
        self,
        num_classes: int = 91,
        pretrained_backbone: bool = True,
        bifpn_channels: int = 96,
        roi_channels: int = 10,
        roi_size: int = 7,
        rpn_pre_nms_topk_test: int = 1000,
        rpn_post_nms_topk_test: int = 300,
        min_size: int = 640,
        max_size: int = 640,
    ):
        backbone = MobileBiFPNBackbone(channels=bifpn_channels, pretrained=pretrained_backbone)
        anchors = AnchorGenerator(
            sizes=((32, 64), (64, 128), (128, 256)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3,
        )
        rpn_head = RPNHead(bifpn_channels, anchors.num_anchors_per_location()[0])
        roi_pool = CompressedMultiScaleRoIAlign(bifpn_channels, roi_channels, roi_size)
        box_head = GlobalAverageRoIHead(roi_channels)
        box_predictor = FastRCNNPredictor(roi_channels, num_classes)
        super().__init__(
            backbone,
            num_classes=None,
            rpn_anchor_generator=anchors,
            rpn_head=rpn_head,
            box_roi_pool=roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_test=rpn_pre_nms_topk_test,
            rpn_post_nms_top_n_test=rpn_post_nms_topk_test,
            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=1000,
            box_detections_per_img=100,
            box_batch_size_per_image=128,
        )


def build_vanilla_faster_rcnn(
    num_classes: int = 91,
    pretrained: bool = True,
    min_size: int = 640,
    max_size: int = 640,
) -> FasterRCNN:
    """Build the ResNet-50-FPN vanilla Faster R-CNN comparison baseline."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    return fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        num_classes=None if pretrained else num_classes,
        min_size=min_size,
        max_size=max_size,
    )


def build_detector(kind: str = "light_rcnn", **kwargs) -> nn.Module:
    if kind == "light_rcnn":
        return LightRCNN(**kwargs)
    if kind == "faster_rcnn":
        return build_vanilla_faster_rcnn(**kwargs)
    raise ValueError(f"Unknown detector kind: {kind}")


def light_rcnn_from_config(config: dict, *, pretrained: bool = False) -> LightRCNN:
    """Use the same architecture when training, validating and loading checkpoints."""
    model = config["model"]
    return LightRCNN(
        num_classes=config["num_classes"],
        pretrained_backbone=pretrained,
        bifpn_channels=model["bifpn_channels"],
        roi_channels=model["roi_channels"],
        roi_size=model["roi_size"],
        rpn_pre_nms_topk_test=model["rpn_pre_nms_topk_test"],
        rpn_post_nms_topk_test=model["rpn_post_nms_topk_test"],
        min_size=config["image_size"],
        max_size=config["image_size"],
    )
