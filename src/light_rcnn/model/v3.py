"""Lite R-CNN v3: four-scale MobileNetV3/FPN with Faster R-CNN detection heads."""

from __future__ import annotations

from collections import OrderedDict

from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign

from light_rcnn.model.detector import LightRoIHead
from light_rcnn.model.layers import SeparableConv


class MobileNetV3PyramidV3(nn.Module):
    """Expose MobileNetV3 C2-C5 maps at strides 4, 8, 16 and 32."""

    feature_names = ("0", "1", "2", "3")
    channels = (24, 40, 112, 960)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs = []
        for index, layer in enumerate(self.features):
            images = layer(images)
            if index in (3, 6, 12, 16):
                outputs.append(images)
        return tuple(outputs)  # type: ignore[return-value]


class LiteFeaturePyramid(nn.Module):
    """Single-pass top-down FPN with separable output convolutions."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 64):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(source, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for source in in_channels
        )
        self.output = nn.ModuleList(SeparableConv(out_channels) for _ in in_channels)

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        inner = [layer(value) for layer, value in zip(self.lateral, features)]
        for index in range(len(inner) - 2, -1, -1):
            inner[index] = inner[index] + nn.functional.interpolate(
                inner[index + 1], size=inner[index].shape[-2:], mode="nearest"
            )
        return tuple(layer(value) for layer, value in zip(self.output, inner))


class MobileFPNBackboneV3(nn.Module):
    """MobileNetV3 C2-C5, a four-scale FPN and cheap P6/P7 RPN levels."""

    roi_feature_names = MobileNetV3PyramidV3.feature_names
    feature_names = ("0", "1", "2", "3", "4", "5")
    distillation_feature_names = roi_feature_names

    def __init__(self, channels: int = 64, pretrained: bool = True):
        super().__init__()
        self.body = MobileNetV3PyramidV3(pretrained=pretrained)
        self.neck = LiteFeaturePyramid(self.body.channels, out_channels=channels)
        self.out_channels = channels

    def forward(self, images: Tensor) -> OrderedDict[str, Tensor]:
        features = self.neck(self.body(images))
        p2, p3, p4, p5 = features
        p6 = nn.functional.max_pool2d(p5, kernel_size=1, stride=2)
        p7 = nn.functional.max_pool2d(p6, kernel_size=1, stride=2)
        return OrderedDict(zip(self.feature_names, (p2, p3, p4, p5, p6, p7)))


class LightRCNNV3(FasterRCNN):
    """Four-scale RoI FPN and six-level RPN Faster R-CNN with a compact head."""

    def __init__(
        self,
        num_classes: int = 91,
        pretrained_backbone: bool = True,
        fpn_channels: int = 64,
        roi_channels: int = 32,
        roi_size: int = 7,
        roi_head_dim: int = 256,
        roi_head_pool_size: int = 7,
        rpn_anchor_sizes: tuple[tuple[int, ...], ...] = (
            (8,), (16,), (32,), (64,), (128,), (256,)
        ),
        rpn_pre_nms_topk_test: int = 1000,
        rpn_post_nms_topk_test: int = 300,
        min_size: int = 640,
        max_size: int = 640,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        box_batch_size_per_image: int = 512,
    ):
        if len(rpn_anchor_sizes) != 6:
            raise ValueError("Lite R-CNN v3 needs one anchor-size group for each RPN level P2-P7")
        backbone = MobileFPNBackboneV3(fpn_channels, pretrained=pretrained_backbone)
        anchors = AnchorGenerator(
            sizes=rpn_anchor_sizes,
            aspect_ratios=((0.5, 1.0, 2.0),) * len(rpn_anchor_sizes),
        )
        roi_pool = MultiScaleRoIAlign(
            list(backbone.roi_feature_names), output_size=roi_size, sampling_ratio=2
        )
        box_head = LightRoIHead(
            roi_channels,
            roi_head_dim,
            roi_head_pool_size,
            input_channels=fpn_channels,
        )
        box_predictor = FastRCNNPredictor(box_head.out_channels, num_classes)
        super().__init__(
            backbone,
            num_classes=None,
            rpn_anchor_generator=anchors,
            rpn_head=RPNHead(fpn_channels, anchors.num_anchors_per_location()[0]),
            box_roi_pool=roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            min_size=min_size,
            max_size=max_size,
            rpn_pre_nms_top_n_test=rpn_pre_nms_topk_test,
            rpn_post_nms_top_n_test=rpn_post_nms_topk_test,
            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=1000,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
            box_batch_size_per_image=box_batch_size_per_image,
        )


def light_rcnn_v3_from_config(config: dict, *, pretrained: bool = False) -> LightRCNNV3:
    model = config["model"]
    train = config.get("train", {})
    anchor_sizes = tuple(tuple(int(size) for size in level) for level in model["rpn_anchor_sizes"])
    return LightRCNNV3(
        num_classes=config["num_classes"],
        pretrained_backbone=pretrained,
        fpn_channels=model.get("fpn_channels", 64),
        roi_channels=model.get("roi_channels", 32),
        roi_size=model.get("roi_size", 7),
        roi_head_dim=model.get("roi_head_dim", 256),
        roi_head_pool_size=model.get("roi_head_pool_size", 7),
        rpn_anchor_sizes=anchor_sizes,
        rpn_pre_nms_topk_test=model.get("rpn_pre_nms_topk_test", 1000),
        rpn_post_nms_topk_test=model.get("rpn_post_nms_topk_test", 300),
        min_size=config["image_size"],
        max_size=config["image_size"],
        box_score_thresh=model.get("box_score_thresh", 0.05),
        box_nms_thresh=model.get("box_nms_thresh", 0.5),
        box_detections_per_img=model.get("box_detections_per_img", 100),
        box_batch_size_per_image=train.get("box_batch_size_per_image", 512),
    )
