"""Loss primitives for feature and RoI-logit knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Spatially align teacher features and compare channel-normalized activations."""
    teacher = F.interpolate(
        teacher.detach(), size=student.shape[-2:], mode="bilinear", align_corners=False
    )
    if teacher.shape[1] != student.shape[1]:
        batch, _, height, width = teacher.shape
        teacher = F.adaptive_avg_pool1d(
            teacher.permute(0, 2, 3, 1).reshape(batch, height * width, -1), student.shape[1]
        )
        teacher = teacher.reshape(batch, height, width, student.shape[1]).permute(0, 3, 1, 2)
    return F.smooth_l1_loss(F.normalize(student, dim=1), F.normalize(teacher, dim=1))


def roi_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """Temperature-scaled KL loss; caller must align teacher/student proposals and classes."""
    teacher_logits = teacher_logits.detach()
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


class DistillationTrainer(torch.nn.Module):
    """Train on one transform and one set of sampled student RoIs.

    ResNet FPN levels 1/2/3 and student levels 0/1/2 have strides 8/16/32.
    Only the student and channel adapters receive gradients. The teacher uses
    the student's normalized, resized and padded tensor, never a second resize.
    """

    def __init__(self, student, teacher, feature_weight=1.0, roi_kl_weight=1.0, temperature=2.0):
        super().__init__()
        if feature_weight < 0 or roi_kl_weight < 0 or temperature <= 0:
            raise ValueError("KD weights must be nonnegative and temperature positive")
        if (
            student.roi_heads.box_predictor.cls_score.out_features
            != teacher.roi_heads.box_predictor.cls_score.out_features
        ):
            raise ValueError("Teacher and student must use identical class IDs and class count")
        self.student = student
        self.teacher = teacher.requires_grad_(False).eval()
        self.feature_weight = feature_weight
        self.roi_kl_weight = roi_kl_weight
        self.temperature = temperature
        self.adapters = (
            torch.nn.ModuleDict(
                {
                    str(index): torch.nn.Conv2d(
                        student.backbone.out_channels, teacher.backbone.out_channels, 1
                    )
                    for index in range(3)
                }
            )
            if feature_weight
            else torch.nn.ModuleDict()
        )

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images, targets):
        from torchvision.models.detection.roi_heads import fastrcnn_loss

        if not self.training or targets is None:
            raise ValueError("DistillationTrainer is a training-only wrapper")
        images, targets = self.student.transform(images, targets)
        for target in targets:
            boxes = target["boxes"]
            if (boxes[:, 2:] <= boxes[:, :2]).any():
                raise ValueError("Target boxes must have positive width and height")
        features = self.student.backbone(images.tensors)
        proposals, rpn_losses = self.student.rpn(images, features, targets)
        rois = self.student.roi_heads
        proposals, _, labels, regression_targets = rois.select_training_samples(proposals, targets)
        pooled = rois.box_roi_pool(features, proposals, images.image_sizes)
        student_logits, box_regression = rois.box_predictor(rois.box_head(pooled))
        classifier, box_loss = fastrcnn_loss(
            student_logits, box_regression, labels, regression_targets
        )
        losses = {**rpn_losses, "loss_classifier": classifier, "loss_box_reg": box_loss}
        if not (self.feature_weight or self.roi_kl_weight):
            return losses
        with torch.no_grad():
            teacher_features = self.teacher.backbone(images.tensors)
            if self.roi_kl_weight:
                teacher_rois = self.teacher.roi_heads
                teacher_pooled = teacher_rois.box_roi_pool(
                    teacher_features, proposals, images.image_sizes
                )
                teacher_logits, _ = teacher_rois.box_predictor(
                    teacher_rois.box_head(teacher_pooled)
                )
        if self.feature_weight:
            feature_losses = [
                feature_distillation_loss(
                    self.adapters[str(index)](features[str(index)]).float(),
                    teacher_features[str(index + 1)].float(),
                )
                for index in range(3)
            ]
            losses["loss_kd_feature"] = self.feature_weight * torch.stack(feature_losses).mean()
        if self.roi_kl_weight:
            losses["loss_kd_roi"] = self.roi_kl_weight * roi_kl_loss(
                student_logits.float(), teacher_logits.float(), self.temperature
            )
        return losses


def build_teacher(config):
    """Load COCO V1 weights or a local ResNet50-FPN checkpoint."""
    from light_rcnn.model.detector import build_vanilla_faster_rcnn

    cfg = config["distillation"]["teacher"]
    if (
        cfg.get("family", "faster_rcnn") != "faster_rcnn"
        or cfg.get("architecture", "fasterrcnn_resnet50_fpn") != "fasterrcnn_resnet50_fpn"
    ):
        raise ValueError("Distillation supports a Faster R-CNN ResNet50-FPN teacher")
    checkpoint = cfg["checkpoint"]
    pretrained = checkpoint == "torchvision_coco_v1"
    if pretrained and config["num_classes"] != 91:
        raise ValueError("The COCO teacher requires num_classes=91 and original COCO IDs")
    teacher = build_vanilla_faster_rcnn(
        num_classes=config["num_classes"],
        pretrained=pretrained,
        min_size=config["image_size"],
        max_size=config["image_size"],
    )
    if not pretrained:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        teacher.load_state_dict(state.get("model", state))
    return teacher
