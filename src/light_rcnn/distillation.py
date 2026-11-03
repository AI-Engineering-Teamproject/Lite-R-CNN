"""Loss primitives for feature, RoI-logit and RoI-box knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FEATURE_LOSSES = ("legacy", "standardized")


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Spatially align teacher features and compare channel-normalized activations.

    Legacy loss: Smooth L1 between unit-norm channel vectors equals (1 - cos) / C for C teacher
    channels, ~1e-3 for C=256 (the v3 run logged loss_kd_feature ~4e-4 at feature_weight 0.5).
    """
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


def _standardize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero mean, unit variance per channel over the batch and all positions."""
    flat = features.transpose(0, 1).flatten(1)
    mean = flat.mean(dim=1, keepdim=True)
    variance = flat.var(dim=1, unbiased=False, keepdim=True)
    return (flat - mean) * torch.rsqrt(variance + eps)


def standardized_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE / 2 between per-channel standardized maps (PKD, Cao et al. 2022).

    Equals the channel mean of 1 - Pearson r, so it is scale-free and lies in [0, 2]: ~1 for
    unrelated maps at the start of training and 0 when the adapted student map is an affine copy
    of the teacher map. The legacy loss is additionally divided by the channel count.
    """
    teacher = F.interpolate(
        teacher.detach(), size=student.shape[-2:], mode="bilinear", align_corners=False
    )
    if teacher.shape[1] != student.shape[1]:
        raise ValueError("Standardized feature KD needs adapters mapping to the teacher width")
    return F.mse_loss(_standardize(student), _standardize(teacher)) / 2


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


def roi_box_distillation_loss(
    student_deltas: torch.Tensor, teacher_deltas: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Smooth L1 (beta 1/9) from student to teacher box deltas of the same sampled RoIs.

    Each RoI uses the teacher's most likely foreground class and is weighted by that class's
    teacher probability, so background RoIs fade out while RoIs the GT matcher calls background
    but the teacher sees as objects still teach localization. Summed over RoIs and divided by
    the RoI count like fastrcnn_loss: box_weight=1 puts it on the scale of loss_box_reg.
    """
    rois = student_deltas.shape[0]
    if not rois:
        return student_deltas.sum() * 0
    probabilities = F.softmax(teacher_logits.detach(), dim=-1)
    weights, classes = probabilities[:, 1:].max(dim=1)
    index = torch.arange(rois, device=student_deltas.device)
    student = student_deltas.reshape(rois, -1, 4)[index, classes + 1]
    teacher = teacher_deltas.detach().reshape(rois, -1, 4)[index, classes + 1]
    loss = F.smooth_l1_loss(student, teacher, beta=1 / 9, reduction="none").sum(dim=1)
    return (weights * loss).sum() / rois


class DistillationTrainer(torch.nn.Module):
    """Train on one transform and one set of sampled student RoIs.

    The v2 three-level student maps to teacher strides 8/16/32; the v3 four-level
    student maps to teacher strides 4/8/16/32. Only the student and channel
    adapters receive gradients. The teacher uses
    the student's normalized, resized and padded tensor, never a second resize.
    ``feature_loss`` picks the imitation loss ("legacy" or "standardized") and ``box_weight``
    enables RoI box-delta distillation; the defaults reproduce the original trainer.
    """

    def __init__(
        self,
        student,
        teacher,
        feature_weight=1.0,
        roi_kl_weight=1.0,
        temperature=2.0,
        *,
        feature_loss="legacy",
        box_weight=0.0,
    ):
        super().__init__()
        if feature_weight < 0 or roi_kl_weight < 0 or box_weight < 0 or temperature <= 0:
            raise ValueError("KD weights must be nonnegative and temperature positive")
        if feature_loss not in FEATURE_LOSSES:
            raise ValueError(f"distillation.feature_loss must be one of {FEATURE_LOSSES}")
        if (
            student.roi_heads.box_predictor.cls_score.out_features
            != teacher.roi_heads.box_predictor.cls_score.out_features
        ):
            raise ValueError("Teacher and student must use identical class IDs and class count")
        if box_weight and (
            student.roi_heads.box_coder.weights != teacher.roi_heads.box_coder.weights
        ):
            raise ValueError("Box distillation needs identical box-coder weights")
        self.student = student
        self.teacher = teacher.requires_grad_(False).eval()
        self.feature_weight = feature_weight
        self.roi_kl_weight = roi_kl_weight
        self.temperature = temperature
        self.feature_loss = feature_loss
        self.box_weight = box_weight
        student_keys = tuple(
            getattr(
                student.backbone,
                "distillation_feature_names",
                getattr(student.backbone, "feature_names", ("0", "1", "2")),
            )
        )
        teacher_keys = ("0", "1", "2", "3") if len(student_keys) == 4 else ("1", "2", "3")
        self.feature_key_pairs = tuple(zip(student_keys, teacher_keys))
        self.adapters = (
            torch.nn.ModuleDict(
                {
                    student_key: torch.nn.Conv2d(
                        student.backbone.out_channels, teacher.backbone.out_channels, 1
                    )
                    for student_key, _ in self.feature_key_pairs
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
        if not (self.feature_weight or self.roi_kl_weight or self.box_weight):
            return losses
        with torch.no_grad():
            teacher_features = self.teacher.backbone(images.tensors)
            if self.roi_kl_weight or self.box_weight:
                teacher_rois = self.teacher.roi_heads
                teacher_pooled = teacher_rois.box_roi_pool(
                    teacher_features, proposals, images.image_sizes
                )
                teacher_logits, teacher_deltas = teacher_rois.box_predictor(
                    teacher_rois.box_head(teacher_pooled)
                )
        if self.feature_weight:
            imitation = (
                standardized_feature_loss
                if self.feature_loss == "standardized"
                else feature_distillation_loss
            )
            feature_losses = [
                imitation(
                    self.adapters[student_key](features[student_key]).float(),
                    teacher_features[teacher_key].float(),
                )
                for student_key, teacher_key in self.feature_key_pairs
            ]
            losses["loss_kd_feature"] = self.feature_weight * torch.stack(feature_losses).mean()
        if self.roi_kl_weight:
            losses["loss_kd_roi"] = self.roi_kl_weight * roi_kl_loss(
                student_logits.float(), teacher_logits.float(), self.temperature
            )
        if self.box_weight:
            losses["loss_kd_box"] = self.box_weight * roi_box_distillation_loss(
                box_regression.float(), teacher_deltas.float(), teacher_logits.float()
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
    try:
        teacher = build_vanilla_faster_rcnn(
            num_classes=config["num_classes"],
            pretrained=pretrained,
            min_size=config["image_size"],
            max_size=config["image_size"],
        )
    except OSError as error:  # urllib errors: offline or blocked download
        raise RuntimeError(
            "Could not download the torchvision COCO teacher weights; point "
            "distillation.teacher.checkpoint at a local ResNet50-FPN state dict"
        ) from error
    if not pretrained:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        teacher.load_state_dict(state.get("model", state))
    return teacher
