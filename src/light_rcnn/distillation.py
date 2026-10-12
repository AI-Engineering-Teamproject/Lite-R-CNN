"""Loss primitives for feature and RoI-logit knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Spatially align teacher features and compare channel-normalized activations."""
    teacher = F.interpolate(teacher.detach(), size=student.shape[-2:], mode="bilinear", align_corners=False)
    if teacher.shape[1] != student.shape[1]:
        batch, _, height, width = teacher.shape
        teacher = F.adaptive_avg_pool1d(teacher.permute(0, 2, 3, 1).reshape(batch, height * width, -1),
                                        student.shape[1])
        teacher = teacher.reshape(batch, height, width, student.shape[1]).permute(0, 3, 1, 2)
    return F.smooth_l1_loss(F.normalize(student, dim=1), F.normalize(teacher, dim=1))


def roi_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Temperature-scaled KL loss; caller must align teacher/student proposals and classes."""
    teacher_logits = teacher_logits.detach()
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1), reduction="batchmean") * temperature**2
