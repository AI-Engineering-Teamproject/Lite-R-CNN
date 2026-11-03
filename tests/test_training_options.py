"""Opt-in training options: v3.1 anchors, EMA, KD losses, multi-scale and CPU smoke training.

The smoke runs use the COCO smoke set when LIGHT_RCNN_SMOKE_COCO (or data/coco) holds
images/val2017 and annotations/instances_val2017.json, else a synthetic set; the teacher always
has random weights (no COCO download).
"""

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from light_rcnn.bench import load_comparison_model
from light_rcnn.data import enable_multiscale, multiscale_min_sizes
from light_rcnn.distillation import (
    DistillationTrainer,
    feature_distillation_loss,
    roi_box_distillation_loss,
    roi_kl_loss,
    standardized_feature_loss,
)
from light_rcnn.model.detector import (
    LightRCNN,
    LightRoIHead,
    build_vanilla_faster_rcnn,
    light_rcnn_from_config,
)
from light_rcnn.model.v3 import FixedStrideAnchorGenerator, LightRCNNV3, MobileFPNBackboneV3

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "147753808597_20161028.jpeg"
RATIOS = ((0.5, 1.0, 2.0),) * 6
V31_SIZES = ((11, 16, 23), (32, 45, 64), (64, 91, 128), (128, 181, 256), (256, 362, 512),
             (512, 724, 1024))
V31_STRIDES = (4, 8, 16, 32, 64, 128)
torch.set_num_threads(2)

_spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts/train.py")
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def load_config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def load_rgb(path, crop=None):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)
    return tensor if crop is None else tensor[:, :, crop].contiguous()


# --- v3.1 anchors -------------------------------------------------------------------------------


def pre_patch_v3(config):
    """LightRCNNV3 exactly as built before the v3.1 patch (stock AnchorGenerator)."""
    model, size = config["model"], config["image_size"]
    sizes = tuple(tuple(int(value) for value in level) for level in model["rpn_anchor_sizes"])
    backbone = MobileFPNBackboneV3(model["fpn_channels"], pretrained=False)
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    box_head = LightRoIHead(model["roi_channels"], model["roi_head_dim"],
                            model["roi_head_pool_size"], input_channels=model["fpn_channels"])
    return FasterRCNN(
        backbone,
        num_classes=None,
        rpn_anchor_generator=anchors,
        rpn_head=RPNHead(model["fpn_channels"], anchors.num_anchors_per_location()[0]),
        box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=model["roi_size"],
                                        sampling_ratio=2),
        box_head=box_head,
        box_predictor=FastRCNNPredictor(box_head.out_channels, config["num_classes"]),
        min_size=size,
        max_size=size,
        rpn_pre_nms_top_n_test=model["rpn_pre_nms_topk_test"],
        rpn_post_nms_top_n_test=model["rpn_post_nms_topk_test"],
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        box_score_thresh=model["box_score_thresh"],
        box_nms_thresh=model["box_nms_thresh"],
        box_detections_per_img=model["box_detections_per_img"],
        box_batch_size_per_image=config["train"]["box_batch_size_per_image"],
    )


def test_v3_checkpoint_is_bit_identical_to_the_pre_patch_model():
    weights = ROOT / "outputs/coco_4090_v3/best.pth"
    if not weights.is_file() or not PHOTO.is_file():
        pytest.skip("v3 checkpoint or test photo is not in the work tree")
    model = load_comparison_model("light_rcnn", load_config("coco.yaml"), str(weights)).eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    assert "rpn_anchor_strides" not in checkpoint["config"]["model"]
    assert type(model.rpn.anchor_generator) is AnchorGenerator
    assert model.rpn.anchor_generator.num_anchors_per_location() == [3] * 6
    reference = pre_patch_v3(checkpoint["config"]).eval()
    reference.load_state_dict(checkpoint["model"])
    assert model.state_dict().keys() == reference.state_dict().keys()
    photo = load_rgb(PHOTO)  # 970x646 -> 640x426, padded to 448x640
    with torch.inference_mode():
        for image in (photo, photo[:, :, 150:750].contiguous()):
            expected, actual = reference([image])[0], model([image])[0]
            assert len(actual["boxes"])
            for key in ("boxes", "scores", "labels"):
                assert torch.equal(actual[key], expected[key]), key


def v3_feature_maps(height, width, device="cpu"):
    shapes = [(height // stride, width // stride) for stride in (4, 8, 16, 32)]
    for _ in range(2):  # P6/P7: kernel-1 stride-2 max pooling of the level above
        shapes.append(tuple(math.ceil(side / 2) for side in shapes[-1]))
    return [torch.zeros(1, 4, *shape, device=device) for shape in shapes]


def test_fixed_strides_equal_stock_on_square_canvas_and_realign_p6_p7():
    fixed = FixedStrideAnchorGenerator(V31_SIZES, RATIOS, V31_STRIDES)
    stock = AnchorGenerator(V31_SIZES, RATIOS)
    square = v3_feature_maps(640, 640)
    images = ImageList(torch.zeros(1, 3, 640, 640), [(640, 640)])
    assert torch.equal(fixed(images, square)[0], stock(images, square)[0])

    maps = v3_feature_maps(480, 640)  # P6 8x10, P7 4x5: derived y-strides 60/120
    images = ImageList(torch.zeros(1, 3, 480, 640), [(480, 640)])
    counts = [9 * feature.shape[-2] * feature.shape[-1] for feature in maps]
    fixed_levels = fixed(images, maps)[0].split(counts)
    stock_levels = stock(images, maps)[0].split(counts)
    for level in range(4):
        assert torch.equal(fixed_levels[level], stock_levels[level])
    for level, stride in ((4, 64), (5, 128)):
        assert not torch.equal(fixed_levels[level], stock_levels[level])
        centers_y = fixed_levels[level][:, [1, 3]].mean(dim=1).unique()
        assert torch.equal(centers_y, torch.arange(maps[level].shape[-2]).float() * stride)
    with pytest.raises(ValueError, match="One stride per anchor level"):
        FixedStrideAnchorGenerator(V31_SIZES, RATIOS, V31_STRIDES[:5])


class HostToDeviceCopies(TorchDispatchMode):
    """Counts CPU -> CUDA copies: blocking transfers (a stream sync each) on a real GPU."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if (func is torch.ops.aten._to_copy.default and device is not None
                and args[0].device.type == "cpu" and torch.device(device).type == "cuda"):
            self.count += 1
        return func(*args, **kwargs)


def host_to_device_copies(generator, forwards=2):
    """CPU -> CUDA copies per forward on fake CUDA tensors.

    Without a GPU, FakeTensorMode also routes torch.tensor(..., device="cuda") through a visible
    copy; with one, the sync debug-mode test below covers that path.
    """
    counts = []
    with FakeTensorMode(allow_non_fake_inputs=True):
        maps = v3_feature_maps(480, 640, "cuda")
        images = ImageList(torch.zeros(1, 3, 480, 640, device="cuda"), [(480, 640)])
        for _ in range(forwards):
            with HostToDeviceCopies() as copies:
                anchors = generator(images, maps)[0]
            assert anchors.device.type == "cuda"
            counts.append(copies.count)
    return counts


def test_fixed_strides_add_no_host_to_device_copy_per_forward():
    # Only the first forward copies (the 6 cell anchors), as in the stock generator; strides made
    # with torch.tensor(stride, device=cuda) would add 6 blocking copies to every forward.
    stock = AnchorGenerator(V31_SIZES, RATIOS)
    fixed = FixedStrideAnchorGenerator(V31_SIZES, RATIOS, V31_STRIDES)
    assert host_to_device_copies(fixed) == host_to_device_copies(stock) == [6, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fixed_strides_do_not_synchronize_cuda():
    fixed = FixedStrideAnchorGenerator(V31_SIZES, RATIOS, V31_STRIDES)
    images = ImageList(torch.zeros(1, 3, 480, 640), [(480, 640)])
    expected = fixed(images, v3_feature_maps(480, 640))[0]
    maps = v3_feature_maps(480, 640, "cuda")
    fixed(images, maps)  # moves the cell anchors to the GPU once
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        anchors = fixed(images, maps)[0]
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert torch.equal(anchors.cpu(), expected)


@pytest.mark.parametrize("name", ["coco_v3_1a.yaml", "coco_v3_1b.yaml", "coco_v3_1a_ema.yaml"])
def test_v3_1_configs_build_nine_anchors_per_location(name):
    config = load_config(name)
    model = light_rcnn_from_config(config)
    generator = model.rpn.anchor_generator
    assert isinstance(generator, FixedStrideAnchorGenerator)
    assert generator.fixed_strides == V31_STRIDES
    assert generator.num_anchors_per_location() == [9] * 6
    assert model.rpn.head.cls_logits.out_channels == 9
    assert model.rpn.head.bbox_pred.out_channels == 36
    with pytest.raises(ValueError, match="same anchor count"):
        LightRCNNV3(pretrained_backbone=False, rpn_anchor_sizes=((8,),) * 5 + ((16, 32),))


def test_ablation_configs_use_own_output_dirs_and_ema_config_is_a1_plus_ema():
    configs = {path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
               for path in sorted((ROOT / "configs").glob("*.yaml"))}
    output_dirs = [config["train"]["output_dir"] for config in configs.values()]
    assert len(output_dirs) == len(set(output_dirs))
    for name in ("coco_v3_short.yaml", "coco_v3_1a.yaml", "coco_v3_1b.yaml",
                 "coco_v3_1a_nokd.yaml", "coco_v3_1.yaml"):
        assert train_script.ema_settings(configs[name]["train"]) is None
        assert train_script.multiscale_settings(configs[name]["train"], 640) is None
    a1, ema = configs["coco_v3_1a.yaml"], configs["coco_v3_1a_ema.yaml"]
    assert {key: value for key, value in ema.items() if key != "train"} == {
        key: value for key, value in a1.items() if key != "train"
    }
    differing = {key for key in a1["train"].keys() | ema["train"].keys()
                 if a1["train"].get(key) != ema["train"].get(key)}
    assert differing == {"ema", "output_dir"}
    assert train_script.ema_settings(ema["train"]) == {"decay": 0.9998, "warmup": 2000}


# --- EMA ----------------------------------------------------------------------------------------


def small_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6))


def perturb(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator))
    model.train()(torch.randn(16, 4, generator=generator))  # moves the BatchNorm statistics


def float_state(module):
    return {key: value.clone() for key, value in module.state_dict().items()
            if value.is_floating_point()}


def test_ema_averages_parameters_and_buffers():
    model = small_model()
    ema = train_script.ModelEMA(model, decay=0.75, warmup=0)
    perturb(model, 1)
    ema.update(model)  # first update copies
    assert all(torch.equal(value, float_state(model)[key])
               for key, value in float_state(ema.module).items())
    before = float_state(ema.module)
    perturb(model, 2)
    ema.update(model)
    current = float_state(model)
    for key, value in float_state(ema.module).items():
        torch.testing.assert_close(value, 0.75 * before[key] + 0.25 * current[key])
    assert "1.running_mean" in before  # BatchNorm statistics are averaged too
    assert ema.updates == 2
    assert not any(parameter.requires_grad for parameter in ema.module.parameters())


def test_ema_warmup_ramps_the_decay():
    ema = train_script.ModelEMA(small_model(), decay=0.9998, warmup=2000)
    model = small_model()
    for step in range(3):
        ema.update(model)
        assert ema.current_decay() == pytest.approx(0.9998 * (1 - math.exp(-(step + 1) / 2000)))
    assert train_script.ModelEMA(small_model(), decay=0.99, warmup=0).current_decay() == 0.99


def test_ema_state_round_trip_continues_identically():
    model = small_model()
    ema = train_script.ModelEMA(model, decay=0.9, warmup=3)
    for seed in range(3):
        perturb(model, seed)
        ema.update(model)
    restored = train_script.ModelEMA(small_model(), decay=0.9, warmup=3)
    restored.load_state_dict(ema.module.state_dict(), ema.state_dict())
    assert restored.updates == ema.updates == 3
    for seed in range(3, 5):
        perturb(model, seed)
        ema.update(model)
        restored.update(model)
    for key, value in ema.module.state_dict().items():
        assert torch.equal(value, restored.module.state_dict()[key]), key


def test_ema_fallback_without_multi_avg_fn_matches(monkeypatch):
    results = []
    for multi in (train_script.get_ema_multi_avg_fn, None):
        monkeypatch.setattr(train_script, "get_ema_multi_avg_fn", multi)
        model = small_model()
        ema = train_script.ModelEMA(model, decay=0.8, warmup=2)
        for seed in range(3):
            perturb(model, seed)
            ema.update(model)
        results.append(float_state(ema.module))
    for key, value in results[0].items():
        torch.testing.assert_close(results[1][key], value, rtol=1e-6, atol=1e-6)


def test_option_settings_treat_absent_and_disabled_alike():
    assert train_script.ema_settings({}) is None
    assert train_script.ema_settings({"ema": {"enabled": False, "decay": 0.5}}) is None
    assert train_script.ema_settings({"ema": {"enabled": True}}) == {"decay": 0.9998,
                                                                      "warmup": 2000}
    assert train_script.ema_settings({"ema": True}) == train_script.ema_settings(
        {"ema": {"enabled": True}})
    with pytest.raises(ValueError, match="train.ema"):
        train_script.ema_settings({"ema": {"enabled": True, "decay": 1.0}})
    assert train_script.multiscale_settings({"multiscale": {"enabled": False}}, 640) is None
    assert train_script.multiscale_settings(
        {"multiscale": {"enabled": True, "min_sizes": [640, 480]}}, 640
    ) == {"min_sizes": [480, 640], "max_size": 640}


# --- distillation -------------------------------------------------------------------------------


def kd_batch():
    return (
        [torch.rand(3, 64, 80), torch.rand(3, 80, 64)],
        [{"boxes": torch.tensor([[4.0, 6.0, 40.0, 42.0], [30.0, 20.0, 60.0, 60.0]]),
          "labels": torch.tensor([1, 2])} for _ in range(2)],
    )


def kd_student():
    torch.manual_seed(0)
    return LightRCNN(pretrained_backbone=False, num_classes=3, bifpn_channels=16, roi_channels=8,
                     min_size=64, max_size=64, rpn_pre_nms_topk_test=20, rpn_post_nms_topk_test=10)


def kd_teacher():
    torch.manual_seed(1)
    return build_vanilla_faster_rcnn(num_classes=3, pretrained=False, min_size=64, max_size=64)


def original_feature_loss(student, teacher):
    """distillation.feature_distillation_loss before the opt-in losses were added."""
    teacher = F.interpolate(teacher.detach(), size=student.shape[-2:], mode="bilinear",
                            align_corners=False)
    return F.smooth_l1_loss(F.normalize(student, dim=1), F.normalize(teacher, dim=1))


def original_forward(trainer, images, targets):
    """DistillationTrainer.forward before the opt-in losses were added."""
    from torchvision.models.detection.roi_heads import fastrcnn_loss

    images, targets = trainer.student.transform(images, targets)
    features = trainer.student.backbone(images.tensors)
    proposals, rpn_losses = trainer.student.rpn(images, features, targets)
    rois = trainer.student.roi_heads
    proposals, _, labels, regression_targets = rois.select_training_samples(proposals, targets)
    pooled = rois.box_roi_pool(features, proposals, images.image_sizes)
    student_logits, box_regression = rois.box_predictor(rois.box_head(pooled))
    classifier, box_loss = fastrcnn_loss(student_logits, box_regression, labels,
                                         regression_targets)
    losses = {**rpn_losses, "loss_classifier": classifier, "loss_box_reg": box_loss}
    with torch.no_grad():
        teacher_features = trainer.teacher.backbone(images.tensors)
        teacher_rois = trainer.teacher.roi_heads
        teacher_pooled = teacher_rois.box_roi_pool(teacher_features, proposals,
                                                   images.image_sizes)
        teacher_logits, _ = teacher_rois.box_predictor(teacher_rois.box_head(teacher_pooled))
    feature_losses = [
        original_feature_loss(trainer.adapters[s](features[s]).float(),
                              teacher_features[t].float())
        for s, t in trainer.feature_key_pairs
    ]
    losses["loss_kd_feature"] = trainer.feature_weight * torch.stack(feature_losses).mean()
    losses["loss_kd_roi"] = trainer.roi_kl_weight * roi_kl_loss(
        student_logits.float(), teacher_logits.float(), trainer.temperature)
    return losses


def test_default_distillation_is_the_original_trainer():
    trainer = DistillationTrainer(kd_student(), kd_teacher(), 0.5, 1.0, 2.0).train()
    assert (trainer.feature_loss, trainer.box_weight) == ("legacy", 0.0)
    images, targets = kd_batch()
    torch.manual_seed(7)
    actual = trainer(images, targets)
    torch.manual_seed(7)
    expected = original_forward(trainer, images, targets)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
    student, teacher = torch.randn(2, 8, 6, 5), torch.randn(2, 8, 12, 10)
    assert torch.equal(feature_distillation_loss(student, teacher),
                       original_feature_loss(student, teacher))


def test_standardized_feature_loss_is_scale_free_and_not_negligible():
    torch.manual_seed(0)
    teacher = torch.randn(2, 256, 10, 12)
    unrelated = torch.randn(2, 256, 10, 12)
    scale = torch.rand(1, 256, 1, 1) + 0.5
    assert float(standardized_feature_loss(teacher.clone(), teacher)) < 1e-5
    assert float(standardized_feature_loss(3 * scale * teacher + 2, teacher)) < 1e-5
    assert float(standardized_feature_loss(-teacher, teacher)) == pytest.approx(2.0, abs=1e-4)
    loss = standardized_feature_loss(unrelated, teacher)
    assert float(loss) == pytest.approx(1.0, abs=0.05)
    assert float(feature_distillation_loss(unrelated, teacher)) < 0.01  # (1 - cos) / 256
    student = unrelated.clone().requires_grad_(True)
    standardized_feature_loss(student * 100, teacher).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0


def test_box_distillation_uses_teacher_class_and_confidence():
    student = torch.zeros(2, 3 * 4)
    teacher = torch.zeros(2, 3 * 4)
    teacher[0, 8:12] = 1.0  # RoI 0: class 2 deltas
    teacher[1, 4:8] = 1.0  # RoI 1: class 1 deltas, but the teacher sees background
    logits = torch.tensor([[0.0, 0.0, 20.0], [20.0, 0.0, 0.0]])
    weights = F.softmax(logits, dim=-1)[:, 1:].max(dim=1).values
    per_coordinate = 1.0 - 0.5 / 9  # smooth L1, beta 1/9, |d| = 1
    expected = (weights[0] * 4 * per_coordinate + weights[1] * 4 * per_coordinate) / 2
    torch.testing.assert_close(roi_box_distillation_loss(student, teacher, logits), expected)
    assert float(roi_box_distillation_loss(teacher, teacher, logits)) == 0.0
    empty = roi_box_distillation_loss(torch.zeros(0, 12), torch.zeros(0, 12), torch.zeros(0, 3))
    assert float(empty) == 0.0


@pytest.mark.parametrize("feature_loss", ["legacy", "standardized"])
def test_opt_in_kd_losses_are_finite_and_train_the_student(feature_loss):
    images, targets = kd_batch()
    losses = {}
    for weight in (1.0, 2.0):
        trainer = DistillationTrainer(kd_student(), kd_teacher(), 0.5, 1.0, 2.0,
                                      feature_loss=feature_loss, box_weight=weight).train()
        torch.manual_seed(3)
        losses[weight] = trainer(images, targets)
    assert {"loss_kd_feature", "loss_kd_roi", "loss_kd_box"} <= losses[1.0].keys()
    assert all(torch.isfinite(value) for value in losses[1.0].values())
    torch.testing.assert_close(losses[2.0]["loss_kd_box"], 2 * losses[1.0]["loss_kd_box"])
    if feature_loss == "standardized":
        assert float(losses[1.0]["loss_kd_feature"].detach()) > 0.1  # legacy: ~1e-3
    sum(losses[2.0].values()).backward()
    student = trainer.student.roi_heads.box_predictor
    assert student.bbox_pred.weight.grad.abs().sum() > 0
    assert trainer.adapters["0"].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in trainer.teacher.parameters())
    with pytest.raises(ValueError, match="feature_loss"):
        DistillationTrainer(kd_student(), kd_teacher(), feature_loss="cosine")
    with pytest.raises(ValueError, match="nonnegative"):
        DistillationTrainer(kd_student(), kd_teacher(), box_weight=-1.0)


def test_v3_student_distills_all_four_levels_with_new_losses():
    torch.manual_seed(0)
    student = LightRCNNV3(num_classes=3, pretrained_backbone=False, fpn_channels=16,
                          roi_channels=8, rpn_anchor_sizes=V31_SIZES,
                          rpn_anchor_strides=V31_STRIDES, min_size=64, max_size=64)
    trainer = DistillationTrainer(student, kd_teacher(), 0.5, 1.0, 2.0,
                                  feature_loss="standardized", box_weight=1.0).train()
    assert [pair[1] for pair in trainer.feature_key_pairs] == ["0", "1", "2", "3"]
    losses = trainer(*kd_batch())
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert all(adapter.weight.grad.abs().sum() > 0 for adapter in trainer.adapters.values())


# --- multi-scale --------------------------------------------------------------------------------


def transform(size=640):
    return GeneralizedRCNNTransform(size, size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def test_multiscale_resizes_within_range_and_keeps_validation_fixed():
    sizes = (384, 512, 640)
    torch.manual_seed(0)
    model = enable_multiscale(transform(), [640, 384, 512], 640).train()
    assert model.min_size == sizes
    for height, width in ((480, 640), (640, 640), (427, 640), (640, 481)):
        image = torch.rand(3, height, width)
        target = {"boxes": torch.tensor([[0.0, 0.0, width, height]]),
                  "labels": torch.tensor([1])}
        expected = set()
        for size in sizes:
            scale = min(size / min(height, width), 640 / max(height, width))
            expected.add((math.floor(height * scale), math.floor(width * scale)))
        seen = set()
        for _ in range(30):
            images, targets = model([image], [dict(target)])
            resized = tuple(images.image_sizes[0])
            assert resized in expected and max(resized) <= 640
            torch.testing.assert_close(targets[0]["boxes"][0, 2:],
                                       torch.tensor([resized[1], resized[0]]).float())
            seen.add(resized)
        if height == width:
            assert len(seen) == len(sizes)  # every short side is drawn
        reference = transform().eval()
        assert torch.equal(model.eval()([image])[0].tensors, reference([image])[0].tensors)
        model.train()


def test_multiscale_rejects_settings_that_would_move_validation():
    assert multiscale_min_sizes([512, 640, 384], 640, 640) == (384, 512, 640)
    with pytest.raises(ValueError, match="max_size"):
        multiscale_min_sizes([512, 640], 704, 640)
    with pytest.raises(ValueError, match="include 640"):
        multiscale_min_sizes([480, 512], 640, 640)
    with pytest.raises(ValueError, match="include 640"):
        multiscale_min_sizes([16, 640], 640, 640)
    with pytest.raises(ValueError, match="max_size"):
        enable_multiscale(transform(), [480, 640], 800)


# --- CPU smoke training -------------------------------------------------------------------------


def synthetic_coco(root, count=8):
    images = root / "images"
    images.mkdir(parents=True)
    generator = np.random.default_rng(0)
    records, annotations = [], []
    for index in range(count):
        width, height = ((200, 150), (150, 200), (160, 160), (240, 160))[index % 4]
        array = generator.integers(0, 60, (height, width, 3), dtype=np.uint8)
        for box_index in range(1 + index % 2):
            x, y = 10 + 40 * box_index, 12 + 30 * box_index
            array[y:y + 60, x:x + 70] = (200, 40 + 80 * box_index, 60)
            annotations.append({"id": len(annotations) + 1, "image_id": index + 1,
                                "category_id": (1, 3)[box_index], "bbox": [x, y, 70, 60],
                                "area": 4200, "iscrowd": 0})
        name = f"{index + 1:012d}.jpg"
        Image.fromarray(array).save(images / name)
        records.append({"id": index + 1, "file_name": name, "width": width, "height": height})
    path = root / "instances.json"
    path.write_text(json.dumps({
        "info": {}, "images": records, "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
    }))
    return images, path


@pytest.fixture(scope="module")
def smoke_files(tmp_path_factory):
    """(images, annotations, teacher checkpoint) for the smoke runs."""
    root = tmp_path_factory.mktemp("smoke")
    coco = Path(os.environ.get("LIGHT_RCNN_SMOKE_COCO", ROOT / "data/coco"))
    images = coco / "images/val2017"
    annotations = coco / "annotations/instances_val2017.json"
    if not (images.is_dir() and annotations.is_file()):
        images, annotations = synthetic_coco(root / "coco")
    teacher = root / "teacher.pth"  # random weights: the COCO teacher is a download
    torch.save(build_vanilla_faster_rcnn(num_classes=91, pretrained=False).state_dict(), teacher)
    return images, annotations, teacher


def smoke_config(smoke_files, tmp_path, image_size=640):
    """configs/coco_v3_1a.yaml (A1, legacy KD) on the CPU with smoke data and a random teacher."""
    images, annotations, teacher = smoke_files
    config = load_config("coco_v3_1a.yaml")
    config.update(device="cpu", image_size=image_size)
    config["data"] = {"train_images": str(images), "train_annotations": str(annotations),
                      "val_images": str(images), "val_annotations": str(annotations)}
    config["model"]["pretrained"] = False
    config["train"].update(workers=0, amp=False, batch_size=4, val_batch_size=4,
                           output_dir=str(tmp_path / "run"))
    config["distillation"]["teacher"]["checkpoint"] = str(teacher)
    return config


def run_train(tmp_path, *args, config=None, check=True):
    """scripts/train.py with ``config`` written to a file, or the checkpoint's saved config."""
    command = [sys.executable, "scripts/train.py", *map(str, args)]
    if config is not None:
        path = tmp_path / f"config_{len(list(tmp_path.glob('config_*.yaml')))}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        command += ["--config", str(path)]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"),
                                                        os.environ.get("PYTHONPATH")])))
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True,
                            timeout=600, check=False)
    if check:
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


OPTIONS = {
    "ema": {"train": {"ema": {"enabled": True, "decay": 0.5, "warmup": 0}}},
    "standardized_feature_kd": {"distillation": {"feature_loss": "standardized"}},
    "box_kd": {"distillation": {"box_weight": 1.0}},
    "multiscale": {"train": {"multiscale": {"enabled": True, "min_sizes": [384, 512, 640],
                                            "max_size": 640}}},
}


def with_options(config, *names):
    config = copy.deepcopy(config)
    for name in names:
        for section, values in OPTIONS[name].items():
            config[section].update(copy.deepcopy(values))
    return config


@pytest.mark.parametrize("option", list(OPTIONS))
def test_smoke_training_runs_with_each_option(option, smoke_files, tmp_path):
    config = with_options(smoke_config(smoke_files, tmp_path), option)
    output = tmp_path / option
    run_train(tmp_path, "--epochs", 1, "--train-limit", 8, "--val-limit", 4,
              "--output-dir", output, config=config)
    last = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
    row = json.loads((output / "history.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["epoch"] == 1 and last["global_step"] == 2
    assert row["validation"]["images_evaluated"] == 4
    assert all(math.isfinite(value) for value in row["losses"].values())
    assert isinstance(last["config"]["model"]["rpn_anchor_strides"], list)
    results = (output / "RESULTS.md").read_text(encoding="utf-8")
    assert "- 학습 옵션: " in results
    if option == "ema":
        assert last["format_version"] == 2 and last["model_weights"] == "ema"
        assert row["validated_weights"] == "ema" and last["ema"]["updates"] == 2
        assert any(not torch.equal(value, last["train_model"][key])
                   for key, value in last["model"].items() if value.is_floating_point())
        best = torch.load(output / "best.pth", map_location="cpu", weights_only=True)
        assert best["model_weights"] == "ema"
    else:  # format 2 all the same: an older train.py would drop the option on resume
        assert last["format_version"] == 2 and last["model_weights"] == "raw"
        assert "train_model" not in last and row["validated_weights"] == "raw"
    if option == "box_kd":
        assert row["losses"]["loss_kd_box"] > 0
    if option == "standardized_feature_kd":
        assert row["losses"]["loss_kd_feature"] > 0.05  # legacy: ~1e-3 x feature_weight
    if option == "multiscale":
        assert "다중 스케일" in results


@pytest.mark.parametrize("strides", [True, False], ids=["anchor_strides", "default"])
def test_only_default_runs_save_format_1(strides, smoke_files, tmp_path):
    """An older train.py resumes format 1 only and ignores rpn_anchor_strides (A1, A2, v3.1)."""
    files = synthetic_coco(tmp_path / "coco")
    config = smoke_config((*files, smoke_files[2]), tmp_path, image_size=128)
    if not strides:  # the configs/coco_v3.yaml model: one anchor size per level, no strides
        config["model"] = {**load_config("coco_v3.yaml")["model"], "pretrained": False}
    run_train(tmp_path, "--epochs", 1, config=config)
    last = torch.load(tmp_path / "run/last.pth", map_location="cpu", weights_only=True)
    assert train_script.training_options(last["config"]) == []
    assert last["config"]["distillation"]["enabled"]
    assert ("rpn_anchor_strides" in last["config"]["model"]) == strides
    assert last["format_version"] == (2 if strides else 1)
    assert last["model_weights"] == "raw" and "train_model" not in last


def test_resume_with_every_option_matches_an_uninterrupted_run(tmp_path):
    files = synthetic_coco(tmp_path / "coco")
    teacher = tmp_path / "teacher.pth"
    torch.save(build_vanilla_faster_rcnn(num_classes=91, pretrained=False).state_dict(), teacher)
    config = with_options(smoke_config((*files, teacher), tmp_path, image_size=128), *OPTIONS)
    config["train"]["multiscale"]["min_sizes"] = [64, 96, 128]
    config["train"]["multiscale"]["max_size"] = 128
    config["train"]["ema"] = {"enabled": True, "decay": 0.9, "warmup": 2}
    config["train"].update(epochs=2, warmup_steps=2)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_train(tmp_path, "--output-dir", full, config=config)
    run_train(tmp_path, "--epochs", 1, "--output-dir", resumed, config=config)
    changed = copy.deepcopy(config)
    changed["train"]["ema"]["decay"] = 0.99
    result = run_train(tmp_path, "--resume", resumed / "last.pth", config=changed, check=False)
    assert result.returncode != 0 and "Cannot change train.ema on resume" in result.stderr
    run_train(tmp_path, "--resume", resumed / "last.pth", "--epochs", 2)
    expected = torch.load(full / "last.pth", map_location="cpu", weights_only=True)
    actual = torch.load(resumed / "last.pth", map_location="cpu", weights_only=True)
    assert actual["epoch"] == 2 and actual["ema"] == expected["ema"]
    assert actual["ema"]["updates"] == actual["global_step"] == 4
    for section in ("model", "train_model", "distillation_adapters"):
        for key, value in expected[section].items():
            torch.testing.assert_close(actual[section][key], value, rtol=0, atol=0)
    rows = [json.loads(line) for line in (resumed / "history.jsonl").read_text().splitlines()]
    assert [row["validated_weights"] for row in rows] == ["ema", "ema"]
    assert all("loss_kd_box" in row["losses"] for row in rows)
