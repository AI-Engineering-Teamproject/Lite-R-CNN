import copy
import dataclasses
import functools
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision.ops import box_iou
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops.poolers import _setup_scales

from light_rcnn import reporting
from light_rcnn.bench import load_comparison_model
from light_rcnn.inference_opts import (
    CUSTOM_LABEL,
    PRESET_LABELS,
    PRESETS,
    InferenceOptions,
    apply_inference_options,
    cuda_graph_note,
    options_to_dict,
    resolve_options,
)
from light_rcnn.model.detector import build_detector
from light_rcnn.postprocess import SOFT_NMS_MIN_SCORE, install_postprocess

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "147753808597_20161028.jpeg"
# Smoke-set ids: 640x426 first (resized height 426 < ~453, the case that breaks unpinned scales),
# then 640x480 and a 481x640 portrait.
SMOKE_IDS = (25, 9, 36)
SCALES = {"v2": [1 / 8, 1 / 16, 1 / 32], "v3": [1 / 4, 1 / 8, 1 / 16, 1 / 32]}
CPU = torch.device("cpu")
DISABLED = "disabled (graph replay differs from eager)"  # _GraphedForward's failed self-check
DISABLED_NOTE = "CUDA 그래프 사용 안 됨 (자체 검사에서 재생 결과가 eager와 달라 백본을 eager로 실행)"


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)


@pytest.fixture(scope="module")
def images() -> list[torch.Tensor]:
    """Three real photos, the first short: COCO smoke images if present, else repo photo crops.

    LIGHT_RCNN_TEST_IMAGES may point at a folder holding the smoke-set files.
    """
    folder = Path(os.environ.get("LIGHT_RCNN_TEST_IMAGES", ROOT / "data/coco/images/val2017"))
    paths = [folder / f"{image_id:012d}.jpg" for image_id in SMOKE_IDS]
    if all(path.is_file() for path in paths):
        return [load_rgb(path) for path in paths]
    if not PHOTO.is_file():
        pytest.skip("no test photo in the work tree")
    photo = load_rgb(PHOTO)  # 970x646, resized to 640x426 by the transform
    return [photo, photo[:, :, :861].contiguous(), photo[:, :, 200:685].contiguous()]


@pytest.fixture(scope="module")
def trained() -> dict[str, nn.Module]:
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    models = {}
    for version in ("v2", "v3"):
        weights = ROOT / f"outputs/coco_4090_{version}/best.pth"
        if not weights.is_file():
            pytest.skip(f"{weights} is not in the work tree")
        models[version] = load_comparison_model("light_rcnn", config, str(weights)).eval()
    return models


@pytest.fixture(scope="module")
def faster_rcnn() -> nn.Module:
    torch.manual_seed(0)
    return build_detector("faster_rcnn", pretrained=False, num_classes=91, min_size=320,
                          max_size=320).eval()


@torch.inference_mode()
def detect(model: nn.Module, images: list[torch.Tensor]) -> list[dict]:
    return [model([image])[0] for image in images]


def assert_identical(expected: list[dict], actual: list[dict]) -> None:
    for left, right in zip(expected, actual, strict=True):
        for key in ("boxes", "scores", "labels"):
            assert torch.equal(left[key], right[key]), key


def assert_detections_close(expected: dict, actual: dict) -> None:
    """Float-noise equality: every confident detection has a same-label twin, counts agree."""
    assert abs(len(expected["scores"]) - len(actual["scores"])) <= 2
    confident = expected["scores"] > 0.1
    if not confident.any():
        return
    overlap = box_iou(expected["boxes"][confident], actual["boxes"])
    overlap *= expected["labels"][confident, None] == actual["labels"][None]
    best_iou, best = overlap.max(dim=1)
    assert float(best_iou.min()) > 0.999
    assert float((actual["scores"][best] - expected["scores"][confident]).abs().max()) < 1e-4


def canvas_reference(model: nn.Module, side: int = 640) -> nn.Module:
    """The investigation's fixed canvas (levers_common.make_fixed_pad + pin_roi_scales)."""
    def batch_images(images, size_divisible=32):
        batched = images[0].new_full((len(images), images[0].shape[0], side, side), 0)
        for index, image in enumerate(images):
            batched[index, :, : image.shape[1], : image.shape[2]].copy_(image)
        return batched

    model.transform.batch_images = batch_images
    pool = model.roi_heads.box_roi_pool
    with torch.inference_mode():
        features = model.backbone(torch.zeros(1, 3, side, side))
    maps = [features[name] for name in pool.featmap_names]
    pool.scales, pool.map_levels = _setup_scales(maps, [(side, side)], pool.canonical_scale,
                                                 pool.canonical_level)
    return model


def test_presets_follow_the_contract():
    assert {"baseline", "deploy", "fast", "accurate", "accurate_fast", "gpu_graph"} <= set(PRESETS)
    assert set(PRESET_LABELS) == set(PRESETS)
    assert PRESET_LABELS["baseline"] == "개선 전 (기본 설정)"
    assert PRESET_LABELS["fast"] == "빠른 설정" and PRESET_LABELS["accurate"] == "정확도 우선"
    assert PRESETS["baseline"] == InferenceOptions()
    assert PRESETS["deploy"] == InferenceOptions(fuse_conv_bn=True, channels_last=None)
    assert PRESETS["fast"] == InferenceOptions(fuse_conv_bn=True, channels_last=None,
                                               rpn_pre_nms_top_n=500, rpn_score_thresh=0.05)
    assert PRESETS["accurate"] == InferenceOptions(fuse_conv_bn=True, channels_last=None,
                                                   fixed_canvas=640, nms="soft")
    assert PRESETS["accurate_fast"] == dataclasses.replace(PRESETS["accurate"], nms="matrix")
    assert PRESETS["gpu_graph"] == InferenceOptions(fuse_conv_bn=True, fixed_canvas=640,
                                                    cuda_graph=True)
    assert len(set(PRESETS.values())) == len(PRESETS)
    assert SOFT_NMS_MIN_SCORE == 0.001
    assert {options.soft_nms_min_score for options in PRESETS.values()} == {SOFT_NMS_MIN_SCORE}
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRESETS["fast"].nms = "soft"


def test_resolve_options_accepts_names_options_and_mappings():
    assert resolve_options("fast") is PRESETS["fast"]
    assert resolve_options(" Accurate-Fast ") is PRESETS["accurate_fast"]
    assert resolve_options(None) == InferenceOptions()
    custom = InferenceOptions(nms="matrix", soft_nms_sigma=0.3)
    assert resolve_options(custom) is custom
    for options in PRESETS.values():
        assert resolve_options(json.loads(json.dumps(options_to_dict(options)))) == options
    tuned = resolve_options("accurate", box_score_thresh=0.001, nms="matrix",
                            fixed_canvas=None, rpn_pre_nms_top_n=None)
    assert tuned == dataclasses.replace(PRESETS["accurate"], box_score_thresh=0.001, nms="matrix")
    assert resolve_options("fast", rpn_pre_nms_top_n=np.int64(300)).rpn_pre_nms_top_n == 300
    assert type(resolve_options("fast", rpn_pre_nms_top_n=np.int64(300)).rpn_pre_nms_top_n) is int
    floor = resolve_options("accurate", soft_nms_min_score=np.float32(0.05)).soft_nms_min_score
    assert type(floor) is float and floor == pytest.approx(0.05)
    assert resolve_options("accurate", soft_nms_min_score=0).soft_nms_min_score == 0.0
    # Records written before the option existed resolve to the same presets.
    for options in PRESETS.values():
        old = {k: v for k, v in options_to_dict(options).items() if k != "soft_nms_min_score"}
        assert resolve_options(old) == options


@pytest.mark.parametrize("call, error", [
    (lambda: resolve_options("turbo"), ValueError),
    (lambda: resolve_options("fast", turbo=True), TypeError),
    (lambda: resolve_options({"turbo": True}), TypeError),
    (lambda: resolve_options(3), TypeError),
    (lambda: InferenceOptions(nms="greedy"), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_sigma=0), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score=-0.1), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score=1.5), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score=float("nan")), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score=True), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score="0.001"), ValueError),
    (lambda: InferenceOptions(nms="soft", soft_nms_min_score=None), ValueError),
    (lambda: InferenceOptions(cuda_graph=True), ValueError),
    (lambda: InferenceOptions(compile_backbone="default"), ValueError),
    (lambda: InferenceOptions(fixed_canvas=640, compile_backbone="turbo"), ValueError),
    (lambda: InferenceOptions(fixed_canvas=640, cuda_graph=True, compile_backbone="default"),
     ValueError),
    (lambda: InferenceOptions(rpn_pre_nms_top_n=0), ValueError),
    (lambda: InferenceOptions(detections_per_img=True), ValueError),
    (lambda: InferenceOptions(fixed_canvas=640.0), ValueError),
    (lambda: InferenceOptions(box_score_thresh=1.5), ValueError),
    (lambda: InferenceOptions(channels_last="yes"), ValueError),
])
def test_invalid_options_are_rejected(call, error):
    with pytest.raises(error):
        call()


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_baseline_leaves_outputs_bit_identical(trained, images, version):
    expected = detect(copy.deepcopy(trained[version]), images)
    model = copy.deepcopy(trained[version])
    record = apply_inference_options(model, "baseline", CPU)
    assert record["preset"] == "baseline" and record["summary"] == "모델 기본 설정"
    assert record["skipped"] == {}
    for owner, name in ((model.transform, "batch_images"), (model.roi_heads,
                                                            "postprocess_detections"),
                        (model.backbone, "forward"), (model.rpn.head, "forward")):
        assert name not in vars(owner)
    assert model.roi_heads.box_roi_pool.scales is None
    assert_identical(expected, detect(model, images))


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_fuse_and_channels_last_stay_within_float_tolerance(trained, images, version):
    expected = detect(copy.deepcopy(trained[version]), images)
    model = copy.deepcopy(trained[version])
    record = apply_inference_options(model, resolve_options("deploy", channels_last=True), CPU)
    assert record["applied"]["conv_bn_fused"] is True
    assert record["applied"]["memory_format"] == "channels_last"
    assert not any(isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d))
                   for module in model.modules())
    conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    actual = detect(model, images)
    assert sum(len(output["scores"]) for output in expected) > 0
    for left, right in zip(expected, actual, strict=True):
        assert_detections_close(left, right)


def test_rpn_knobs_reach_the_proposals(trained, images):
    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, resolve_options("fast", rpn_post_nms_top_n=50), CPU)
    assert model.rpn._pre_nms_top_n == {"training": 2000, "testing": 500}
    assert model.rpn._post_nms_top_n == {"training": 1000, "testing": 50}
    assert model.rpn.score_thresh == 0.05
    assert trained["v2"].rpn._pre_nms_top_n["testing"] == 1000  # the source model is untouched
    applied = record["applied"]
    assert (applied["rpn_pre_nms_top_n_test"], applied["rpn_post_nms_top_n_test"],
            applied["rpn_score_thresh"]) == (500, 50, 0.05)
    kept = []
    filter_proposals = model.rpn.filter_proposals

    def spy(*args):
        boxes, scores = filter_proposals(*args)
        kept.append(scores[0])
        return boxes, scores

    model.rpn.filter_proposals = spy
    detect(model, images[:1])
    assert 0 < len(kept[0]) <= 50
    assert float(kept[0].min()) >= 0.05


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_fixed_canvas_pins_roi_scales_against_a_short_first_image(trained, images, version):
    model = copy.deepcopy(trained[version])
    record = apply_inference_options(model, InferenceOptions(fixed_canvas=640), CPU)
    assert record["applied"]["fixed_canvas"] == [640, 640]
    assert record["applied"]["roi_scales"] == SCALES[version]
    short_first = detect(model, images)
    assert model.roi_heads.box_roi_pool.scales == SCALES[version]

    reference = canvas_reference(copy.deepcopy(trained[version]))
    assert_identical(detect(reference, images[1:]), short_first[1:])

    # Control: the same canvas unpinned caches 2x scales from the short first image.
    unpinned = copy.deepcopy(trained[version])
    unpinned.transform.batch_images = reference.transform.batch_images
    corrupted = detect(unpinned, images)
    assert unpinned.roi_heads.box_roi_pool.scales == [2 * scale for scale in SCALES[version]]
    assert any(len(bad["boxes"]) != len(good["boxes"]) or not torch.equal(bad["boxes"],
                                                                          good["boxes"])
               for bad, good in zip(corrupted[1:], short_first[1:]))


def test_fixed_canvas_pads_to_the_canvas_and_widens_for_larger_images(trained):
    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, InferenceOptions(fixed_canvas=600), CPU)
    assert record["applied"]["fixed_canvas"] == [608, 608]  # rounded up to size_divisible 32
    batch = model.transform.batch_images
    padded = batch([torch.ones(3, 426, 600), torch.ones(3, 300, 200)], size_divisible=32)
    assert padded.shape == (2, 3, 608, 608)
    assert padded[0, :, :426, :600].eq(1).all() and padded[0, :, 426:].eq(0).all()
    assert padded[1, :, :300, :200].eq(1).all() and padded[1, :, :, 200:].eq(0).all()
    assert batch([torch.ones(3, 700, 500)], size_divisible=32).shape == (1, 3, 704, 608)


def test_record_reports_device_skips_and_round_trips(trained):
    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, "gpu_graph", CPU)
    assert json.loads(json.dumps(record)) == record
    assert (record["preset"], record["label"]) == ("gpu_graph", PRESET_LABELS["gpu_graph"])
    assert record["device"] == "cpu"
    assert record["skipped"] == {"cuda_graph": "skipped (CPU)"}
    assert record["applied"]["cuda_graph"] is False and "forward" not in vars(model.backbone)
    assert record["applied"]["conv_bn_fused"] is True
    assert record["applied"]["memory_format"] == "contiguous"
    assert resolve_options(record["options"]) == PRESETS["gpu_graph"]
    assert "CUDA 그래프 미적용 (CPU)" in record["summary"]
    assert "640×640 고정 캔버스" in record["summary"]

    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, "gpu_fp16", CPU)
    assert record["skipped"] == {"fp16": "skipped (CPU)"} and record["applied"]["fp16"] is False
    assert "forward" not in vars(model.backbone) and "forward" not in vars(model.rpn.head)

    model = copy.deepcopy(trained["v2"])
    options = resolve_options("accurate_fast", box_score_thresh=0.001, detections_per_img=50)
    record = apply_inference_options(model, options, CPU)
    assert (record["preset"], record["label"]) == (None, CUSTOM_LABEL)
    assert record["applied"]["nms"] == "matrix" and record["applied"]["soft_nms_sigma"] == 0.5
    assert record["applied"]["matrix_nms_top_k"] > 0
    assert (record["applied"]["box_score_thresh"], record["applied"]["detections_per_img"]) == (
        0.001, 50)
    assert (model.roi_heads.score_thresh, model.roi_heads.detections_per_img) == (0.001, 50)
    assert "Matrix NMS (σ=0.5)" in record["summary"] and "감쇠 하한" not in record["summary"]
    assert "soft_nms_min_score" not in record["applied"]  # Matrix NMS drops at the RoI floor
    assert "postprocess_detections" in vars(model.roi_heads)


def test_soft_nms_min_score_reaches_the_postprocess_and_the_record(trained, images):
    # The default floor behaves exactly as before the option existed (install_postprocess' default).
    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, "accurate", CPU)
    assert record["preset"] == "accurate"
    assert record["applied"]["soft_nms_min_score"] == SOFT_NMS_MIN_SCORE
    assert "Soft-NMS (σ=0.5, 감쇠 하한 0.001)" in record["summary"]
    assert model.roi_heads.postprocess_detections.min_score == SOFT_NMS_MIN_SCORE
    reference = copy.deepcopy(trained["v2"])
    apply_inference_options(reference, "accurate", CPU)
    install_postprocess(reference.roi_heads, "soft", 0.5)
    default = detect(model, images)
    assert_identical(detect(reference, images), default)

    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, resolve_options("accurate", soft_nms_min_score=0.05),
                                     CPU)
    assert (record["preset"], record["label"]) == (None, CUSTOM_LABEL)
    assert record["applied"]["soft_nms_min_score"] == 0.05
    assert record["options"]["soft_nms_min_score"] == 0.05
    assert resolve_options(record["options"]) == resolve_options("accurate", soft_nms_min_score=0.05)
    assert "Soft-NMS (σ=0.5, 감쇠 하한 0.05)" in record["summary"]
    pruned = detect(model, images)
    assert all(bool((output["scores"] > 0.05).all()) for output in pruned)
    assert sum(map(len, (o["scores"] for o in pruned))) < sum(map(len, (o["scores"] for o in default)))

    model = copy.deepcopy(trained["v2"])
    record = apply_inference_options(model, "deploy", CPU)  # hard NMS: nothing to record
    assert "soft_nms_min_score" not in record["applied"] and "Soft-NMS" not in record["summary"]


def test_cuda_graph_note_is_reporting_s_torch_free_helper():
    from light_rcnn import inference_opts

    assert inference_opts.cuda_graph_note is reporting.cuda_graph_note is cuda_graph_note


@pytest.mark.parametrize("record, note", [
    (None, None),
    ({}, None),
    ({"preset": "fast", "summary": "Conv+BN 융합", "applied": {"cuda_graph": False}}, None),
    ({"cuda_graph_status": {"status": "captured", "replays": 9, "eager_calls": 0}}, None),
    ({"cuda_graph_status": {"status": DISABLED, "replays": 0, "eager_calls": 9}}, DISABLED_NOTE),
    ({"cuda_graph_status": {"status": "pending (captured on the first CUDA call)", "replays": 0,
                            "eager_calls": 0}},
     "CUDA 그래프 상태: pending (captured on the first CUDA call)"),
    ({"cuda_graph_status": {"replays": 0, "eager_calls": 3}}, "CUDA 그래프 상태: 기록 없음"),
    ({"cuda_graph_status": "disabled (hand-edited report)"}, DISABLED_NOTE),
])
def test_cuda_graph_note_reads_the_status_not_the_summary(record, note):
    assert cuda_graph_note(record) == note


def test_cuda_graph_note_corrects_a_gpu_graph_summary_after_a_failed_self_check(faster_rcnn):
    model = copy.deepcopy(faster_rcnn)
    record = apply_inference_options(model, resolve_options("gpu_graph", fixed_canvas=320), CPU)
    assert "CUDA 그래프 미적용 (CPU)" in record["summary"]
    assert cuda_graph_note(record) is None  # skipped on a CPU, and the summary says so
    # The CUDA path: 'CUDA 그래프' in the summary at apply time and the backbone wrapper's live
    # status, which the first CUDA forward and its self-check fill in afterwards.
    record["applied"]["cuda_graph"], record["skipped"] = True, {}
    record["summary"] = record["summary"].replace("CUDA 그래프 미적용 (CPU)", "CUDA 그래프")
    assert record["summary"] == "Conv+BN 융합 · 320×320 고정 캔버스 · CUDA 그래프"
    status = {"status": "pending (captured on the first CUDA call)", "replays": 0,
              "eager_calls": 0}
    record["cuda_graph_status"] = status
    assert cuda_graph_note(record).startswith("CUDA 그래프 상태: pending")
    status.update(status=DISABLED, input_shape=[1, 3, 320, 320], self_check_max_abs_diff=0.5,
                  eager_calls=9)
    assert cuda_graph_note(record) == DISABLED_NOTE
    assert cuda_graph_note(json.loads(json.dumps(record))) == DISABLED_NOTE  # as in a report
    status["status"] = "captured"
    assert cuda_graph_note(record) is None


@pytest.mark.parametrize("preset", ["accurate", "accurate_fast"])
def test_accuracy_presets_run_on_v3(trained, images, preset):
    model = copy.deepcopy(trained["v3"])
    record = apply_inference_options(model, preset, CPU)
    assert record["applied"]["roi_scales"] == SCALES["v3"]
    installed = model.roi_heads.postprocess_detections  # Soft-NMS drops at its own min_score
    floor = installed.min_score if installed.method == "soft" else model.roi_heads.score_thresh
    for output in detect(model, images):
        scores = output["scores"]
        assert len(scores) <= model.roi_heads.detections_per_img
        assert bool((scores > floor).all())
        assert torch.equal(scores, scores.sort(descending=True).values)


def test_compile_backbone_traces_the_canvas_backbone(trained, images, monkeypatch):
    # The eager backend checks the wiring without minutes of Inductor code generation.
    monkeypatch.setattr(torch, "compile", functools.partial(torch.compile, backend="eager"))
    model = copy.deepcopy(trained["v2"])
    options = InferenceOptions(fixed_canvas=640, compile_backbone="default")
    record = apply_inference_options(model, options, CPU)
    assert record["applied"]["compile_backbone"] == "default"
    assert model.backbone._compiled_call_impl is not None
    reference = copy.deepcopy(trained["v2"])
    apply_inference_options(reference, InferenceOptions(fixed_canvas=640), CPU)
    assert_identical(detect(reference, images[:1]), detect(model, images[:1]))


@pytest.mark.parametrize("preset", ["deploy", "fast", "accurate", "accurate_fast", "gpu_graph"])
def test_presets_apply_to_torchvision_faster_rcnn(faster_rcnn, images, preset):
    model = copy.deepcopy(faster_rcnn)
    canvas = 320 if PRESETS[preset].fixed_canvas else None
    options = resolve_options(preset, fixed_canvas=canvas, box_score_thresh=0.001)
    record = apply_inference_options(model, options, CPU)
    assert record["applied"]["conv_bn_fused"] is True
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    if preset == "fast":
        assert model.rpn._pre_nms_top_n["testing"] == 500 and model.rpn.score_thresh == 0.05
    if canvas:
        assert record["applied"]["roi_scales"] == [1 / 4, 1 / 8, 1 / 16, 1 / 32]
    assert ("postprocess_detections" in vars(model.roi_heads)) == (options.nms != "hard")
    output = detect(model, images[:1])[0]
    height, width = images[0].shape[-2:]
    assert 0 < len(output["scores"]) <= model.roi_heads.detections_per_img
    assert float(output["boxes"][:, 2].max()) <= width and float(output["boxes"][:, 3].max()) <= height
    assert torch.equal(output["scores"], output["scores"].sort(descending=True).values)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
def test_cuda_graph_replays_the_eager_backbone(trained, images):
    device = torch.device("cuda")
    eager = copy.deepcopy(trained["v2"])
    apply_inference_options(eager, InferenceOptions(fuse_conv_bn=True, fixed_canvas=640), device)
    graphed = copy.deepcopy(trained["v2"])
    record = apply_inference_options(graphed, "gpu_graph", device)
    inputs = [image.to(device) for image in images]
    expected, actual = detect(eager, inputs), detect(graphed, inputs)
    assert record["cuda_graph_status"]["status"] == "captured" and cuda_graph_note(record) is None
    assert record["cuda_graph_status"]["replays"] == len(images)  # the capture call replays too
    for left, right in zip(expected, actual, strict=True):
        assert_detections_close(left, right)


@cuda
def test_fp16_autocast_keeps_float32_boxes_close(trained, images):
    device = torch.device("cuda")
    reference = copy.deepcopy(trained["v2"])
    apply_inference_options(reference, "deploy", device)
    half = copy.deepcopy(trained["v2"])
    record = apply_inference_options(half, "gpu_fp16", device)
    assert record["applied"]["fp16"] is True and record["skipped"] == {}
    inputs = [image.to(device) for image in images]
    for left, right in zip(detect(reference, inputs), detect(half, inputs), strict=True):
        assert right["boxes"].dtype == torch.float32
        confident = left["scores"] > 0.5
        if confident.any():
            overlap = box_iou(left["boxes"][confident], right["boxes"])
            assert float(overlap.max(dim=1).values.min()) > 0.9
