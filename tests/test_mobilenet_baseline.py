"""torchvision Faster R-CNN MobileNetV3-Large-FPN ("the MobileNet benchmark") as a compared model.

Nothing is downloaded: random builds use weights=None and weights_backbone=None, and the COCO_V1
path (FrozenBatchNorm2d backbone, frozen early layers) runs on a fake state dict patched into
torchvision's weights enum. Outputs of random weights only check the wiring, never accuracy.
"""

import copy
import json
import os
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops.poolers import _setup_scales

from light_rcnn.bench import (
    DEFAULT_WEIGHTS,
    TORCHVISION_BASELINES,
    load_comparison_model,
    set_detection_limits,
)
from light_rcnn.inference_opts import (
    PRESETS,
    apply_inference_options,
    resolve_options,
)
from light_rcnn.model.deploy import fuse_conv_bn, use_channels_last
from light_rcnn.model.detector import build_detector
from light_rcnn.postprocess import SOFT_NMS_MIN_SCORE, PostprocessDetections

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "147753808597_20161028.jpeg"
KIND = "mobilenet_faster_rcnn"
CONFIG = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
CPU = torch.device("cpu")
WEIGHTS = FasterRCNN_MobileNet_V3_Large_FPN_Weights
# Smoke-set ids as in test_inference_opts: 640x426 first (a short first image), then 640x480.
SMOKE_IDS = (25, 9)
# Every MobileNetV3-Large norm follows its conv: stem, 2 in the unexpanded first block, 3 in each
# of the other 14 inverted residuals, the last 1x1 conv. FPN, RPN and box head have none.
MOBILENET_NORMS = 46
# Both pooled FPN levels ('0' after the last strided block, '1' after the 960-channel conv) are
# stride 32; the extra 'pool' level (stride 64) feeds only the RPN.
ROI_SCALES = [1 / 32, 1 / 32]
NORMS = (nn.BatchNorm2d, FrozenBatchNorm2d)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)


@pytest.fixture(scope="module")
def images() -> list[torch.Tensor]:
    """Two real photos, the first short: COCO smoke images if present, else repo photo crops.

    LIGHT_RCNN_TEST_IMAGES may point at a folder holding the smoke-set files.
    """
    folder = Path(os.environ.get("LIGHT_RCNN_TEST_IMAGES", ROOT / "data/coco/images/val2017"))
    paths = [folder / f"{image_id:012d}.jpg" for image_id in SMOKE_IDS]
    if all(path.is_file() for path in paths):
        return [load_rgb(path) for path in paths]
    if not PHOTO.is_file():
        pytest.skip("no test photo in the work tree")
    photo = load_rgb(PHOTO)  # 970x646, resized to 640x426 by the transform
    return [photo, photo[:, :, :861].contiguous()]


def randomize_norms(model: nn.Module, seed: int = 1) -> nn.Module:
    """Non-trivial norm statistics, so folding them into the convs is a real change."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, NORMS):
                size = module.running_mean.numel()
                module.running_mean.copy_(torch.randn(size, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(size, generator=generator) * 1.5 + 0.5)
                module.weight.copy_(torch.rand(size, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(size, generator=generator) * 0.1)
    return model


def count(model: nn.Module, kinds) -> int:
    return sum(isinstance(module, kinds) for module in model.modules())


@pytest.fixture(scope="module")
def random_model() -> nn.Module:
    """weights=None, weights_backbone=None: BatchNorm2d backbone, as every offline run builds it."""
    torch.manual_seed(0)
    model = build_detector(KIND, pretrained=False, num_classes=91, min_size=640, max_size=640)
    return randomize_norms(model).eval()


@pytest.fixture(scope="module")
def pretrained_model(random_model) -> nn.Module:
    """The COCO_V1 code path (FrozenBatchNorm2d, frozen early layers) on a fake weights file."""
    state, calls = random_model.state_dict(), []

    def get_state_dict(weights, *args, **kwargs):
        calls.append((weights, kwargs))
        return copy.deepcopy(state)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(WEIGHTS, "get_state_dict", get_state_dict)
        model = load_comparison_model(KIND, CONFIG, None, 640)
    assert [weights for weights, _ in calls] == [WEIGHTS.COCO_V1]  # the full model, not the 320
    return model.eval()


@pytest.fixture
def no_download(monkeypatch):
    def refuse(weights, *args, **kwargs):
        raise URLError("download.pytorch.org is blocked in tests")

    monkeypatch.setattr(WEIGHTS, "get_state_dict", refuse)


@torch.inference_mode()
def detect(model: nn.Module, images: list[torch.Tensor]) -> list[dict]:
    return [model([image])[0] for image in images]


def assert_identical(expected: list[dict], actual: list[dict]) -> None:
    for left, right in zip(expected, actual, strict=True):
        for key in ("boxes", "scores", "labels"):
            assert torch.equal(left[key], right[key]), key


@torch.inference_mode()
def stage_outputs(model: nn.Module, image: torch.Tensor, proposals=None):
    """Every stage's raw output for one image: FPN maps, RPN head, RoI head on ``proposals``.

    Random heads give near-tied scores, so a top-k cut can flip on float noise; comparing the
    RoI head on the same proposals checks the numbers without that cut.
    """
    batch, _ = model.transform([image])
    features = model.backbone(batch.tensors)
    objectness, deltas = model.rpn.head(list(features.values()))
    if proposals is None:
        proposals, _ = model.rpn(batch, features)
    heads = model.roi_heads
    pooled = heads.box_roi_pool(features, proposals, batch.image_sizes)
    logits, regression = heads.box_predictor(heads.box_head(pooled))
    outputs = {f"feature {name}": value for name, value in features.items()}
    outputs.update({f"objectness {level}": value for level, value in enumerate(objectness)})
    outputs.update({f"deltas {level}": value for level, value in enumerate(deltas)})
    outputs.update(class_logits=logits, box_regression=regression)
    return outputs, proposals


def assert_stages_close(expected: nn.Module, actual: nn.Module, images: list[torch.Tensor],
                        tolerance: float = 1e-4) -> None:
    """Relative max error per stage; ``expected`` is copied, so a module fixture never caches
    its RoI pooler scales here."""
    expected = copy.deepcopy(expected)
    for image in images:
        reference, proposals = stage_outputs(expected, image)
        outputs, _ = stage_outputs(actual, image, proposals)
        assert outputs.keys() == reference.keys()
        for name, value in reference.items():
            scale = max(float(value.abs().max()), 1e-6)
            assert float((outputs[name] - value).abs().max()) <= tolerance * scale, name


def canvas_reference(model: nn.Module, side: int = 640) -> nn.Module:
    """The investigation's fixed canvas, pooling the levels torchvision's pooler pools."""
    def batch_images(images, size_divisible=32):
        batched = images[0].new_full((len(images), images[0].shape[0], side, side), 0)
        for index, image in enumerate(images):
            batched[index, :, : image.shape[1], : image.shape[2]].copy_(image)
        return batched

    model.transform.batch_images = batch_images
    pool = model.roi_heads.box_roi_pool
    with torch.inference_mode():
        features = model.backbone(torch.zeros(1, 3, side, side))
    maps = [value for name, value in features.items() if name in pool.featmap_names]
    pool.scales, pool.map_levels = _setup_scales(maps, [(side, side)], pool.canonical_scale,
                                                 pool.canonical_level)
    return model


def assert_well_formed(model: nn.Module, output: dict, image: torch.Tensor) -> None:
    height, width = image.shape[-2:]
    scores = output["scores"]
    assert 0 < len(scores) <= model.roi_heads.detections_per_img
    assert float(output["boxes"].min()) >= 0
    assert float(output["boxes"][:, 2].max()) <= width
    assert float(output["boxes"][:, 3].max()) <= height
    assert int(output["labels"].min()) >= 1 and int(output["labels"].max()) <= 90
    assert torch.equal(scores, scores.sort(descending=True).values)


# --------------------------------------------------------------------------- builder and loader


def test_builder_follows_the_contract_offline(no_download):
    model = build_detector(KIND, pretrained=False, num_classes=91, min_size=640, max_size=640)
    assert isinstance(model, FasterRCNN)
    # Our protocol's 640 resize, not torchvision's native 800/1333.
    assert (model.transform.min_size, model.transform.max_size) == ((640,), 640)
    assert model.transform.size_divisible == 32
    # Every other torchvision default stays (the full model, not the 320 one).
    rpn, heads = model.rpn, model.roi_heads
    assert rpn._pre_nms_top_n == {"training": 2000, "testing": 1000}
    assert rpn._post_nms_top_n == {"training": 2000, "testing": 1000}
    assert (rpn.score_thresh, rpn.nms_thresh) == (0.05, 0.7)
    assert rpn.anchor_generator.sizes == ((32, 64, 128, 256, 512),) * 3
    assert (heads.score_thresh, heads.nms_thresh, heads.detections_per_img) == (0.05, 0.5, 100)
    assert heads.box_predictor.cls_score.out_features == 91
    assert count(model, nn.BatchNorm2d) == MOBILENET_NORMS and count(model, FrozenBatchNorm2d) == 0
    with torch.inference_mode():
        features = model.eval().backbone(torch.zeros(1, 3, 640, 640))
    assert {name: tuple(value.shape[-2:]) for name, value in features.items()} == {
        "0": (20, 20), "1": (20, 20), "pool": (10, 10)}
    assert KIND in TORCHVISION_BASELINES and DEFAULT_WEIGHTS[KIND] == "torchvision_coco_v1"
    with pytest.raises(ValueError, match="Unknown detector kind"):
        build_detector("mobilenet_v3_320")


def test_eval_forward_on_two_smoke_images_at_640(random_model, images):
    model = copy.deepcopy(random_model)
    set_detection_limits(model, 0.001, 100)  # random heads score ~1/91 per class
    assert (model.roi_heads.score_thresh, model.roi_heads.detections_per_img) == (0.001, 100)
    shapes = []
    handle = model.transform.register_forward_hook(
        lambda module, args, output: shapes.append((tuple(output[0].tensors.shape[-2:]),
                                                    output[0].image_sizes[0])))
    try:
        outputs = detect(model, images)
    finally:
        handle.remove()
    for image, (padded, resized), output in zip(images, shapes, outputs, strict=True):
        assert abs(max(resized) - 640) <= 1  # the longer side (floor of a float scale)
        assert all(side % 32 == 0 and side <= 640 for side in padded)
        assert_well_formed(model, output, image)


def test_detection_limits_and_rpn_knobs_exist(random_model):
    model = copy.deepcopy(random_model)
    set_detection_limits(model, 0.25, 50)
    assert (model.roi_heads.score_thresh, model.roi_heads.detections_per_img) == (0.25, 50)
    # The proposal budgets the measurement scripts read and set (evaluate_three_models.set_rpn_limit).
    assert model.rpn.pre_nms_top_n() == 1000 and model.rpn.post_nms_top_n() == 1000
    model.rpn._post_nms_top_n["testing"] = 300
    assert model.rpn.post_nms_top_n() == 300


@pytest.mark.parametrize("wrapped", [False, True], ids=["state_dict", "model_key"])
def test_local_state_dict_round_trips_through_the_loader(random_model, images, tmp_path, wrapped,
                                                         no_download):
    state = random_model.state_dict()
    path = tmp_path / "mobilenet.pth"
    torch.save({"model": state} if wrapped else state, path)
    model = load_comparison_model(KIND, CONFIG, str(path), 640).eval()
    assert (model.transform.min_size, model.transform.max_size) == ((640,), 640)
    loaded = model.state_dict()
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[key], value) for key, value in state.items())
    assert_identical(detect(copy.deepcopy(random_model), images[:1]), detect(model, images[:1]))
    # The input size follows image_size, as for the ResNet-50 baseline.
    small = load_comparison_model(KIND, CONFIG, str(path), 320)
    assert (small.transform.min_size, small.transform.max_size) == ((320,), 320)


def test_coco_weights_path_builds_the_pretrained_structure(pretrained_model):
    model = pretrained_model  # built by load_comparison_model(weights=None) on fake COCO_V1
    assert count(model, FrozenBatchNorm2d) == MOBILENET_NORMS and count(model, nn.BatchNorm2d) == 0
    assert (model.transform.min_size, model.transform.max_size) == ((640,), 640)
    assert model.rpn.score_thresh == 0.05 and model.roi_heads.box_predictor.cls_score.out_features == 91
    assert any(not parameter.requires_grad for parameter in model.backbone.body.parameters())


def test_unavailable_coco_weights_raise_a_catchable_error(no_download):
    # The scripts catch OSError/URLError and report "weights unavailable", as for Faster R-CNN.
    with pytest.raises(URLError) as caught:
        load_comparison_model(KIND, CONFIG, None, 640)
    assert isinstance(caught.value, OSError)


def test_pretrained_checkpoint_loads_into_the_local_weights_build(pretrained_model, images,
                                                                   tmp_path, no_download):
    """A state dict of the COCO model (FrozenBatchNorm2d, no num_batches_tracked) as a local file,
    e.g. torchvision's cached checkpoint: the BatchNorm2d build scores it alike."""
    path = tmp_path / "fasterrcnn_mobilenet_v3_large_fpn.pth"
    torch.save(pretrained_model.state_dict(), path)
    model = load_comparison_model(KIND, CONFIG, str(path), 640).eval()
    assert count(model, nn.BatchNorm2d) == MOBILENET_NORMS
    assert_stages_close(pretrained_model, model, images[:1])


# --------------------------------------------------------------------------- inference presets


@pytest.mark.parametrize("which", ["random_model", "pretrained_model"],
                         ids=["BatchNorm2d", "FrozenBatchNorm2d"])
def test_conv_bn_folding_covers_every_mobilenet_norm(which, request, images):
    source = request.getfixturevalue(which)
    norm = nn.BatchNorm2d if which == "random_model" else FrozenBatchNorm2d
    assert count(source, norm) == MOBILENET_NORMS
    # Registration order matches forward order in every Conv2dNormActivation of the body, so
    # each norm folds into the conv registered just before it.
    folded = copy.deepcopy(source)
    assert fuse_conv_bn(folded) == MOBILENET_NORMS and count(folded, NORMS) == 0
    model = copy.deepcopy(source)
    record = apply_inference_options(model, "deploy", CPU)
    assert record["applied"]["conv_bn_fused"] is True and count(model, NORMS) == 0
    expected_format = "channels_last" if use_channels_last(CPU) else "contiguous"
    assert record["applied"]["memory_format"] == expected_format  # Ultralytics' auto rule
    if expected_format == "channels_last":
        conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
        assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert_stages_close(source, model, images)


def test_baseline_leaves_mobilenet_outputs_bit_identical(random_model, images):
    expected = detect(copy.deepcopy(random_model), images)
    model = copy.deepcopy(random_model)
    record = apply_inference_options(model, "baseline", CPU)
    assert record["preset"] == "baseline" and record["summary"] == "모델 기본 설정"
    assert record["skipped"] == {} and record["applied"]["conv_bn_fused"] is False
    for owner, name in ((model.transform, "batch_images"),
                        (model.roi_heads, "postprocess_detections"),
                        (model.backbone, "forward"), (model.rpn.head, "forward")):
        assert name not in vars(owner)
    assert model.roi_heads.box_roi_pool.scales is None
    assert_identical(expected, detect(model, images))


def test_rpn_knobs_reach_the_mobilenet_proposals(random_model, images):
    model = copy.deepcopy(random_model)
    # The fast preset's 0.05 RPN floor is this model's own default; a higher one must bite.
    options = resolve_options("fast", rpn_post_nms_top_n=50, rpn_score_thresh=0.5)
    record = apply_inference_options(model, options, CPU)
    assert model.rpn._pre_nms_top_n == {"training": 2000, "testing": 500}
    assert model.rpn._post_nms_top_n == {"training": 2000, "testing": 50}
    assert model.rpn.score_thresh == 0.5 and random_model.rpn.score_thresh == 0.05
    applied = record["applied"]
    assert (applied["rpn_pre_nms_top_n_test"], applied["rpn_post_nms_top_n_test"],
            applied["rpn_score_thresh"]) == (500, 50, 0.5)
    kept = []
    filter_proposals = model.rpn.filter_proposals

    def spy(*args):
        boxes, scores = filter_proposals(*args)
        kept.append(scores[0])
        return boxes, scores

    model.rpn.filter_proposals = spy
    detect(model, images[:1])
    assert 0 < len(kept[0]) <= 50
    assert float(kept[0].min()) >= 0.5


def test_fixed_canvas_pins_mobilenet_roi_scales_against_a_short_first_image(random_model, images):
    assert images[0].shape[-2] / images[0].shape[-1] < 453 / 640  # resizes below ~453 px high
    model = copy.deepcopy(random_model)
    set_detection_limits(model, 0.001, 100)
    record = apply_inference_options(model, resolve_options("baseline", fixed_canvas=640), CPU)
    assert record["applied"]["fixed_canvas"] == [640, 640]
    assert record["applied"]["roi_scales"] == ROI_SCALES
    short_first = detect(model, images)
    assert model.roi_heads.box_roi_pool.scales == ROI_SCALES

    reference = copy.deepcopy(random_model)
    set_detection_limits(reference, 0.001, 100)
    canvas_reference(reference)
    assert_identical(detect(reference, images[1:]), short_first[1:])

    # Control: the same canvas unpinned caches 2x scales from the short first image.
    unpinned = copy.deepcopy(random_model)
    set_detection_limits(unpinned, 0.001, 100)
    unpinned.transform.batch_images = reference.transform.batch_images
    corrupted = detect(unpinned, images)
    assert unpinned.roi_heads.box_roi_pool.scales == [2 * scale for scale in ROI_SCALES]
    assert any(len(bad["boxes"]) != len(good["boxes"]) or not torch.equal(bad["boxes"],
                                                                          good["boxes"])
               for bad, good in zip(corrupted[1:], short_first[1:]))


@pytest.mark.parametrize("preset", ["deploy", "fast", "accurate", "accurate_fast", "gpu_fp16",
                                    "gpu_graph"])
def test_presets_apply_to_mobilenet_faster_rcnn(random_model, images, preset):
    model = copy.deepcopy(random_model)
    options = resolve_options(preset, box_score_thresh=0.001)  # random heads score ~1/91
    record = apply_inference_options(model, options, CPU)
    assert json.loads(json.dumps(record)) == record
    applied = record["applied"]
    assert applied["conv_bn_fused"] is True and count(model, NORMS) == 0
    auto = use_channels_last(CPU) if options.channels_last is None else options.channels_last
    assert applied["memory_format"] == ("channels_last" if auto else "contiguous")
    if preset == "fast":
        assert model.rpn._pre_nms_top_n["testing"] == 500 and model.rpn.score_thresh == 0.05
    if PRESETS[preset].fixed_canvas:
        assert applied["fixed_canvas"] == [640, 640] and applied["roi_scales"] == ROI_SCALES
    skipped = {"gpu_fp16": {"fp16": "skipped (CPU)"},
               "gpu_graph": {"cuda_graph": "skipped (CPU)"}}.get(preset, {})
    assert record["skipped"] == skipped
    assert applied["fp16"] is False and applied["cuda_graph"] is False
    assert "forward" not in vars(model.backbone)
    assert ("postprocess_detections" in vars(model.roi_heads)) == (options.nms != "hard")
    output = detect(model, images[:1])[0]
    assert_well_formed(model, output, images[0])
    floor = SOFT_NMS_MIN_SCORE if options.nms == "soft" else model.roi_heads.score_thresh
    assert bool((output["scores"] > floor).all())


def test_hard_replica_is_bit_identical_on_the_mobilenet_head(random_model, images):
    """Soft and Matrix NMS reuse torchvision's own pre-NMS code; 'hard' must equal stock here too."""
    model = copy.deepcopy(random_model)
    set_detection_limits(model, 0.001, 100)
    expected = detect(model, images[:1])
    model.roi_heads.postprocess_detections = PostprocessDetections(model.roi_heads, "hard")
    assert_identical(expected, detect(model, images[:1]))


# --------------------------------------------------------------------------- reports and dashboard

T4 = {"device": "cuda:0", "device_type": "cuda", "device_name": "Tesla T4"}
RUN = "20261103_120000"  # one comparison run, as compare_improvements.py names it
# model -> (AP@[.50:.95], p50 ms, parameters M, GFLOPs, weights): synthetic T4-like numbers.
FOUR_MODELS = {
    "faster_rcnn": (0.352, 48.0, 41.8, 134.4, "torchvision_coco_v1"),
    "mobilenet_faster_rcnn": (0.289, 14.0, 19.4, 4.5, "torchvision_coco_v1"),
    "light_rcnn": (0.271, 9.0, 3.3, 5.5, "outputs/coco_4090_v3/best.pth"),
    "yolo26": (0.392, 4.5, 2.4, 5.4, "yolo26n.pt"),
}
FAST_P50 = {"faster_rcnn": 0.92, "mobilenet_faster_rcnn": 0.85, "light_rcnn": 0.80}  # R-CNN only
FAST_RECORD = {"preset": "fast", "label": "빠른 설정", "device": "cuda:0", "skipped": {},
               "summary": "Conv+BN 융합 · RPN NMS 전 후보 레벨당 500개 · RPN 점수 하한 0.05",
               "options": {"fuse_conv_bn": True, "channels_last": None, "rpn_pre_nms_top_n": 500,
                           "rpn_score_thresh": 0.05}}
STAGE_SHARES = {"faster_rcnn": (0.02, 0.05, 0.62, 0.11, 0.17, 0.03),
                "mobilenet_faster_rcnn": (0.04, 0.09, 0.38, 0.17, 0.27, 0.05),
                "light_rcnn": (0.05, 0.11, 0.34, 0.20, 0.24, 0.06)}


def four_model_reports(models=FOUR_MODELS) -> list[dict]:
    """Schema-v2 reports of one T4 comparison run: every model's baseline accuracy, latency and
    (R-CNN) stages, plus the R-CNNs' 'fast' preset with a paired p50 CI."""
    from light_rcnn.reporting import (
        COLD_START_FIRST_FORWARD,
        COLD_START_FIRST_PREDICT,
        MODELS,
        SCOPE_HOST_TO_DETECTIONS,
        build_report,
        latency_summary,
    )

    found = []
    for variant in ("baseline", "fast"):
        for model, (ap, p50, params, gflops, weights) in models.items():
            if variant == "fast" and model not in FAST_P50:
                continue
            speed = FAST_P50[model] if variant == "fast" else 1.0
            ap -= 0.001 if variant == "fast" else 0.0
            common = {"precision": "fp32", "image_size": 640, "batch_size": 1, "environment": T4,
                      "weights": weights, "weights_sha256": model[:4] * 16, "comparison_run": RUN,
                      "model_label": "Light R-CNN v3" if model == "light_rcnn" else MODELS[model]}
            if variant == "fast":
                common.update(variant="fast", variant_label="빠른 설정",
                              inference_options=FAST_RECORD)
            accuracy = {"evaluator": "pycocotools", "images_evaluated": 5000, "images_total": 5000,
                        "is_partial": False, "image_ids_sha256": "ids", "score_threshold": 0.001,
                        "dataset": {"annotations_sha256": "ann"}, "AP_50_95": ap,
                        "AP_50": ap + 0.19, "AP_75": ap + 0.02, "AP_small": ap - 0.15,
                        "AP_medium": ap + 0.03, "AP_large": ap + 0.16, "AR_100": ap + 0.17,
                        "per_category_ap": {"person": ap + 0.2, "car": ap + 0.05, "dog": ap + 0.1,
                                            "cup": ap - 0.08, "book": ap - 0.17}}
            found.append(build_report("accuracy", model, **common, accuracy=accuracy))
            center = p50 * speed
            samples = [center * (1 + 0.03 * ((index * 37) % 11 - 5) / 5)
                       + (center * 0.4 if index % 97 == 0 else 0.0) for index in range(500)]
            latency = latency_summary(samples, cold_start_ms=center * 40)
            latency.update(scope_id=SCOPE_HOST_TO_DETECTIONS, warmup=50, iterations=500,
                           image_count=64, score_threshold=0.25, max_detections=100,
                           memory_format="contiguous", mean_detections=3.2,
                           cold_start_scope=(COLD_START_FIRST_PREDICT if model == "yolo26"
                                             else COLD_START_FIRST_FORWARD))
            if variant == "fast":
                change = (speed - 1) * 100
                latency["paired"] = {"baseline": {"variant": "baseline",
                                                  "ci95_pct": [change - 1.2, change + 1.1]}}
            found.append(build_report("latency", model, **common, latency=latency,
                                      model_stats={"parameters_m": params, "gflops": gflops}))
            if model in STAGE_SHARES:
                stages = [{"stage": stage, "mean_ms": center * share, "p50_ms": center * share,
                           "p95_ms": center * share * 1.1, "share": share}
                          for stage, share in zip(("input", "preprocess", "backbone", "rpn",
                                                   "roi_heads", "postprocess"),
                                                  STAGE_SHARES[model])]
                found.append(build_report("stages", model, **common,
                                          stages={"items": stages, "iterations": 200}))
    return found


def write_four_model_results(root: Path, models=FOUR_MODELS) -> None:
    from light_rcnn.reporting import write_report

    for report in four_model_reports(models):
        variant = report.get("variant", "baseline")
        write_report(report, root / "reports" / f"{report['model']}_{report['kind']}_{variant}.json")


def test_reports_know_the_fourth_model():
    from light_rcnn import reporting
    from light_rcnn.reporting import MODEL_ORDER, MODELS, identify_model, normalize

    assert MODEL_ORDER == ["faster_rcnn", "mobilenet_faster_rcnn", "light_rcnn", "yolo26"]
    assert MODELS[KIND] == "MobileNetV3 Faster R-CNN"
    for raw, model in (({"model": KIND}, KIND),
                       ({"model_name": "Faster R-CNN MobileNetV3-Large-FPN"}, KIND),
                       ({"architecture": "fasterrcnn_mobilenet_v3_large_fpn"}, KIND),
                       ({"model_name": "Faster R-CNN ResNet50-FPN"}, "faster_rcnn"),
                       ({"model_name": "Light R-CNN (MobileNetV3 + BiFPN)"}, "light_rcnn")):
        assert identify_model(raw) == model
    reports = [r for index, raw in enumerate(four_model_reports())
               for r in normalize(json.loads(json.dumps(raw)), f"r{index}")]
    baseline = reporting.select_reports(reports, variant="baseline")
    assert {model for _, model in baseline} == set(MODELS)
    assert ("stages", KIND) in baseline and ("stages", "yolo26") not in baseline
    assert not [w for w in reporting.comparability_warnings(baseline) if w.level == "error"]
    fast = reporting.select_reports(reports, variant="fast")
    before = reporting.select_baseline(reports, fast)
    assert before[("latency", KIND)] is baseline[("latency", KIND)]
    assert not reporting.improvement_warnings(before, fast)
    # A mismatch on the new model is caught like any other model's.
    other = [r for r in reports if not (r.model == KIND and r.kind == "accuracy")]
    raw = next(r for r in four_model_reports() if r["model"] == KIND and r["kind"] == "accuracy")
    raw["accuracy"].update(images_evaluated=500, is_partial=True)
    picked = reporting.select_reports([*other, *normalize(raw, "partial")], variant="baseline")
    messages = [w.message for w in reporting.comparability_warnings(picked)]
    assert any("평가한 이미지 수가 다릅니다" in m for m in messages)
    assert any("부분 평가 결과입니다 (MobileNetV3 Faster R-CNN)" in m for m in messages)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_chart_and_table_carries_the_fourth_model(theme):
    from light_rcnn import viz
    from light_rcnn.reporting import MODEL_ORDER, MODELS, normalize, select_baseline, select_reports

    assert viz.model_color(KIND, theme) == {"light": "#eda100", "dark": "#c98500"}[theme]
    assert len({viz.model_color(model, theme) for model in MODEL_ORDER}) == 4
    assert viz.MODEL_SYMBOLS[KIND] == "triangle-up"
    assert len({viz.MODEL_SYMBOLS[model] for model in MODEL_ORDER}) == 4
    reports = [r for index, raw in enumerate(four_model_reports())
               for r in normalize(raw, f"r{index}")]
    selected = select_reports(reports, variant="baseline")
    targets = {"latency_ms": 15, "latency_percentile": "p50", "ap_metric": "AP_50_95", "ap": 0.6}
    figures = viz.build_all(selected, targets, theme)
    assert not any(viz.is_empty(fig) for fig in figures.values())
    name = MODELS[KIND]
    tradeoff = figures["tradeoff"]
    assert [trace.name for trace in tradeoff.data] == [MODELS[model] for model in MODEL_ORDER]
    marker = next(trace for trace in tradeoff.data if trace.name == name)
    assert marker.marker.symbol == "triangle-up" and list(marker.text) == [name]
    assert marker.marker.size > tradeoff.data[0].marker.size  # a triangle covers less area
    assert marker.textfont.color == viz.THEMES[theme]["ink"]  # text in ink, never the series color
    # Light R-CNN sits just below-left of MobileNet: its label moves under its marker.
    assert {trace.name: trace.textposition for trace in tradeoff.data} == {
        "Vanilla Faster R-CNN": "top center", name: "top center",
        "Light R-CNN": "bottom center", "YOLO26": "top center"}
    lines = [trace for trace in figures["latency_stability"].data if trace.mode == "lines+markers"]
    assert [trace.marker.symbol for trace in lines] == [viz.MODEL_SYMBOLS[m] for m in MODEL_ORDER]
    assert all(trace.marker.maxdisplayed == viz.STABILITY_MARKERS for trace in lines)
    assert name in figures["stages"].layout.yaxis.categoryarray
    assert name in figures["efficiency"].layout.yaxis.categoryarray
    assert name in figures["latency_percentiles"].layout.yaxis.categoryarray
    assert {trace.name for trace in figures["accuracy"].data if trace.showlegend} == set(MODELS.values())
    assert [row["모델"] for row in viz.summary_rows(selected)] == [MODELS[m] for m in MODEL_ORDER]
    fast = select_reports(reports, variant="fast")
    improvement = viz.improvement_figure(select_baseline(reports, fast), fast, theme)
    assert list(improvement.layout.yaxis.categoryarray)[::-1] == [
        MODELS[m] for m in MODEL_ORDER if m != "yolo26"]
    rows = viz.improvement_rows(select_baseline(reports, fast), fast)
    mobile = {row["지표"]: row for row in rows if row["모델"] == name}
    assert mobile["p50 (ms)"]["판정"] == "좋아짐" and mobile["p50 (ms)"]["95% CI"] == "-16.2 ~ -13.9%"
    page = viz.html_report(figures, viz.summary_rows(selected), [], selected, theme, "보고서")
    assert page.count(name) >= 2  # summary table and the sources table
    # The sources table follows the display order, not the alphabetical key order.
    assert [key for key, _ in viz.in_display_order(selected)][:4] == [
        ("accuracy", model) for model in MODEL_ORDER]
    sources = page[page.index("측정 출처"):]
    assert [sources.index(MODELS[m]) for m in MODEL_ORDER] == sorted(
        sources.index(MODELS[m]) for m in MODEL_ORDER)


def test_tradeoff_direct_labels_avoid_labels_markers_and_whiskers():
    from light_rcnn import viz

    assert viz._label_positions([(5.0, 10.0, 0.0, "A"), (50.0, 60.0, 0.0, "B")], 60.0, 70.0) == [
        "top center", "top center"]
    # Three models within 2 %p at one latency: above, below, then beside the stack (left, away
    # from the p95 whiskers).
    tight = [(30.0, 31.0, 2.0, "MobileNetV3 Faster R-CNN"), (30.0, 29.0, 2.0, "Light R-CNN v3"),
             (30.0, 30.0, 2.0, "YOLO26")]
    assert viz._label_positions(tight, 60.0, 70.0) == ["top center", "bottom center",
                                                       "middle left"]


def dashboard(folder: Path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    app.run()
    app.text_input[0].set_value(str(folder)).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def card_titles(app) -> list[str]:
    return [m.value.rsplit("</span>", 1)[-1] for m in app.markdown if m.value.endswith("**")]


def tab(app, label):
    return next(item for item in app.tabs if item.label == label)


def chart_specs(block) -> dict[str, dict]:
    specs = [json.loads(chart.proto.spec) for chart in block.get("plotly_chart")]
    return {spec["layout"]["title"]["text"].strip("<b>/"): spec for spec in specs}


def test_dashboard_renders_four_models_on_every_tab(tmp_path, monkeypatch):
    from light_rcnn import viz

    write_four_model_results(tmp_path)
    monkeypatch.setenv("LIGHT_RCNN_RESULTS_DIR", str(tmp_path / "missing"))
    pages, original = [], viz.html_report
    monkeypatch.setattr(viz, "html_report", lambda *a, **k: pages.append(original(*a, **k)) or pages[-1])
    app = dashboard(tmp_path)
    name = "MobileNetV3 Faster R-CNN"
    assert card_titles(app) == ["**Vanilla Faster R-CNN**", f"**{name}**", "**Light R-CNN v3**",
                                "**YOLO26**"]
    assert [m.value for m in app.metric if m.label.startswith("AP")] == [
        "35.2%", "28.9%", "27.1%", "39.2%"]
    assert [m.value for m in app.metric if m.label.endswith("지연")] == [
        "48.0 ms", "14.0 ms", "9.0 ms", "4.5 ms"]
    assert not app.error, [e.value for e in app.error]
    assert [t.label for t in app.tabs] == ["한눈에 보기", "개선 전/후", "지연 시간", "정확도", "병목 분석",
                                          "모델 크기", "데이터·출처"]
    tradeoff = chart_specs(tab(app, "한눈에 보기"))["정확도 대 속도"]
    assert [trace["name"] for trace in tradeoff["data"]] == [
        "Vanilla Faster R-CNN", name, "Light R-CNN v3", "YOLO26"]
    for label, titles in (("지연 시간", {"지연 시간 분포", "측정 안정성"}),
                          ("정확도", {"COCO 정확도 세부 지표", "클래스별 AP"}),
                          ("병목 분석", {"단계별 처리 시간"}), ("모델 크기", {"모델 크기와 연산량"})):
        block = tab(app, label)
        specs = chart_specs(block)
        assert set(specs) == titles, label
        assert all(name in json.dumps(spec["data"], ensure_ascii=False) for spec in specs.values())
        frames = [df.value for df in block.dataframe]
        assert all(name in set(frame["모델"]) for frame in frames if "모델" in frame), label
    sources = next(df.value for df in tab(app, "데이터·출처").dataframe if "파일" in df.value.columns)
    assert set(sources[sources["사용"]]["모델"]) == {"Vanilla Faster R-CNN", name, "Light R-CNN",
                                                   "YOLO26"}
    assert name in pages[-1]

    # The 'fast' preset: every R-CNN, MobileNet included, before -> after with its paired CI.
    next(box for box in app.sidebar.selectbox if box.label == "추론 설정").set_value("fast").run()
    assert not app.exception, [e.value for e in app.exception]
    improvement = tab(app, "개선 전/후")
    assert not improvement.error and not improvement.warning
    spec = chart_specs(improvement)["개선 전/후 비교"]
    assert name in spec["layout"]["yaxis"]["categoryarray"]
    deltas = next(df.value for df in improvement.dataframe if "판정" in df.value.columns)
    mobile = deltas[deltas["모델"] == name].set_index("지표")
    assert mobile.loc["p50 (ms)", "95% CI"] == "-16.2 ~ -13.9%"
    assert mobile.loc["p50 (ms)", "판정"] == "좋아짐"
    assert "이 추론 설정으로 측정한 결과 없음" in " ".join(c.value for c in app.caption)  # YOLO26


def test_dashboard_without_mobilenet_results_still_renders(tmp_path, monkeypatch):
    """Result folders from before the fourth model: its card says so, nothing else changes."""
    write_four_model_results(tmp_path, {k: v for k, v in FOUR_MODELS.items() if k != KIND})
    monkeypatch.setenv("LIGHT_RCNN_RESULTS_DIR", str(tmp_path / "missing"))
    app = dashboard(tmp_path)
    assert card_titles(app)[1] == "**MobileNetV3 Faster R-CNN**"
    assert [m.value for m in app.metric if m.label.endswith("지연")] == ["48.0 ms", "9.0 ms", "4.5 ms"]
    assert "결과 없음" in [c.value for c in app.caption]
    tradeoff = chart_specs(tab(app, "한눈에 보기"))["정확도 대 속도"]
    assert len(tradeoff["data"]) == 3
    assert not app.error, [e.value for e in app.error]
    assert any("결과가 없는 모델: MobileNetV3 Faster R-CNN" in m.value for m in app.markdown)
