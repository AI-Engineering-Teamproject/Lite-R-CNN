"""Inference-time options for the compared R-CNN detectors, applied to one model instance.

Light R-CNN v2/v3 and the torchvision Faster R-CNN baseline share torchvision's GeneralizedRCNN
pipeline, so every option is a setting or an instance-level swap on that pipeline and one preset
applies to every compared R-CNN alike: apply the same preset to the baseline for a fair table.
Nothing is retrained; Conv+BN folding is exact.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops.poolers import _setup_scales

from light_rcnn.model.deploy import prepare_for_inference, use_channels_last
from light_rcnn.postprocess import (
    MATRIX_NMS_TOP_K,
    NMS_METHODS,
    SOFT_NMS_MIN_SCORE,
    install_postprocess,
)
from light_rcnn.reporting import cuda_graph_note  # noqa: F401 - re-exported; lives torch-free

COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
# Convolution/linear-heavy modules run under fp16 autocast. Anchors, box decoding, RoIAlign and
# NMS stay float32: batched_nms offsets boxes by class index * max coordinate, which float16
# cannot resolve (its spacing is 32 above 32768).
FP16_MODULES = ("backbone", "rpn.head", "roi_heads.box_head", "roi_heads.box_predictor")
CUSTOM_LABEL = "사용자 지정 설정"


@dataclass(frozen=True)
class InferenceOptions:
    """What to change at inference time; None keeps the model's own value."""

    fuse_conv_bn: bool = False
    channels_last: bool | None = False  # None = deploy.use_channels_last (Ultralytics' rule)
    rpn_pre_nms_top_n: int | None = None  # RPN candidates per level before NMS
    rpn_post_nms_top_n: int | None = None  # RPN proposals kept after NMS
    rpn_score_thresh: float | None = None  # RPN objectness floor before NMS
    box_score_thresh: float | None = None
    box_nms_thresh: float | None = None
    detections_per_img: int | None = None
    nms: str = "hard"  # 'hard' (stock), 'soft' (exact Gaussian Soft-NMS), 'matrix' (Matrix NMS)
    soft_nms_sigma: float = 0.5  # Gaussian sigma of Soft-NMS and Matrix NMS
    soft_nms_min_score: float = SOFT_NMS_MIN_SCORE  # Soft-NMS drops decayed scores at or below
    fixed_canvas: int | None = None  # pad every batch to N x N and pin the RoI scales
    fp16: bool = False  # CUDA float16 autocast of FP16_MODULES
    cuda_graph: bool = False  # replay backbone body+neck from a CUDA graph (CUDA only)
    compile_backbone: str | None = None  # torch.compile mode for the backbone

    def __post_init__(self):
        for name in ("rpn_pre_nms_top_n", "rpn_post_nms_top_n", "detections_per_img",
                     "fixed_canvas"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, numbers.Integral) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
            if value is not None:  # plain Python numbers keep records JSON-serializable
                object.__setattr__(self, name, int(value))
        for name in ("rpn_score_thresh", "box_score_thresh", "box_nms_thresh"):
            value = getattr(self, name)
            if value is not None and not (isinstance(value, numbers.Real) and 0 <= value <= 1):
                raise ValueError(f"{name} must be within [0, 1] or None, got {value!r}")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.channels_last not in (True, False, None):
            raise ValueError(f"channels_last must be True, False or None, got {self.channels_last!r}")
        if self.nms not in NMS_METHODS:
            raise ValueError(f"nms must be one of {', '.join(NMS_METHODS)}, got {self.nms!r}")
        if not (isinstance(self.soft_nms_sigma, numbers.Real) and self.soft_nms_sigma > 0):
            raise ValueError(f"soft_nms_sigma must be positive, got {self.soft_nms_sigma!r}")
        floor = self.soft_nms_min_score  # NaN fails the range check
        if isinstance(floor, bool) or not (isinstance(floor, numbers.Real) and 0 <= floor <= 1):
            raise ValueError(f"soft_nms_min_score must be within [0, 1], got {floor!r}")
        object.__setattr__(self, "soft_nms_min_score", float(floor))
        if self.compile_backbone is not None and self.compile_backbone not in COMPILE_MODES:
            raise ValueError(f"compile_backbone must be one of {', '.join(COMPILE_MODES)}, "
                             f"got {self.compile_backbone!r}")
        if (self.cuda_graph or self.compile_backbone) and self.fixed_canvas is None:
            raise ValueError("cuda_graph and compile_backbone need fixed_canvas "
                             "(one backbone input shape)")
        if self.cuda_graph and self.compile_backbone:
            raise ValueError("use cuda_graph or compile_backbone, not both")


PRESETS: dict[str, InferenceOptions] = {
    "baseline": InferenceOptions(),
    "deploy": InferenceOptions(fuse_conv_bn=True, channels_last=None),
    "fast": InferenceOptions(fuse_conv_bn=True, channels_last=None, rpn_pre_nms_top_n=500,
                             rpn_score_thresh=0.05),
    "accurate": InferenceOptions(fuse_conv_bn=True, channels_last=None, fixed_canvas=640,
                                 nms="soft"),
    "accurate_fast": InferenceOptions(fuse_conv_bn=True, channels_last=None, fixed_canvas=640,
                                      nms="matrix"),
    "gpu_fp16": InferenceOptions(fuse_conv_bn=True, channels_last=None, fp16=True),
    "gpu_graph": InferenceOptions(fuse_conv_bn=True, fixed_canvas=640, cuda_graph=True),
}
PRESET_LABELS: dict[str, str] = {
    "baseline": "개선 전 (기본 설정)",
    "deploy": "배포 기본 (Conv+BN 융합·메모리 형식)",
    "fast": "빠른 설정",
    "accurate": "정확도 우선",
    "accurate_fast": "정확도 우선 (빠른 NMS)",
    "gpu_fp16": "GPU FP16 (CUDA 전용)",
    "gpu_graph": "GPU CUDA 그래프 (CUDA 전용)",
}
_FIELDS = frozenset(field.name for field in dataclasses.fields(InferenceOptions))


def resolve_options(name_or_options: str | InferenceOptions | Mapping | None = None,
                    **overrides) -> InferenceOptions:
    """A preset name, InferenceOptions or a mapping (e.g. a report's ``options``) plus overrides.

    Overrides set to None are ignored, so unset command-line flags keep the preset's value.
    """
    if name_or_options is None:
        options = PRESETS["baseline"]
    elif isinstance(name_or_options, InferenceOptions):
        options = name_or_options
    elif isinstance(name_or_options, str):
        key = name_or_options.strip().lower().replace("-", "_")
        if key not in PRESETS:
            raise ValueError(f"unknown inference preset {name_or_options!r}; "
                             f"choose from {', '.join(PRESETS)}")
        options = PRESETS[key]
    elif isinstance(name_or_options, Mapping):
        unknown = sorted(set(name_or_options) - _FIELDS)
        if unknown:
            raise TypeError(f"unknown inference option(s): {', '.join(unknown)}")
        options = InferenceOptions(**name_or_options)
    else:
        raise TypeError("expected a preset name, InferenceOptions or a mapping, "
                        f"got {type(name_or_options).__name__}")
    changes = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(changes) - _FIELDS)
    if unknown:
        raise TypeError(f"unknown inference option(s): {', '.join(unknown)}")
    return dataclasses.replace(options, **changes) if changes else options


def options_to_dict(options: InferenceOptions) -> dict[str, Any]:
    return dataclasses.asdict(options)


def apply_inference_options(model: nn.Module,
                            options: str | InferenceOptions | Mapping | None = "baseline",
                            device: torch.device | str | None = None) -> dict[str, Any]:
    """Apply ``options`` to this model instance in place; returns a JSON-serializable record.

    The model is moved to ``device`` (default: where its parameters are) in eval mode. Options
    the device cannot run (fp16 and cuda_graph off CUDA) are left out and listed under
    ``skipped``. ``applied`` holds the effective settings read back from the model. Apply once,
    to a freshly loaded model: Conv+BN folding cannot be undone. CUDA graph capture and
    torch.compile happen on the first forward, so that call's latency includes them; the
    record's ``cuda_graph_status`` is updated in place by that capture and its self-check, and
    ``cuda_graph_note`` reads it.
    """
    options = resolve_options(options)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(device).eval()
    rpn, heads = model.rpn, model.roi_heads
    applied: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    extra: dict[str, Any] = {}

    if options.fuse_conv_bn:
        layout = prepare_for_inference(model, device, options.channels_last)
        layout["conv_bn_fused"] = not any(isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d))
                                          for module in model.modules())
    else:
        channels_last = (use_channels_last(device) if options.channels_last is None
                         else options.channels_last)
        if channels_last:
            model.to(memory_format=torch.channels_last)
        layout = {"conv_bn_fused": False,
                  "memory_format": "channels_last" if channels_last else "contiguous"}
    applied.update(layout)

    if options.rpn_pre_nms_top_n is not None:
        rpn._pre_nms_top_n = dict(rpn._pre_nms_top_n, testing=options.rpn_pre_nms_top_n)
    if options.rpn_post_nms_top_n is not None:
        rpn._post_nms_top_n = dict(rpn._post_nms_top_n, testing=options.rpn_post_nms_top_n)
    if options.rpn_score_thresh is not None:
        rpn.score_thresh = options.rpn_score_thresh
    if options.box_score_thresh is not None:
        heads.score_thresh = options.box_score_thresh
    if options.box_nms_thresh is not None:
        heads.nms_thresh = options.box_nms_thresh
    if options.detections_per_img is not None:
        heads.detections_per_img = options.detections_per_img
    applied.update(
        rpn_pre_nms_top_n_test=int(rpn._pre_nms_top_n["testing"]),
        rpn_post_nms_top_n_test=int(rpn._post_nms_top_n["testing"]),
        rpn_score_thresh=float(rpn.score_thresh),
        box_score_thresh=float(heads.score_thresh),
        box_nms_thresh=float(heads.nms_thresh),
        detections_per_img=int(heads.detections_per_img),
        nms=options.nms,
    )
    if options.nms != "hard":
        install_postprocess(heads, options.nms, options.soft_nms_sigma, options.soft_nms_min_score)
        applied["soft_nms_sigma"] = float(options.soft_nms_sigma)
    if options.nms == "soft":
        applied["soft_nms_min_score"] = options.soft_nms_min_score
    if options.nms == "matrix":
        applied["matrix_nms_top_k"] = MATRIX_NMS_TOP_K

    applied["fixed_canvas"] = None
    if options.fixed_canvas is not None:
        stride = int(model.transform.size_divisible)
        side = math.ceil(options.fixed_canvas / stride) * stride
        model.transform.batch_images = _canvas_batch_images(side)
        applied["fixed_canvas"] = [side, side]
        if isinstance(heads.box_roi_pool, MultiScaleRoIAlign):
            applied["roi_scales"] = pin_roi_scales(model, side)

    applied["fp16"] = False
    if options.fp16:
        if device.type == "cuda":
            for name in FP16_MODULES:
                module = model.get_submodule(name)
                if not isinstance(module.forward, _Fp16Forward):
                    module.forward = _Fp16Forward(module.forward)
            applied.update(fp16=True, fp16_modules=list(FP16_MODULES))
        else:
            skipped["fp16"] = _skip_reason(device)

    applied["cuda_graph"] = False
    if options.cuda_graph:
        if device.type == "cuda":
            backbone = model.backbone
            if not isinstance(backbone.forward, _GraphedForward):
                backbone.forward = _GraphedForward(backbone.forward)
            applied["cuda_graph"] = True
            extra["cuda_graph_status"] = backbone.forward.status  # filled in on capture
        else:
            skipped["cuda_graph"] = _skip_reason(device)

    applied["compile_backbone"] = None
    if options.compile_backbone is not None:
        if getattr(model.backbone, "_compiled_call_impl", None) is None:
            model.backbone.compile(mode=options.compile_backbone, dynamic=False)
        applied["compile_backbone"] = options.compile_backbone

    preset = next((name for name, value in PRESETS.items() if value == options), None)
    return {
        "preset": preset,
        "label": PRESET_LABELS.get(preset, CUSTOM_LABEL),
        "summary": _summary(options, applied, skipped, device),
        "device": str(device),
        "options": options_to_dict(options),
        "applied": applied,
        "skipped": skipped,
        **extra,
    }


def pin_roi_scales(model: nn.Module, canvas: int) -> list[float]:
    """Set the RoI pooler's level scales from the backbone features of one canvas x canvas input.

    torchvision's MultiScaleRoIAlign infers the scales once, on its first call, from feature size
    / largest unpadded image height, and caches them. On a fixed canvas a short first image
    (640x426 resizes to height 426, below ~453) makes every scale 2x too large and RoIAlign
    samples the wrong places for every later image. Measured on the canvas itself the scales are
    the true 1/stride, right for any input. Works for Light R-CNN's CompressedMultiScaleRoIAlign
    (its projections keep the map size) and torchvision's pooler alike.
    """
    pool = model.roi_heads.box_roi_pool
    parameter = next(model.parameters())
    with torch.no_grad():
        features = model.backbone(parameter.new_zeros((1, 3, canvas, canvas)))
    maps = [features[name] for name in pool.featmap_names]
    pool.scales, pool.map_levels = _setup_scales(maps, [(canvas, canvas)], pool.canonical_scale,
                                                 pool.canonical_level)
    return [float(scale) for scale in pool.scales]


def _canvas_batch_images(side: int):
    """GeneralizedRCNNTransform.batch_images that pads every batch to side x side.

    A resized image larger than the canvas widens that batch to it (rounded up to
    ``size_divisible``, as torchvision pads).
    """
    def batch_images(images: list[torch.Tensor], size_divisible: int = 32) -> torch.Tensor:
        height = max(side, *(int(image.shape[-2]) for image in images))
        width = max(side, *(int(image.shape[-1]) for image in images))
        height = math.ceil(height / size_divisible) * size_divisible
        width = math.ceil(width / size_divisible) * size_divisible
        batched = images[0].new_zeros((len(images), images[0].shape[0], height, width))
        for index, image in enumerate(images):
            batched[index, :, : image.shape[-2], : image.shape[-1]].copy_(image)
        return batched

    return batch_images


def _skip_reason(device: torch.device) -> str:
    return f"skipped ({'CPU' if device.type == 'cpu' else device.type})"


def _to_float32(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return type(value)((key, _to_float32(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return type(value)(_to_float32(item) for item in value)
    return value


class _Fp16Forward:
    """Run a module's forward under CUDA float16 autocast and pass float32 tensors on."""

    def __init__(self, forward):
        self.forward = forward

    def __call__(self, *args, **kwargs):
        # cache_enabled=False keeps the autocast weight-cast cache out of CUDA graph captures.
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            output = self.forward(*args, **kwargs)
        return _to_float32(output)


class _GraphedForward:
    """Replay a backbone forward from one CUDA graph captured on its first CUDA call.

    From the investigation's GraphedBackbone (not validated on a GPU there or here): a static
    input buffer, three warmup calls on a side stream, capture, then one replay checked against
    eager; a mismatch keeps the backbone eager for good. Other input shapes, autograd and
    non-CUDA inputs run eagerly. Outputs are the graph's static buffers, valid until the next
    call, which is how GeneralizedRCNN consumes them. ``status`` is shared with the record.
    """

    def __init__(self, forward):
        self.forward = forward
        self.graph = self.static_input = self.static_output = None
        self.status: dict[str, Any] = {"status": "pending (captured on the first CUDA call)",
                                       "replays": 0, "eager_calls": 0}

    def __call__(self, images: torch.Tensor):
        if (images.device.type != "cuda" or torch.is_grad_enabled()
                or self.status["status"].startswith("disabled")):
            self.status["eager_calls"] += 1
            return self.forward(images)
        if self.graph is None:
            self._capture(images)
            if self.graph is None:
                self.status["eager_calls"] += 1
                return self.forward(images)
        if (images.shape != self.static_input.shape or images.dtype != self.static_input.dtype
                or images.stride() != self.static_input.stride()):
            self.status["eager_calls"] += 1
            return self.forward(images)
        self.static_input.copy_(images)
        self.graph.replay()
        self.status["replays"] += 1
        return OrderedDict(self.static_output)

    def _capture(self, images: torch.Tensor) -> None:
        device = images.device
        with torch.inference_mode(False):  # a normal tensor, so copy_ works in any grad mode
            static_input = torch.empty_strided(images.shape, images.stride(), dtype=images.dtype,
                                               device=device)
        static_input.copy_(images)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.forward(static_input)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.forward(static_input)
        reference = self.forward(static_input)
        graph.replay()
        torch.cuda.synchronize(device)
        diff = max(float((reference[key] - static_output[key]).abs().max()) for key in reference)
        scale = max(float(reference[key].abs().max()) for key in reference)
        self.status.update(input_shape=list(images.shape), self_check_max_abs_diff=diff)
        if diff <= 1e-3 * max(1.0, scale):
            self.graph, self.static_input, self.static_output = graph, static_input, static_output
            self.status["status"] = "captured"
        else:
            self.status["status"] = "disabled (graph replay differs from eager)"


_SUMMARY_NAMES = {"fp16": "FP16", "cuda_graph": "CUDA 그래프"}


def _summary(options: InferenceOptions, applied: dict, skipped: dict,
             device: torch.device) -> str:
    """One Korean line naming what changed from the model's defaults."""
    parts = []
    if applied["conv_bn_fused"]:
        parts.append("Conv+BN 융합")
    if applied["memory_format"] == "channels_last":
        parts.append("channels_last")
    if options.rpn_pre_nms_top_n is not None:
        parts.append(f"RPN NMS 전 후보 레벨당 {options.rpn_pre_nms_top_n}개")
    if options.rpn_post_nms_top_n is not None:
        parts.append(f"RPN 제안 {options.rpn_post_nms_top_n}개")
    if options.rpn_score_thresh is not None:
        parts.append(f"RPN 점수 하한 {options.rpn_score_thresh:g}")
    if options.box_score_thresh is not None:
        parts.append(f"검출 점수 하한 {options.box_score_thresh:g}")
    if options.box_nms_thresh is not None:
        parts.append(f"NMS IoU {options.box_nms_thresh:g}")
    if options.detections_per_img is not None:
        parts.append(f"이미지당 최대 {options.detections_per_img}개")
    if options.nms == "soft":
        parts.append(f"Soft-NMS (σ={options.soft_nms_sigma:g}, "
                     f"감쇠 하한 {options.soft_nms_min_score:g})")
    elif options.nms == "matrix":
        parts.append(f"Matrix NMS (σ={options.soft_nms_sigma:g})")
    if applied["fixed_canvas"]:
        parts.append("{}×{} 고정 캔버스".format(*applied["fixed_canvas"]))
    if applied["fp16"]:
        parts.append("FP16 autocast")
    if applied["cuda_graph"]:
        parts.append("CUDA 그래프")
    if applied["compile_backbone"]:
        parts.append(f"torch.compile ({applied['compile_backbone']})")
    where = "CPU" if device.type == "cpu" else device.type
    parts.extend(f"{_SUMMARY_NAMES[name]} 미적용 ({where})" for name in skipped)
    return " · ".join(parts) or "모델 기본 설정"
