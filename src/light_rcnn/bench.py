"""Helpers shared by the R-CNN measurement scripts (benchmark, profile_stages, evaluate_coco)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Recorded in the report when --weights is omitted and build_from_config loads library weights.
DEFAULT_WEIGHTS = {
    "faster_rcnn": "torchvision_coco_v1",
    "light_rcnn": "imagenet_mobilenet_v3_large (untrained detection heads)",
}
# Configs saved into run folders by train.py predate the score/detection keys of configs/coco.yaml.
COMPARISON_DEFAULTS = {
    "warmup": 50,
    "iterations": 500,
    "image_count": 64,
    "score_threshold": 0.001,
    "max_detections": 100,
    "latency_score_threshold": 0.25,
}


def comparison_setting(config: dict, key: str):
    """Read ``comparison.<key>`` from a config, falling back to the shared default."""
    return (config.get("comparison") or {}).get(key, COMPARISON_DEFAULTS[key])


def resolve_device(name: str, purpose: str = "measuring") -> torch.device:
    """Accept 'cpu', 'cuda', 'cuda:N' or a bare GPU index; fall back to CPU without CUDA."""
    try:
        device = torch.device(f"cuda:{name}" if name.isdigit() else name)
    except RuntimeError as exc:
        raise ValueError(f"unknown device {name!r}; use cpu, cuda, cuda:N or a GPU index") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA is unavailable; {purpose} on cpu instead.", file=sys.stderr)
        return torch.device("cpu")
    return device


def load_images(source: str | Path, count: int) -> tuple[list[Path], list[torch.Tensor]]:
    """Decode one image file, or the first ``count`` images of a folder, once into uint8 CHW
    tensors kept in host memory.

    Uses Pillow rather than torchvision.io, whose decoders are deprecated since torchvision 0.29.
    """
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"Unsupported image file: {source}")
        paths = [source]
    else:
        paths = sorted(path for path in source.rglob("*")
                       if path.suffix.lower() in IMAGE_EXTENSIONS)[:count]
    if not paths:
        raise SystemExit(f"No images found under {source}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
        images.append(torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous())
    return paths, images


def load_comparison_model(kind: str, config: dict, weights: str | None = None,
                          image_size: int | None = None) -> torch.nn.Module:
    """Build a compared detector the way the team's training code does.

    Light R-CNN checkpoints carry the config they were trained with, so their architecture comes
    from the checkpoint (only the input size follows ``image_size`` or the YAML config); without a
    checkpoint the YAML config decides. The vanilla baseline uses torchvision's COCO weights unless
    a checkpoint is given.
    """
    from light_rcnn.model.detector import build_detector, light_rcnn_from_config

    size = image_size or config["image_size"]
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True) if weights else {}
    if kind == "light_rcnn":
        source = checkpoint.get("config") or config
        model = light_rcnn_from_config(dict(source, image_size=size), pretrained=weights is None)
    elif kind == "faster_rcnn":
        model = build_detector(kind, pretrained=weights is None, num_classes=config["num_classes"],
                               min_size=size, max_size=size)
    else:
        raise ValueError(f"Unknown detector kind: {kind}")
    if weights:
        model.load_state_dict(checkpoint.get("model", checkpoint))
    return model


def set_detection_limits(model: torch.nn.Module, score_threshold: float, max_detections: int) -> None:
    """Apply one score floor and detection cap so every compared model is scored alike."""
    model.roi_heads.score_thresh = score_threshold
    model.roi_heads.detections_per_img = max_detections
