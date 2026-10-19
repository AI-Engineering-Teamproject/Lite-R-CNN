"""Adapters reuse only the existing Lite model builder, never its evaluation scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)

from light_rcnn.comparison.protocol import Prediction, file_hash, prepare_image
from light_rcnn.model.detector import light_rcnn_from_config

MODEL_NAMES = {
    "lite_rcnn": "Lite R-CNN",
    "faster_rcnn": "Faster R-CNN ResNet50-FPN",
    "mobilenet_faster_rcnn": "Faster R-CNN MobileNetV3-Large-FPN",
    "yolo26": "YOLO26 nano",
}


def class_mapping(names: dict[int, str], categories: list[dict]) -> dict[int, int]:
    """Resolve by class name, so sparse COCO IDs are never treated as contiguous IDs."""
    original = {category["name"]: category["id"] for category in categories}
    if len(original) != len(categories) or set(names.values()) != set(original):
        raise ValueError("YOLO class names must match the annotation's complete COCO taxonomy")
    return {int(index): original[name] for index, name in names.items()}


class TorchvisionAdapter:
    def __init__(self, model, *, device, size, geometry, metadata):
        self.model = model.to(device).float().eval()
        self.device = torch.device(device)
        self.size = size
        self.geometry = geometry
        self.metadata = metadata
        self.metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        self.metadata["postprocess"] = "torchvision class-aware NMS; no containment suppression"
        self.metadata["inference_settings"] = {
            "score_threshold": model.roi_heads.score_thresh,
            "nms_threshold": model.roi_heads.nms_thresh,
            "max_detections": model.roi_heads.detections_per_img,
            "rpn_score_threshold": model.rpn.score_thresh,
            "rpn_nms_threshold": model.rpn.nms_thresh,
            "image_mean": list(model.transform.image_mean),
            "image_std": list(model.transform.image_std),
            "padding_stride": model.transform.size_divisible,
        }

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> Prediction:
        prepared, geometry = prepare_image(rgb, self.size, self.geometry)
        tensor = torch.from_numpy(prepared).permute(2, 0, 1).float().div(255).to(self.device)
        output = self.model([tensor])[0]
        return Prediction(output["boxes"], output["scores"], output["labels"]).restored(geometry)

    def inspect_geometry(self, rgb: np.ndarray) -> dict:
        captured = {}

        def hook(module, args, output):
            image_list = output[0]
            captured.update(
                model_input_nchw=list(image_list.tensors.shape),
                model_resized_hw=[list(shape) for shape in image_list.image_sizes],
                model_device=str(image_list.tensors.device),
                model_dtype=str(image_list.tensors.dtype),
            )

        handle = self.model.transform.register_forward_hook(hook)
        try:
            self.predict(rgb)
        finally:
            handle.remove()
        _, geometry = prepare_image(rgb, self.size, self.geometry)
        return {"shared_geometry": asdict(geometry), **captured}


class YoloAdapter:
    def __init__(
        self, yolo, *, categories, device, size, geometry, score, nms, max_detections, metadata
    ):
        self.yolo = yolo
        self.device = torch.device(device)
        self.size = size
        self.geometry = geometry
        self.mapping = class_mapping(dict(yolo.names), categories)
        self.metadata = metadata
        self.metadata["parameters"] = sum(p.numel() for p in yolo.model.parameters())
        definition = getattr(yolo.model, "yaml", {})
        self.metadata.update(
            model_definition=definition.get("yaml_file"),
            model_scale=definition.get("scale"),
            parameters_scope="original model before Ultralytics inference fusion",
        )
        end_to_end = bool(getattr(yolo.model, "end2end", False))
        self.metadata["end_to_end_head"] = end_to_end
        self.metadata["postprocess"] = (
            "Ultralytics native end-to-end head (NMS-free)"
            if end_to_end
            else "Ultralytics class-aware NMS"
        )
        self.options = {
            "imgsz": size,
            # torch.device also avoids older Ultralytics remapping cuda:N through visible devices.
            "device": self.device,
            "half": False,
            "conf": score,
            "iou": nms,
            "max_det": max_detections,
            "rect": False,
            "augment": False,
            "agnostic_nms": False,
            "batch": 1,
            "verbose": False,
            "save": False,
            "stream": False,
        }
        self.metadata["inference_settings"] = {**self.options, "device": str(self.device)}

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> Prediction:
        prepared, geometry = prepare_image(rgb, self.size, self.geometry)
        # Ultralytics ndarray input is BGR. Both adapters start with the same CPU RGB array.
        threads = torch.get_num_threads()
        try:
            results = self.yolo.predict(source=prepared[..., ::-1].copy(), **self.options)[0]
        finally:
            # Ultralytics select_device resets CPU thread counts during lazy backend setup.
            # Restore the shared setting before warmup/timing and before the next model runs.
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
        boxes = results.boxes
        labels = boxes.cls.detach().to(device="cpu", dtype=torch.int64)
        mapped = torch.tensor([self.mapping[int(label)] for label in labels], dtype=torch.int64)
        return Prediction(boxes.xyxy, boxes.conf, mapped).restored(geometry)

    def inspect_geometry(self, rgb: np.ndarray) -> dict:
        # First call creates AutoBackend; the second captures its actual input without timed hooks.
        if self.yolo.predictor is None:
            self.predict(rgb)
        captured = {}

        def hook(module, args):
            captured["model_input_nchw"] = list(args[0].shape)
            captured["model_device"] = str(args[0].device)
            captured["model_dtype"] = str(args[0].dtype)

        handle = self.yolo.predictor.model.register_forward_pre_hook(hook)
        try:
            self.predict(rgb)
        finally:
            handle.remove()
        _, geometry = prepare_image(rgb, self.size, self.geometry)
        return {
            "shared_geometry": asdict(geometry),
            **captured,
            "native_resize": "Ultralytics LetterBox, rect=False",
        }


def build_adapter(name: str, config: dict, categories: list[dict], root: Path):
    device, size, geometry = config["device"], config["image_size"], config["geometry"]
    settings = config["models"][name]
    metadata = {"name": MODEL_NAMES[name], "architecture": name}
    common = {"device": device, "size": size, "geometry": geometry, "metadata": metadata}
    if name == "yolo26":
        from ultralytics import YOLO

        path = (root / settings["weights"]).resolve()
        # Local weights are intentional: never silently substitute a different YOLO variant.
        model = YOLO(str(path), task="detect")
        metadata.update(weights=str(path), weights_sha256=file_hash(path))
        return YoloAdapter(
            model,
            categories=categories,
            score=config["score_threshold"],
            nms=config["nms_threshold"],
            max_detections=config["max_detections"],
            **common,
        )
    if name == "lite_rcnn":
        path = (root / settings["weights"]).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError("Lite checkpoint must contain model and training config")
        model_config = checkpoint["config"]
        model = light_rcnn_from_config(dict(model_config, image_size=size))
        model.load_state_dict(checkpoint["model"], strict=True)
        metadata.update(
            weights=str(path),
            weights_sha256=file_hash(path),
            checkpoint_epoch=checkpoint.get("epoch"),
            training_config=model_config,
        )
    else:
        builders = {
            "faster_rcnn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
            "mobilenet_faster_rcnn": (
                fasterrcnn_mobilenet_v3_large_fpn,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
            ),
        }
        builder, weights = builders[name]
        model = builder(weights=weights, min_size=size, max_size=size)
        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
        metadata.update(
            weights=str(weights), weights_url=weights.url, weights_sha256=file_hash(cached)
        )
    model.roi_heads.score_thresh = config["score_threshold"]
    model.roi_heads.nms_thresh = config["nms_threshold"]
    model.roi_heads.detections_per_img = config["max_detections"]
    # Proposal budgets are model defaults/checkpoint settings, not silently equalized.
    model.eval()
    metadata.update(
        rpn_pre_nms_test=model.rpn.pre_nms_top_n(), rpn_post_nms_test=model.rpn.post_nms_top_n()
    )
    return TorchvisionAdapter(model, **common)
