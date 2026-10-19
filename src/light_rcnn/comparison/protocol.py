"""Shared image selection, geometry and CPU prediction contract."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def select_ids(available: list[int], limit: int, seed: int) -> list[int]:
    ordered = sorted(available)
    if limit < 0:
        raise ValueError("Image limits must be non-negative")
    return random.Random(seed).sample(ordered, limit) if 0 < limit < len(ordered) else ordered


class CocoImages:
    def __init__(self, images: Path, annotations: Path):
        self.root = images
        self.annotations = annotations
        self.document = json.loads(annotations.read_text("utf-8"))
        records = self.document["images"]
        self.records = {record["id"]: record for record in records}
        if len(self.records) != len(records) or not records:
            raise ValueError("COCO images must have unique IDs and must not be empty")
        self.categories = self.document["categories"]

    def validate_ids(self, ids: list[int]) -> None:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Image IDs must be a non-empty list without duplicates")
        for image_id in ids:
            if image_id not in self.records:
                raise ValueError(f"Unknown COCO image ID: {image_id}")
            path = self.root / self.records[image_id]["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)

    def read(self, image_id: int) -> np.ndarray:
        record = self.records[image_id]
        with Image.open(self.root / record["file_name"]) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Image dimensions disagree with annotation for ID {image_id}")
        return rgb


@dataclass(frozen=True)
class Geometry:
    original_hw: tuple[int, int]
    content_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    pad_xy: tuple[int, int]

    def restore(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes.clone()
        height, width = self.original_hw
        content_h, content_w = self.content_hw
        left, top = self.pad_xy
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * (width / content_w)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * (height / content_h)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        return boxes


def prepare_image(rgb: np.ndarray, size: int, mode: str) -> tuple[np.ndarray, Geometry]:
    """Run inside every measured call, including the common letterbox cost."""
    height, width = rgb.shape[:2]
    if mode == "native":
        return rgb, Geometry((height, width), (height, width), (height, width), (0, 0))
    if mode != "square":
        raise ValueError(f"Unknown geometry: {mode}")
    ratio = min(size / height, size / width)
    content_w, content_h = max(1, round(width * ratio)), max(1, round(height * ratio))
    resized = np.asarray(
        Image.fromarray(rgb).resize((content_w, content_h), Image.Resampling.BILINEAR)
    )
    left, top = (size - content_w) // 2, (size - content_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + content_h, left : left + content_w] = resized
    return canvas, Geometry((height, width), (content_h, content_w), (size, size), (left, top))


@dataclass
class Prediction:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def restored(self, geometry: Geometry) -> Prediction:
        # Copy results back to CPU before returning; this transfer is timed for every model.
        boxes = geometry.restore(self.boxes.detach().to(device="cpu", dtype=torch.float32))
        scores = self.scores.detach().to(device="cpu", dtype=torch.float32)
        labels = self.labels.detach().to(device="cpu", dtype=torch.int64)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid &= torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        return Prediction(boxes[valid], scores[valid], labels[valid])

    def coco_rows(self, image_id: int, category_ids: set[int]) -> list[dict]:
        rows = []
        for box, score, label in zip(
            self.boxes.tolist(), self.scores.tolist(), self.labels.tolist()
        ):
            if label not in category_ids:
                raise ValueError(f"Prediction has unknown original COCO category ID {label}")
            x1, y1, x2, y2 = box
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
        return rows
