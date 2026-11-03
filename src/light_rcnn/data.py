"""COCO dataset loading, batching and multi-scale training helpers."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF


class CocoTensorDataset(CocoDetection):
    def __init__(self, root, annFile, horizontal_flip_prob: float = 0.0):
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0 and 1")
        self.horizontal_flip_prob = horizontal_flip_prob
        super().__init__(root, annFile)

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)
        # Keep the detector input contract at three channels for grayscale or
        # palette-mode files as well as ordinary RGB COCO images.
        image = image.convert("RGB")
        boxes, labels = [], []
        for annotation in annotations:
            # COCO crowd regions are ignored by COCOeval and should not be
            # treated as ordinary positive boxes by the detector losses.
            if annotation.get("iscrowd", 0):
                continue
            x, y, width, height = annotation["bbox"]
            if width > 0 and height > 0:
                boxes.append([x, y, x + width, y + height])
                # Torchvision Faster R-CNN uses label 0 for background and COCO's
                # original category IDs (1..90, with gaps) for foreground classes.
                labels.append(annotation["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([self.ids[index]]),
        }
        if self.horizontal_flip_prob and random.random() < self.horizontal_flip_prob:
            image_width = image.width
            image = TF.hflip(image)
            x1 = target["boxes"][:, 0].clone()
            x2 = target["boxes"][:, 2].clone()
            target["boxes"][:, 0] = image_width - x2
            target["boxes"][:, 2] = image_width - x1
        return TF.to_tensor(image), target


def detection_collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def build_coco_loader(
    image_dir: str,
    annotation_file: str,
    batch_size: int,
    workers: int,
    shuffle: bool,
    horizontal_flip_prob: float = 0.0,
):
    dataset = CocoTensorDataset(
        Path(image_dir), str(annotation_file), horizontal_flip_prob=horizontal_flip_prob
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=detection_collate)


def multiscale_min_sizes(min_sizes, max_size, image_size) -> tuple[int, ...]:
    """Validated short-side choices for GeneralizedRCNNTransform multi-scale training.

    In training mode the transform draws one of ``min_size`` per image (torch_choice) and caps the
    long side at ``max_size``; in eval mode it uses ``min_size[-1]`` and ``max_size``. The sizes
    are sorted so ``image_size`` comes last and ``max_size`` must equal it: validation, export and
    saved checkpoints keep the fixed canvas. With a 640 cap a 4:3 image already reaches 640x480
    at short side 480, so only sizes below the short side of the full-size image shrink it.
    """
    sizes = tuple(sorted(int(size) for size in min_sizes))
    if int(max_size) != int(image_size):
        raise ValueError(f"train.multiscale.max_size must equal image_size ({image_size})")
    if not sizes or sizes[0] < 32 or sizes[-1] != int(image_size):
        raise ValueError(
            f"train.multiscale.min_sizes must lie in [32, {image_size}] and include {image_size}"
        )
    return sizes


def enable_multiscale(transform, min_sizes, max_size):
    """Random per-image short side in the model transform (resizing in the dataset would be
    undone by the transform); eval mode keeps the transform's own size."""
    if int(max_size) != int(transform.max_size):
        raise ValueError(f"train.multiscale.max_size must equal image_size ({transform.max_size})")
    transform.min_size = multiscale_min_sizes(min_sizes, max_size, transform.min_size[-1])
    return transform
