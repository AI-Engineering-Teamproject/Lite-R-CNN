"""COCO dataset loading and batching helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF


class CocoTensorDataset(CocoDetection):
    def __init__(self, root, annFile):
        super().__init__(root, annFile)

    def __getitem__(self, index: int):
        image, annotations = super().__getitem__(index)
        boxes, labels = [], []
        for annotation in annotations:
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
        return TF.to_tensor(image), target


def detection_collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def build_coco_loader(image_dir: str, annotation_file: str, batch_size: int, workers: int, shuffle: bool):
    dataset = CocoTensorDataset(Path(image_dir), str(annotation_file))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=detection_collate)
