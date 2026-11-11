"""Offline checks that Hugging Face rows convert to the CocoTensorDataset target contract."""

import pytest
import torch
from PIL import Image

from light_rcnn.datasets import ALIASES, HF_DATASETS, build_dataset, hflip, make_target

OBJECTS = {
    "bbox": [[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 0.0, 3.0]],
    "category_id": [1, 3, 18],
    "iscrowd": [0, 1, 0],
}


def test_target_matches_coco_tensor_dataset_contract():
    target = make_target(OBJECTS, 42)
    # Crowd/ignore regions and empty boxes are dropped, xywh becomes xyxy.
    assert torch.equal(target["boxes"], torch.tensor([[10.0, 20.0, 40.0, 60.0]]))
    assert torch.equal(target["labels"], torch.tensor([1]))
    assert target["labels"].dtype == torch.int64
    assert target["image_id"].item() == 42


def test_empty_image_keeps_box_shape():
    target = make_target({"bbox": [], "category_id": [], "iscrowd": []}, 7)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_hflip_mirrors_boxes():
    image = Image.new("RGB", (100, 50))
    _, target = hflip(image, make_target(OBJECTS, 1))
    assert torch.equal(target["boxes"], torch.tensor([[60.0, 20.0, 90.0, 60.0]]))


def test_registry_and_unknown_name():
    assert all(alias in HF_DATASETS for alias in ALIASES.values())
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("not_a_dataset")
