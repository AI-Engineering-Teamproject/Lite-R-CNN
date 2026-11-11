"""Hugging Face (Hyeon1222/COCO) detection splits with the same contract as ``CocoTensorDataset``.

Every split below is already in the COCO label space (original COCO ``category_id`` 1..90, matching
``num_classes: 91``), so a dataset built here drops into ``train.py``, ``evaluate_coco`` and the
evaluation scripts exactly where ``CocoTensorDataset`` is used:

* ``dataset[i]`` -> ``(FloatTensor[3, H, W] in 0..1, {"boxes", "labels", "image_id"})`` at the
  original image size (the model transform resizes to ``image_size``); ``image_id`` is
  ``tensor([id])`` so ``target["image_id"].item()`` works.
* ``iscrowd=1`` boxes (COCO crowds, VOC ``difficult``, VisDrone ignored regions and the
  van -> truck ignore boxes) are left out of ``target`` so the losses never see them as positives.
  ``dataset.coco`` keeps them, so pycocotools still ignores detections on those regions.
* ``dataset.ids[i]`` is the image id of ``dataset[i]`` and ``dataset.coco`` is the pycocotools
  ground truth of that split, as ``evaluate_coco`` expects.

Only the requested split's parquet files are downloaded (``load_dataset(repo, config, split=...)``
would fetch every split of the config, e.g. 19GB of COCO train for a val run).

    from light_rcnn.datasets import build_dataset
    val = build_dataset("voc2007_test")
    train = build_dataset("visdrone2019_train", horizontal_flip_prob=0.5)

Needs ``pip install -e ".[huggingface]"`` (``datasets`` + ``huggingface_hub``).
"""

from __future__ import annotations

import io
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from light_rcnn.data import CocoTensorDataset, detection_collate

HF_REPO = "Hyeon1222/COCO"

# name -> (parquet config, split, ground-truth json under annotations/ in the HF repo)
HF_DATASETS = {
    "coco2017_train": ("coco2017", "train", "instances_train2017.json"),
    "coco2017_val": ("coco2017", "validation", "instances_val2017.json"),
    "voc2007_train": ("voc2007", "train", "voc2007_train_coco.json"),
    "voc2007_val": ("voc2007", "validation", "voc2007_val_coco.json"),
    "voc2007_test": ("voc2007", "test", "voc2007_test_coco.json"),
    "voc2012_train": ("voc2012", "train", "voc2012_train_coco.json"),
    "voc2012_val": ("voc2012", "validation", "voc2012_val_coco.json"),
    "visdrone2019_train": ("visdrone2019", "train", "visdrone2019_train_coco.json"),
    "visdrone2019_val": ("visdrone2019", "validation", "visdrone2019_val_coco.json"),
    "visdrone2019_test": ("visdrone2019", "test", "visdrone2019_test_dev_coco.json"),
}
ALIASES = {"coco_val2017": "coco2017_val", "visdrone2019_test_dev": "visdrone2019_test"}


def make_target(objects, image_id: int):
    """``objects`` is a parquet row's ``objects`` struct (COCO ``[x, y, w, h]`` boxes)."""
    boxes, labels = [], []
    for (x, y, width, height), category_id, crowd in zip(
        objects["bbox"], objects["category_id"], objects["iscrowd"]
    ):
        if crowd or width <= 0 or height <= 0:
            continue
        boxes.append([x, y, x + width, y + height])
        labels.append(int(category_id))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([int(image_id)]),
    }


def hflip(image: Image.Image, target: dict):
    width = image.width
    image = TF.hflip(image)
    x1 = target["boxes"][:, 0].clone()
    x2 = target["boxes"][:, 2].clone()
    target["boxes"][:, 0] = width - x2
    target["boxes"][:, 2] = width - x1
    return image, target


class HFDetectionDataset(Dataset):
    """One split of a parquet config in the Hugging Face dataset repo."""

    def __init__(
        self,
        config: str,
        split: str,
        annotation: str,
        horizontal_flip_prob: float = 0.0,
        repo_id: str = HF_REPO,
        cache_dir: str | None = None,
    ):
        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be between 0 and 1")
        from datasets import Image as HFImage
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from pycocotools.coco import COCO

        rows = load_dataset(
            repo_id,
            data_files={split: f"data/{config}/{split}-*.parquet"},
            split=split,
            cache_dir=cache_dir,
        )
        # Decode in __getitem__ so RGB conversion matches CocoTensorDataset.
        self.rows = rows.cast_column("image", HFImage(decode=False))
        # Same order as CocoDetection (sorted image ids), so ``limit``/``subset_seed`` subsets
        # in evaluate_coco pick the same images as the local CocoTensorDataset.
        row_ids = [int(image_id) for image_id in self.rows["image_id"]]
        self.rows_order = sorted(range(len(row_ids)), key=row_ids.__getitem__)
        self.ids = [row_ids[row] for row in self.rows_order]
        self.annotation_file = hf_hub_download(
            repo_id, f"annotations/{annotation}", repo_type="dataset", cache_dir=cache_dir
        )
        self.coco = COCO(self.annotation_file)
        missing = set(self.ids) - set(self.coco.imgs)
        if missing:
            raise ValueError(f"{len(missing)} images of {config}/{split} are not in {annotation}")
        self.horizontal_flip_prob = horizontal_flip_prob

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[self.rows_order[index]]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        target = make_target(row["objects"], row["image_id"])
        if self.horizontal_flip_prob and random.random() < self.horizontal_flip_prob:
            image, target = hflip(image, target)
        return TF.to_tensor(image), target


def build_dataset(
    name: str | None = None,
    images: str | None = None,
    annotations: str | None = None,
    horizontal_flip_prob: float = 0.0,
    **kwargs,
):
    """A Hugging Face split by name, or any local COCO-format folder (``images`` + ``annotations``,
    e.g. the multi_dataset_eval outputs for Open Images, BDD100K or COCO-O)."""
    if images is not None and annotations is not None:
        return CocoTensorDataset(images, annotations, horizontal_flip_prob=horizontal_flip_prob)
    key = ALIASES.get(name, name)
    if key not in HF_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}; choose one of {sorted(HF_DATASETS)} "
            "or pass images= and annotations= for a local COCO-format dataset"
        )
    config, split, annotation = HF_DATASETS[key]
    return HFDetectionDataset(
        config, split, annotation, horizontal_flip_prob=horizontal_flip_prob, **kwargs
    )


def build_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate,
    )
