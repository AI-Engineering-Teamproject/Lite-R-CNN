import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
import torch
from PIL import Image

from light_rcnn.coco_eval import STAT_NAMES
from light_rcnn.evaluation import evaluation_subset, subset_record
from light_rcnn.reporting import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_coco.py"
IMAGE_IDS = [9, 2, 7, 4, 5]  # deliberately unsorted; CocoDetection sorts them
BOX = [4.0, 6.0, 40.0, 42.0]


@pytest.fixture
def coco(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records, annotations = [], []
    for image_id in IMAGE_IDS:
        name = f"{image_id:012d}.jpg"
        Image.new("RGB", (80, 64), (image_id * 20, 80, 120)).save(images / name)
        records.append({"id": image_id, "file_name": name, "width": 80, "height": 64})
        annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                            "bbox": [4, 6, 36, 36], "area": 36 * 36, "iscrowd": 0})
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"images": records, "annotations": annotations,
                                "categories": [{"id": 1, "name": "thing"}]}))
    return images, path


class StubDetector(torch.nn.Module):
    """Predicts the ground-truth box everywhere and records the limits it was given."""

    def __init__(self):
        super().__init__()
        self.roi_heads = SimpleNamespace(score_thresh=None, detections_per_img=None)
        self.batches = []

    def forward(self, images):
        self.batches.append(len(images))
        return [{"boxes": torch.tensor([BOX]), "scores": torch.tensor([0.9]),
                 "labels": torch.tensor([1])} for _ in images]


@pytest.fixture
def evaluate(monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluate_coco", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = {"model": StubDetector()}

    def fake_load(kind, config, weights, image_size):
        built.update(kind=kind, config=config, weights=weights, image_size=image_size)
        return built["model"]

    monkeypatch.setattr(module, "load_comparison_model", fake_load)
    return module, built


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["evaluate_coco.py", *map(str, argv)])
    module.main()


def test_evaluation_subset_is_the_seeded_training_rule():
    ids = list(range(100, 120))
    expected = torch.randperm(20, generator=torch.Generator().manual_seed(42))[:5].tolist()
    assert evaluation_subset(ids, 5, 42) == expected
    assert evaluation_subset(ids, 5, 42) != evaluation_subset(ids, 5, 43)
    assert evaluation_subset(ids, 0, 42) == evaluation_subset(ids, 20, 42) == list(range(20))


def test_subset_record_identifies_the_scored_images():
    partial = subset_record([7, 2], 5, 42)
    assert partial["is_partial"] and partial["image_ids"] == [2, 7]
    assert partial["subset_selection"] == "seeded_random" and partial["subset_seed"] == 42
    full = subset_record([2, 4, 5, 7, 9], 5, 42)
    assert not full["is_partial"] and full["subset_seed"] is None and "image_ids" not in full
    assert subset_record([2, 7], 5, 0)["image_ids_sha256"] == partial["image_ids_sha256"]


def test_schema_v2_report_with_overrides(evaluate, monkeypatch, coco, tmp_path):
    module, built = evaluate
    images, annotations = coco
    output = tmp_path / "report.json"
    run(module, monkeypatch, "--config", ROOT / "configs/coco.yaml", "--model", "faster_rcnn",
        "--device", "cpu", "--images", images, "--annotations", annotations, "--limit", 3,
        "--batch-size", 2, "--imgsz", 320, "--score-thresh", 0.2, "--max-dets", 50,
        "--output", output)
    report = json.loads(output.read_text())
    accuracy = report["accuracy"]
    assert (report["schema_version"], report["kind"], report["model"]) == (2, "accuracy",
                                                                           "faster_rcnn")
    assert report["weights"] == "torchvision_coco_v1" and report["weights_sha256"] is None
    assert built["image_size"] == report["image_size"] == 320
    assert built["model"].roi_heads.score_thresh == accuracy["score_threshold"] == 0.2
    assert built["model"].roi_heads.detections_per_img == accuracy["max_detections"] == 50
    assert report["batch_size"] == 2 and built["model"].batches == [2, 1]
    sorted_ids = sorted(IMAGE_IDS)
    assert accuracy["image_ids"] == sorted(sorted_ids[i] for i in evaluation_subset(sorted_ids, 3, 42))
    assert accuracy["evaluator"] == "pycocotools" and accuracy["subset_seed"] == 42
    assert (accuracy["images_evaluated"], accuracy["images_total"]) == (3, 5)
    assert accuracy["is_partial"] and report["images_evaluated"] == 3
    assert accuracy["dataset"]["annotations_sha256"] == file_sha256(annotations)
    assert all(name in accuracy for name in STAT_NAMES)
    assert accuracy["AP_50_95"] == pytest.approx(1.0)
    assert accuracy["per_category_ap"] == {"thing": pytest.approx(1.0)}
    assert report["environment"]["device"] == "cpu"


def test_config_defaults_to_checkpoint_config(evaluate, monkeypatch, coco, tmp_path):
    module, built = evaluate
    images, annotations = coco
    config = {"seed": 7, "image_size": 256, "device": "cpu", "num_classes": 2,
              "data": {"val_images": str(images), "val_annotations": str(annotations)},
              "train": {"val_batch_size": 3}}
    weights = tmp_path / "best.pth"
    torch.save({"config": config, "model": {}}, weights)
    output = tmp_path / "report.json"
    run(module, monkeypatch, "--model", "light_rcnn", "--weights", weights, "--limit", 2,
        "--output", output)
    report = json.loads(output.read_text())
    accuracy = report["accuracy"]
    assert built["config"] == config and built["weights"] == str(weights)
    assert "checkpoint config" in report["config"]
    # Accuracy is scored at batch 1, not train.val_batch_size: padding changes R-CNN detections.
    assert report["image_size"] == 256 and report["batch_size"] == 1
    assert set(built["model"].batches) == {1}
    assert report["weights_sha256"] == file_sha256(weights)
    # The subset seed is the one evaluate_yolo26.py reads from configs/coco.yaml, not the
    # checkpoint's training seed, so every model scores the same --limit images.
    sorted_ids = sorted(IMAGE_IDS)
    assert accuracy["subset_seed"] == 42
    assert accuracy["image_ids"] == sorted(sorted_ids[i] for i in evaluation_subset(sorted_ids, 2, 42))
    # Checkpoints saved before the comparison settings existed fall back to the shared defaults.
    assert (accuracy["score_threshold"], accuracy["max_detections"]) == (0.001, 100)


def test_batch_size_defaults_to_comparison_not_validation(evaluate, monkeypatch, coco, tmp_path):
    module, built = evaluate
    images, annotations = coco
    output = tmp_path / "report.json"
    # configs/coco.yaml trains with val_batch_size 4; the comparison protocol is batch 1.
    run(module, monkeypatch, "--config", ROOT / "configs/coco.yaml", "--model", "faster_rcnn",
        "--device", "cpu", "--images", images, "--annotations", annotations, "--output", output)
    assert json.loads(output.read_text())["batch_size"] == 1
    assert built["model"].batches == [1] * len(IMAGE_IDS)


def test_light_rcnn_requires_weights(evaluate, monkeypatch):
    module, _ = evaluate
    with pytest.raises(SystemExit) as exc:
        run(module, monkeypatch, "--model", "light_rcnn")
    assert exc.value.code == 2


def test_rejects_non_positive_batch_size(evaluate, monkeypatch):
    module, _ = evaluate
    with pytest.raises(SystemExit) as exc:
        run(module, monkeypatch, "--config", ROOT / "configs/coco.yaml", "--model", "faster_rcnn",
            "--batch-size", 0)
    assert exc.value.code == 2


def test_blocked_weight_download_exits_with_hint(evaluate, monkeypatch, coco):
    module, _ = evaluate

    def offline(*args):
        raise URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(module, "load_comparison_model", offline)
    with pytest.raises(SystemExit, match="--weights"):
        run(module, monkeypatch, "--config", ROOT / "configs/coco.yaml", "--model", "faster_rcnn",
            "--device", "cpu")
