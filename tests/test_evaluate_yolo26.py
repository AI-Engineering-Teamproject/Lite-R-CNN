import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluate_coco

pytest.importorskip("ultralytics")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_yolo26.py"
IMAGE_IDS = [9, 2, 7, 4, 5]  # deliberately unsorted; CocoDetection sorts them


@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("evaluate_yolo26", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def coco(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records, annotations = [], []
    for image_id in IMAGE_IDS:
        name = f"{image_id:012d}.jpg"
        Image.new("RGB", (80, 64), (image_id * 20, 80, 120)).save(images / name)
        records.append({"id": image_id, "file_name": name, "width": 80, "height": 64})
        annotations.append({"id": image_id, "image_id": image_id, "category_id": 3,
                            "bbox": [4, 6, 36, 36], "area": 36 * 36, "iscrowd": 0})
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"images": records, "annotations": annotations,
                                "categories": [{"id": 1, "name": "person"},
                                               {"id": 3, "name": "car"}]}))
    return images, path


@pytest.fixture
def keep_threads():
    threads = torch.get_num_threads()
    yield threads
    torch.set_num_threads(threads)


def predictor(module, end2end=False):
    instance = module.MultiLabelPredictor.__new__(module.MultiLabelPredictor)
    instance.args = SimpleNamespace(conf=0.001, iou=0.7, classes=None, agnostic_nms=False,
                                    max_det=100)
    instance.model = SimpleNamespace(end2end=end2end, names={0: "a", 1: "b", 2: "c"})
    instance.batch = [["image.jpg"]]
    return instance


def test_multi_label_keeps_every_class_above_threshold(evaluate):
    # One xywh box scoring 0.9 for class 0 and 0.4 for class 1; class 2 stays below conf.
    preds = torch.tensor([[[32.0], [32.0], [16.0], [16.0], [0.9], [0.4], [0.0005]]])
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = predictor(evaluate).postprocess(preds, torch.zeros(1, 3, 64, 64), [image])[0].boxes
    assert sorted(boxes.cls.tolist()) == [0.0, 1.0]
    assert sorted(boxes.conf.tolist()) == pytest.approx([0.4, 0.9])
    assert boxes.xyxy.tolist() == [[24.0, 24.0, 40.0, 40.0]] * 2


def test_category_map_matches_names_then_coco91(evaluate, coco):
    dataset = CocoTensorDataset(*coco)
    mapping = evaluate.category_map({0: "person", 1: "bicycle", 2: "car"}, dataset.coco)
    assert mapping == {0: 1, 1: 2, 2: 3}  # bicycle is absent here, so the 80->91 table decides


def test_select_images_matches_what_evaluate_coco_scores(evaluate, coco):
    class Empty(torch.nn.Module):
        def forward(self, images):
            return [{"boxes": torch.empty(0, 4), "scores": torch.empty(0),
                     "labels": torch.empty(0, dtype=torch.long)} for _ in images]

    dataset = CocoTensorDataset(*coco)
    for limit, seed in ((3, 42), (2, 7), (0, 42), (9, 42)):
        scored = evaluate_coco(Empty(), dataset, "cpu", limit, subset_seed=seed, detailed=True)
        assert evaluate.select_images(dataset, limit, seed) == scored["image_ids"]


@pytest.fixture
def predict_calls(evaluate, monkeypatch, keep_threads):
    """Swap ultralytics.YOLO for a fake: one 'car' box per image, plus select_device's thread drop."""
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            self.names = {0: "person", 1: "car"}
            self.ckpt_path = None
            self.model = SimpleNamespace(model=[SimpleNamespace(end2end=True)])
            self.predictor = None

        def predict(self, source, predictor, **kwargs):
            calls.append({"source": source, "predictor": predictor, **kwargs})
            if self.predictor is None:
                torch.set_num_threads(1)
                self.predictor = SimpleNamespace(model=SimpleNamespace(end2end=False, fp16=False))
            boxes = SimpleNamespace(xyxy=torch.tensor([[4.0, 6.0, 40.0, 42.0]]),
                                    cls=torch.tensor([1.0]), conf=torch.tensor([0.8]))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr(evaluate, "YOLO", FakeYOLO)
    return calls


def test_report_scores_the_evaluate_coco_subset(evaluate, coco, tmp_path, monkeypatch,
                                                predict_calls, keep_threads):
    images, annotations = coco
    output = tmp_path / "yolo.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_yolo26.py", "--config", str(ROOT / "configs/coco.yaml"), "--weights", "fake.pt",
        "--device", "cpu", "--images", str(images), "--annotations", str(annotations),
        "--limit", "3", "--max-dets", "50", "--output", str(output)])
    evaluate.main()
    report = json.loads(output.read_text())
    accuracy = report["accuracy"]
    dataset = CocoTensorDataset(images, str(annotations))
    assert accuracy["image_ids"] == sorted(evaluate.select_images(dataset, 3, 42))
    assert sorted(Path(call["source"]).name for call in predict_calls) == [
        f"{image_id:012d}.jpg" for image_id in accuracy["image_ids"]]
    assert all(call["predictor"] is evaluate.MultiLabelPredictor for call in predict_calls)
    assert {(call["conf"], call["max_det"]) for call in predict_calls} == {(0.001, 50)}
    assert torch.get_num_threads() == keep_threads
    assert (report["schema_version"], report["kind"], report["model"]) == (2, "accuracy",
                                                                           "yolo26")
    assert accuracy["evaluator"] == "pycocotools" and accuracy["inference_mode"] == "NMS"
    assert (accuracy["images_evaluated"], accuracy["images_total"]) == (3, 5)
    assert accuracy["is_partial"] and accuracy["subset_seed"] == 42
    assert (accuracy["score_threshold"], accuracy["max_detections"]) == (0.001, 50)
    assert accuracy["AP_50_95"] == pytest.approx(1.0)
    assert accuracy["per_category_ap"] == {"car": pytest.approx(1.0)}
    assert report["images_evaluated"] == 3 and report["precision"] == "fp32"


def test_nms_free_flag_reaches_predict(evaluate, coco, tmp_path, monkeypatch, predict_calls):
    images, annotations = coco
    monkeypatch.setattr(sys, "argv", [
        "evaluate_yolo26.py", "--config", str(ROOT / "configs/coco.yaml"), "--weights", "fake.pt",
        "--device", "cpu", "--images", str(images), "--annotations", str(annotations),
        "--limit", "1", "--nms-free", "--output", str(tmp_path / "yolo.json")])
    evaluate.main()
    call = predict_calls[0]
    assert call.get("nms") is False or call.get("end2end") is True
