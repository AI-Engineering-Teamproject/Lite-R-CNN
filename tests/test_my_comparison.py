"""Regression checks for fairness: sparse IDs, crowd handling, timing scope and real adapters."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)

from light_rcnn.comparison import cli
from light_rcnn.comparison.adapters import TorchvisionAdapter, YoloAdapter, class_mapping
from light_rcnn.comparison.measurement import benchmark, official_metrics
from light_rcnn.comparison.protocol import CocoImages, Prediction, prepare_image, select_ids
from light_rcnn.model.detector import LightRCNN, light_rcnn_from_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/lite_rcnn_comparison_test_ultralytics")


@pytest.fixture
def coco(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records, annotations = [], []
    for image_id, (width, height) in zip((101, 503), ((96, 64), (64, 96))):
        filename = f"{image_id}.png"
        Image.new("RGB", (width, height), (image_id % 256, 40, 90)).save(images / filename)
        records.append({"id": image_id, "width": width, "height": height, "file_name": filename})
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1 if image_id == 101 else 3,
                "bbox": [4, 6, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        )
    annotations.append(
        {
            "id": 3,
            "image_id": 101,
            "category_id": 1,
            "bbox": [40, 0, 40, 40],
            "area": 1600,
            "iscrowd": 1,
        }
    )
    path = tmp_path / "instances.json"
    path.write_text(
        json.dumps(
            {
                "images": records,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
            }
        )
    )
    return CocoImages(images, path)


def test_mapping_and_original_coordinate_roundtrip():
    mapping = class_mapping(
        {0: "person", 1: "car"}, [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}]
    )
    assert mapping == {0: 1, 1: 3}
    with pytest.raises(ValueError, match="taxonomy"):
        class_mapping({0: "truck"}, [{"id": 1, "name": "person"}])
    # Non-square dimensions exercise rounding and separate X/Y scales.
    rgb = np.zeros((37, 91, 3), dtype=np.uint8)
    padded, geometry = prepare_image(rgb, 64, "square")
    assert padded.shape == (64, 64, 3)
    original = torch.tensor([[5.0, 4.0, 80.0, 30.0]])
    transformed = original.clone()
    transformed[:, [0, 2]] *= geometry.content_hw[1] / 91
    transformed[:, [1, 3]] *= geometry.content_hw[0] / 37
    transformed[:, [0, 2]] += geometry.pad_xy[0]
    transformed[:, [1, 3]] += geometry.pad_xy[1]
    torch.testing.assert_close(geometry.restore(transformed), original)
    prediction = Prediction(transformed, torch.tensor([0.9]), torch.tensor([3])).restored(geometry)
    row = prediction.coco_rows(503, {1, 3})[0]
    assert row["category_id"] == 3
    assert row["image_id"] == 503
    assert row["bbox"] == pytest.approx([5, 4, 75, 26])


def test_official_coco_preserves_crowds_sparse_ids_and_empty_predictions(coco):
    before = coco.annotations.read_bytes()
    rows = [
        {"image_id": 101, "category_id": 1, "bbox": [4, 6, 20, 20], "score": 0.9},
        {"image_id": 503, "category_id": 3, "bbox": [4, 6, 20, 20], "score": 0.9},
        # Higher-scored detection inside a crowd must be ignored, not a false positive.
        {"image_id": 101, "category_id": 1, "bbox": [45, 5, 10, 10], "score": 0.99},
    ]
    metrics = official_metrics(coco.annotations, [101, 503], rows)
    assert metrics["AP_50_95"] == pytest.approx(1)
    assert metrics["AR_100"] == pytest.approx(1)
    assert metrics["AP_large"] is None
    assert coco.annotations.read_bytes() == before
    assert official_metrics(coco.annotations, [101, 503], [])["AP_50_95"] == 0


def test_yolo_nonempty_output_maps_class_index_and_rgb_input():
    class Detector:
        def __init__(self):
            self.names = {0: "person", 1: "car"}
            self.model = torch.nn.Linear(1, 1)

        def predict(self, source, **options):
            assert source[0, 0].tolist() == [30, 20, 10]  # RGB -> BGR inside timing boundary
            assert options["half"] is False
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[4.0, 6.0, 24.0, 26.0]]),
                        conf=torch.tensor([0.9]),
                        cls=torch.tensor([1.0]),
                    )
                )
            ]

    adapter = YoloAdapter(
        Detector(),
        categories=[{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        device="cpu",
        size=64,
        geometry="native",
        score=0.001,
        nms=0.5,
        max_detections=100,
        metadata={},
    )
    prediction = adapter.predict(np.full((64, 64, 3), [10, 20, 30], dtype=np.uint8))
    assert prediction.labels.tolist() == [3]
    assert prediction.coco_rows(503, {1, 3})[0]["bbox"] == [4, 6, 20, 20]


def test_benchmark_times_cpu_contract_and_covers_all_images(monkeypatch):
    from light_rcnn.comparison import measurement

    events = []

    class Adapter:
        def inspect_geometry(self, rgb):
            events.append("inspect")
            return {"model_input_nchw": [1, 3, 64, 64]}

        def predict(self, rgb):
            assert rgb.dtype == np.uint8
            events.append("predict")
            return Prediction(torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long))

    def timer():
        events.append("timer")
        return len(events) * 1_000_000

    monkeypatch.setattr(measurement.time, "perf_counter_ns", timer)
    monkeypatch.setattr(measurement, "synchronize", lambda device: events.append("sync"))
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    result = benchmark(Adapter(), [(101, rgb), (503, rgb)], warmup=1, iterations=4, device="cpu")
    assert events[:4] == ["inspect", "inspect", "predict", "sync"]
    assert events[4:] == ["sync", "timer", "predict", "sync", "timer"] * 4
    assert [sample["image_id"] for sample in result["samples"]] == [101, 503, 101, 503]
    assert result["p50_ms"] > 0
    with pytest.raises(ValueError, match="cover every"):
        benchmark(Adapter(), [(101, rgb), (503, rgb)], warmup=1, iterations=1, device="cpu")


@pytest.mark.parametrize("kind", ("lite", "resnet", "mobilenet", "yolo"))
@pytest.mark.parametrize("geometry", ("square", "native"))
def test_real_model_cpu_adapters_without_weight_downloads(kind, geometry):
    rgb = np.zeros((48, 80, 3), dtype=np.uint8)
    if kind == "yolo":
        from ultralytics import YOLO

        # Architecture-only fixture; never exposed as a trained comparison in the CLI.
        model = YOLO("yolo26n.yaml", verbose=False)
        categories = [
            {"id": index, "name": name}
            for index, name in enumerate(FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"])
            if index and name != "N/A"
        ]
        model.model.names = {index: category["name"] for index, category in enumerate(categories)}
        adapter = YoloAdapter(
            model,
            categories=categories,
            device="cpu",
            size=64,
            geometry=geometry,
            score=0.25,
            nms=0.5,
            max_detections=100,
            metadata={},
        )
    else:
        if kind == "lite":
            model = LightRCNN(
                pretrained_backbone=False,
                min_size=64,
                max_size=64,
                bifpn_channels=16,
                roi_channels=8,
            )
        else:
            builder = (
                fasterrcnn_resnet50_fpn if kind == "resnet" else fasterrcnn_mobilenet_v3_large_fpn
            )
            model = builder(weights=None, weights_backbone=None, min_size=64, max_size=64)
        model.rpn._pre_nms_top_n["testing"] = 20
        model.rpn._post_nms_top_n["testing"] = 10
        adapter = TorchvisionAdapter(model, device="cpu", size=64, geometry=geometry, metadata={})
    threads = torch.get_num_threads()
    prediction = adapter.predict(rgb)
    assert torch.get_num_threads() == threads
    assert prediction.boxes.device.type == prediction.labels.device.type == "cpu"
    assert prediction.boxes.shape[1] == 4
    assert torch.isfinite(prediction.boxes).all()
    observed = adapter.inspect_geometry(rgb)
    shape = observed["model_input_nchw"]
    assert shape[:2] == [1, 3]
    assert observed["model_device"] == "cpu"
    assert observed["model_dtype"] == "torch.float32"
    if geometry == "square":
        assert shape[2:] == [64, 64]
    if kind == "yolo":
        # Must release setup AutoBackend between parameterized tests.
        assert adapter.mapping[0] == 1 and adapter.mapping[2] == 3


class OracleAdapter:
    def __init__(self, name):
        self.metadata = {"name": name, "parameters": 42}

    def predict(self, rgb):
        label = 1 if int(rgb[0, 0, 0]) == 101 else 3
        return Prediction(
            torch.tensor([[4.0, 6.0, 24.0, 26.0]]), torch.tensor([0.9]), torch.tensor([label])
        )

    def inspect_geometry(self, rgb):
        _, geometry = prepare_image(rgb, 64, "square")
        return {"shared_geometry": asdict(geometry), "model_input_nchw": [1, 3, 64, 64]}


def cli_args(coco, tmp_path):
    config = yaml.safe_load((ROOT / "configs/my_comparison.yaml").read_text())
    config.update(device="cpu", cpu_threads=2, image_size=64)
    config["data"] = {"images": str(coco.root), "annotations": str(coco.annotations)}
    config["benchmark"] = {"warmup": 1, "iterations": 2, "image_count": 2}
    # Existing temporary files pass preflight; the oracle factory does not load weights.
    config["models"]["lite_rcnn"]["weights"] = str(coco.annotations)
    config["models"]["yolo26"]["weights"] = str(coco.annotations)
    path = tmp_path / "comparison.yaml"
    path.write_text(yaml.safe_dump(config))
    return cli.parser().parse_args(
        ["--config", str(path), "--output-dir", str(tmp_path / "result")]
    )


def test_four_model_runner_reports_and_manifest_replay(coco, tmp_path, monkeypatch):
    seen = []

    def factory(name, config, categories, root):
        seen.append(name)
        return OracleAdapter(name)

    monkeypatch.setattr(cli, "build_adapter", factory)
    args = cli_args(coco, tmp_path)
    output = cli.run(args, ROOT)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["evaluation_image_ids"]) == {101, 503}
    assert manifest["evaluation_scope"] == "full"
    summary = json.loads((output / "comparison.json").read_text())
    assert len(summary["table"]) == 4
    assert all(row["AP"] == pytest.approx(100) for row in summary["table"])
    assert all(row["p50 (ms)"] > 0 for row in summary["table"])
    assert (output / "comparison.csv").is_file() and (output / "REPORT.md").is_file()
    for name in seen:
        predictions = json.loads((output / f"{name}_predictions.json").read_text())
        assert {row["category_id"] for row in predictions} == {1, 3}
        assert {row["image_id"] for row in predictions} == {101, 503}
    args.image_ids = str(output / "manifest.json")
    args.output_dir = str(tmp_path / "replay")
    replay = json.loads((cli.run(args, ROOT) / "manifest.json").read_text())
    assert replay["benchmark_image_ids"] == manifest["benchmark_image_ids"]
    assert replay["evaluation_image_ids"] == manifest["evaluation_image_ids"]
    assert select_ids(list(range(100)), 10, 42) == select_ids(list(range(100)), 10, 42)


def test_failure_is_visible_without_fabricated_zero_metrics(coco, tmp_path, monkeypatch):
    def factory(name, config, categories, root):
        if name == "faster_rcnn":
            raise RuntimeError("deliberate fixture failure")
        return OracleAdapter(name)

    monkeypatch.setattr(cli, "build_adapter", factory)
    args = cli_args(coco, tmp_path)
    with pytest.raises(RuntimeError, match="fixture failure"):
        cli.run(args, ROOT)
    output = Path(args.output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_model"] == "faster_rcnn"
    rows = json.loads((output / "comparison.json").read_text())["table"]
    assert rows[0]["Status"] == "complete"
    assert rows[1]["Status"] == "failed"
    assert rows[1]["AP"] is None and rows[1]["Parameters (M)"] is None


def test_preflight_rejects_missing_checkpoint_and_invalid_device(coco, tmp_path):
    args = cli_args(coco, tmp_path)
    args.light_weights = str(tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError, match="local trained weights"):
        cli.load_config(args, ROOT)
    args.device = "mps"
    with pytest.raises(ValueError, match="Only CPU and CUDA"):
        cli.load_config(args, ROOT)


def test_actual_lite_checkpoint_cli_to_report(coco, tmp_path):
    """Exercise the real checkpoint loader and entry point, without an oracle adapter."""
    args = cli_args(coco, tmp_path)
    training_config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text())
    training_config.update(image_size=64)
    training_config["model"].update(
        bifpn_channels=16, roi_channels=8, rpn_pre_nms_topk_test=20, rpn_post_nms_topk_test=10
    )
    model = light_rcnn_from_config(training_config)
    checkpoint = tmp_path / "fixture.pth"
    torch.save({"config": training_config, "model": model.state_dict(), "epoch": 0}, checkpoint)
    config = yaml.safe_load(Path(args.config).read_text())
    config["score_threshold"] = 0.999
    Path(args.config).write_text(yaml.safe_dump(config))
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_my_comparison.py"),
            "--config",
            args.config,
            "--models",
            "lite_rcnn",
            "--light-weights",
            str(checkpoint),
            "--output-dir",
            args.output_dir,
        ],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((Path(args.output_dir) / "lite_rcnn.json").read_text())
    assert report["status"] == "complete"
    assert report["metadata"]["checkpoint_epoch"] == 0
    assert len(report["metadata"]["weights_sha256"]) == 64
    assert report["metadata"]["rpn_post_nms_test"] == 10
    assert report["accuracy"]["AP_50_95"] == 0
    assert report["latency"]["image_count"] == 2
    assert all(row["model_input_nchw"] == [1, 3, 64, 64] for row in report["latency"]["geometry"])
