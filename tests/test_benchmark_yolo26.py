import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from light_rcnn.model.deploy import use_channels_last
from light_rcnn.reporting import COLD_START_FIRST_PREDICT, SCOPE_HOST_TO_DETECTIONS

cv2 = pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_yolo26.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_yolo26", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conv_bn():
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.BatchNorm2d(8),
                               torch.nn.Conv2d(8, 8, 1)).eval()


def test_graph_layout_reports_contiguous_unfused(bench):
    assert bench.graph_layout(conv_bn()) == {"memory_format": "contiguous",
                                             "conv_bn_fused": False}


def test_graph_layout_detects_channels_last_and_fusion(bench):
    net = conv_bn()
    fused = torch.nn.Sequential(torch.nn.utils.fusion.fuse_conv_bn_eval(net[0], net[1]), net[2])
    fused.to(memory_format=torch.channels_last)
    assert bench.graph_layout(fused) == {"memory_format": "channels_last", "conv_bn_fused": True}


def test_graph_layout_ignores_ambiguous_1x1_kernels(bench):
    net = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1)).to(memory_format=torch.channels_last)
    assert bench.graph_layout(net)["memory_format"] == "contiguous"
    assert bench.graph_layout(None) == {"memory_format": None, "conv_bn_fused": None}


def test_predictor_graph_prefers_backend_model(bench):
    net = conv_bn()
    backend = SimpleNamespace(backend=SimpleNamespace(model=net), model=None)
    assert bench.predictor_graph(SimpleNamespace(predictor=SimpleNamespace(model=backend))) is net
    fallback = SimpleNamespace(predictor=SimpleNamespace(model=SimpleNamespace(model=net)))
    assert bench.predictor_graph(fallback) is net
    assert bench.predictor_graph(SimpleNamespace(predictor=None)) is None


def test_inference_mode_prefers_predictor_backend(bench):
    head = SimpleNamespace(end2end=True)
    model = SimpleNamespace(model=SimpleNamespace(end2end=None, model=[head]),
                            predictor=SimpleNamespace(model=SimpleNamespace(end2end=False)))
    assert bench.inference_mode(model) == "NMS"
    model.predictor = None
    assert bench.inference_mode(model) == "end2end (NMS-free)"
    head.end2end = "yes"  # only a real bool is trusted
    assert bench.inference_mode(model) == "unknown"


def test_count_flops_returns_none_when_profiling_fails(bench):
    assert bench.count_flops(torch.nn.Identity(), 64) is None


def test_ultralytics_device_maps_torch_devices(bench):
    assert bench.ultralytics_device(torch.device("cpu")) == "cpu"
    assert bench.ultralytics_device(torch.device("cuda")) == "0"
    assert bench.ultralytics_device(torch.device("cuda:1")) == "1"
    with pytest.raises(SystemExit):
        bench.ultralytics_device(torch.device("meta"))


def test_load_images_accepts_file_or_folder(bench, tmp_path):
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    for name in ("b.jpg", "a.png", "c.png"):
        cv2.imwrite(str(tmp_path / name), pixels)
    (tmp_path / "notes.txt").write_text("skip me")
    paths, images = bench.load_images(tmp_path, count=2)
    assert [path.name for path in paths] == ["a.png", "b.jpg"]
    assert all(image.dtype == np.uint8 and image.shape == (4, 6, 3) for image in images)
    paths, images = bench.load_images(tmp_path / "c.png", count=5)
    assert [path.name for path in paths] == ["c.png"] and len(images) == 1
    for missing in (tmp_path / "notes.txt", tmp_path / "missing"):
        with pytest.raises(SystemExit, match="No benchmark images"):
            bench.load_images(missing, count=1)


def test_load_images_rejects_undecodable_files(bench, tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")
    with pytest.raises(SystemExit, match="Failed to decode"):
        bench.load_images(tmp_path, count=1)


def run_main(bench, monkeypatch, tmp_path, *extra):
    """Run the CLI on an untrained yolo26n.yaml graph (no download) at a tiny input size."""
    for index, height in enumerate((48, 64)):
        cv2.imwrite(str(tmp_path / f"img{index}.jpg"), np.full((height, 80, 3), 127, np.uint8))
    config = {"image_size": 64, "device": "cpu", "data": {"val_images": str(tmp_path)},
              "comparison": {"warmup": 1, "iterations": 2, "image_count": 4,
                             "latency_score_threshold": 0.25, "max_detections": 100}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_yolo26.py", "--config", str(config_path),
                                      "--weights", "yolo26n.yaml", "--output", str(output),
                                      *extra])
    bench.main()
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.mark.parametrize("extra", [("--max-dets", "0"), ("--imgsz", "0"), ("--warmup", "-1"),
                                   ("--device", "bogus")])
def test_main_rejects_invalid_arguments(bench, monkeypatch, tmp_path, capsys, extra):
    with pytest.raises(SystemExit) as exit_info:
        run_main(bench, monkeypatch, tmp_path, *extra)
    assert exit_info.value.code == 2
    assert extra[0] in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_main_keeps_threads_and_records_fair_settings(bench, monkeypatch, tmp_path):
    from ultralytics.utils import NUM_THREADS

    original = torch.get_num_threads()
    wanted = NUM_THREADS + 1  # differs from what select_device() sets on the first predict
    torch.set_num_threads(wanted)
    try:
        report = run_main(bench, monkeypatch, tmp_path)
        assert torch.get_num_threads() == wanted
    finally:
        torch.set_num_threads(original)
    assert report["schema_version"] == 2 and report["kind"] == "latency"
    assert report["model"] == "yolo26" and report["precision"] == "fp32"
    assert report["environment"]["torch_num_threads"] == wanted
    latency = report["latency"]
    assert latency["scope_id"] == SCOPE_HOST_TO_DETECTIONS
    assert latency["cold_start_scope"] == COLD_START_FIRST_PREDICT
    assert latency["cold_start_ms"] > 0 and latency["n"] == 2 and latency["image_count"] == 2
    assert latency["inference_mode"] == "NMS" and latency["nms_free_head_available"] is True
    assert latency["conv_bn_fused"] is True
    # Same auto rule as the R-CNN benchmarks when comparison.channels_last is unset.
    expected = "channels_last" if use_channels_last(torch.device("cpu")) else "contiguous"
    assert latency["memory_format"] == expected
    stats = report["model_stats"]
    assert stats["flops_input"] == [64, 64]
    assert stats["parameters_m"] < stats["checkpoint_parameters_m"]
    if stats["gflops"] is not None:
        assert stats["gflops"] < stats["checkpoint_gflops"]


def test_main_single_image_nms_free_contiguous(bench, monkeypatch, tmp_path):
    original = torch.get_num_threads()
    try:
        report = run_main(bench, monkeypatch, tmp_path, "--source", str(tmp_path / "img1.jpg"),
                          "--nms-free", "--no-channels-last")
    finally:
        torch.set_num_threads(original)
    latency = report["latency"]
    assert latency["inference_mode"] == "end2end (NMS-free)"
    assert latency["memory_format"] == "contiguous"
    assert latency["image_count"] == 1 and latency["images"][0].endswith("img1.jpg")
    assert report["weights"] == "yolo26n.yaml" and report["weights_sha256"] is None
