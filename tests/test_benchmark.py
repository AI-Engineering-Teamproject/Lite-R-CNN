import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from light_rcnn.bench import COMPARISON_DEFAULTS, load_images
from light_rcnn.model import build_detector
from light_rcnn.model.deploy import fuse_conv_bn, use_channels_last
from light_rcnn.model.detector import light_rcnn_from_config
from light_rcnn.reporting import SCHEMA_VERSION, SCOPE_HOST_TO_DETECTIONS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("benchmark_script", ROOT / "scripts/benchmark.py")
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def randomize_norms(model: nn.Module) -> None:
    generator = torch.Generator().manual_seed(0)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
            size = module.running_mean.shape
            module.running_mean.copy_(torch.randn(size, generator=generator) * 0.1)
            module.running_var.copy_(torch.rand(size, generator=generator) + 0.5)
            module.weight.data.copy_(torch.rand(size, generator=generator) + 0.5)
            module.bias.data.copy_(torch.randn(size, generator=generator) * 0.1)


def has_norm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)) for m in model.modules())


def write_image(path: Path, height: int = 96, width: int = 128) -> np.ndarray:
    array = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return array


@torch.no_grad()
def test_fuse_conv_bn_folds_batchnorm_and_frozen_batchnorm():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 1, bias=False), FrozenBatchNorm2d(8), nn.ReLU(),
    )
    randomize_norms(model)
    model.eval()
    x = torch.randn(2, 3, 16, 16)
    expected = model(x)

    assert fuse_conv_bn(model) == 2
    assert not has_norm(model)
    torch.testing.assert_close(model(x), expected, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_fuse_conv_bn_leaves_no_norm_in_light_rcnn_and_keeps_features():
    # Random heads give near-tied scores, so compare the fused stages rather than post-NMS boxes.
    torch.manual_seed(0)
    model = build_detector("light_rcnn", pretrained_backbone=False, min_size=192, max_size=192)
    randomize_norms(model)
    model.eval()
    x = torch.rand(1, 3, 192, 192)
    projections = model.roi_heads.box_roi_pool.projections

    def stages():
        features = model.backbone(x)
        return [*features.values(), *(projections[key](features[key]) for key in projections)]

    expected = stages()
    assert fuse_conv_bn(model) > 0
    assert not has_norm(model)
    for actual, reference in zip(stages(), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-4)


def test_use_channels_last_only_on_cpu():
    assert not use_channels_last(torch.device("cuda"))


@pytest.mark.parametrize("flag", ["--max-dets", "--max-det"])
def test_max_dets_override_is_validated(flag, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("sys.argv", ["benchmark.py", "--size", "320", flag, "0"])
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == 2
    message = capsys.readouterr().err.replace("\n", " ")
    assert "--max-dets" in message and "must be >= 1" in message


def test_load_images_accepts_single_file_or_folder(tmp_path):
    array = write_image(tmp_path / "b.jpg")
    write_image(tmp_path / "a.png")

    paths, images = load_images(tmp_path / "b.jpg", count=5)
    assert paths == [tmp_path / "b.jpg"]
    assert images[0].dtype == torch.uint8 and images[0].shape == (3, *array.shape[:2])
    _, folder_images = load_images(tmp_path, 5)
    assert torch.equal(images[0], folder_images[1])

    paths, _ = load_images(tmp_path, count=1)
    assert paths == [tmp_path / "a.png"]

    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="Unsupported image file"):
        load_images(tmp_path / "notes.txt", count=1)


def test_single_image_report_uses_checkpoint_architecture(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    # Narrower than the YAML model, so loading only succeeds if the checkpoint config is used.
    trained = copy.deepcopy(config)
    trained["model"].update(bifpn_channels=32, roi_channels=16, roi_head_dim=64)
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save({"model": light_rcnn_from_config(trained).state_dict(), "config": trained}, weights)
    image = tmp_path / "one.jpg"
    write_image(image)
    output = tmp_path / "run" / "light_rcnn_latency.json"
    monkeypatch.setattr("sys.argv", [
        "benchmark.py", "--model", "light_rcnn", "--weights", str(weights), "--device", "cpu",
        "--size", "128", "--source", str(image), "--image-count", "8", "--warmup", "1",
        "--iterations", "3", "--output", str(output),
    ])
    benchmark.main()

    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    report = json.loads(output.read_text(encoding="utf-8"))
    comparison = config["comparison"]
    assert report["schema_version"] == SCHEMA_VERSION
    assert (report["kind"], report["model"], report["image_size"]) == ("latency", "light_rcnn", 128)
    assert report["weights_sha256"] and report["environment"]["device"] == "cpu"
    latency = report["latency"]
    assert latency["scope_id"] == SCOPE_HOST_TO_DETECTIONS
    assert (latency["image_count"], latency["n"], latency["warmup"]) == (1, 3, 1)
    assert latency["images"] == [str(image)]
    assert latency["score_threshold"] == comparison["latency_score_threshold"]
    assert latency["max_detections"] == comparison["max_detections"]
    assert latency["conv_bn_fused"] is True
    assert latency["cold_start_ms"] > 0 and len(latency["samples_ms"]) == 3
    stats = report["model_stats"]
    assert stats["flops_input"] == [128, 128] and stats["gflops"] > 0
    assert stats["deployed_parameters_m"] < stats["parameters_m"]


def test_run_folder_config_without_comparison_keys_uses_shared_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    # train.py saves this config into each run folder; it predates the score/detection keys.
    config = yaml.safe_load((ROOT / "outputs/coco_4090_v2/config.yaml").read_text(encoding="utf-8"))
    assert "latency_score_threshold" not in config["comparison"]
    trained = copy.deepcopy(config)
    trained["model"].update(bifpn_channels=32, roi_channels=16, roi_head_dim=64)
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save({"model": light_rcnn_from_config(trained).state_dict(), "config": trained}, weights)
    image = tmp_path / "one.jpg"
    write_image(image)
    output = tmp_path / "light_rcnn_latency.json"
    monkeypatch.setattr("sys.argv", [
        "benchmark.py", "--config", "outputs/coco_4090_v2/config.yaml", "--model", "light_rcnn",
        "--weights", str(weights), "--device", "cpu", "--size", "128", "--source", str(image),
        "--warmup", "0", "--iterations", "1", "--output", str(output),
    ])
    benchmark.main()

    latency = json.loads(output.read_text(encoding="utf-8"))["latency"]
    assert latency["score_threshold"] == COMPARISON_DEFAULTS["latency_score_threshold"]
    assert latency["max_detections"] == COMPARISON_DEFAULTS["max_detections"]
