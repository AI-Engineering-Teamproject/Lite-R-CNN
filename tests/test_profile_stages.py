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
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from light_rcnn.model.detector import LightRoIHead, build_detector, light_rcnn_from_config
from light_rcnn.reporting import SCHEMA_VERSION, STAGE_LABELS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("profile_stages_script",
                                               ROOT / "scripts/profile_stages.py")
profile_stages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_stages)
PATCHED = {"transform": {"forward", "postprocess"}, "backbone": {"forward"},
           "rpn": {"forward", "filter_proposals"},
           "roi_heads": {"forward", "postprocess_detections"}}
BACKBONE_SUBSTAGES = ("backbone.body", "backbone.neck")


def random_mobilenet(image_size: int) -> torch.nn.Module:
    """torchvision's MobileNetV3-Large-FPN Faster R-CNN, random weights (nothing downloaded)."""
    torch.manual_seed(0)
    return build_detector("mobilenet_faster_rcnn", pretrained=False, num_classes=91,
                          min_size=image_size, max_size=image_size)


class PlainBackbone(nn.Sequential):
    """One feature map from one module: no body/neck split to time."""

    def __init__(self):
        super().__init__(nn.Conv2d(3, 8, 3, stride=8, padding=1), nn.ReLU())
        self.out_channels = 8


def plain_detector(image_size: int) -> FasterRCNN:
    torch.manual_seed(0)
    return FasterRCNN(PlainBackbone(), num_classes=2, min_size=image_size, max_size=image_size,
                      rpn_anchor_generator=AnchorGenerator(((32,),), ((1.0,),)),
                      box_roi_pool=MultiScaleRoIAlign(["0"], 7, 2))


def tiny_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text(encoding="utf-8"))
    # Narrower than the YAML model, so loading only succeeds if the checkpoint config is used.
    trained = copy.deepcopy(config)
    trained["model"].update(bifpn_channels=32, roi_channels=16, roi_head_dim=64)
    return trained


def write_image(path: Path, height: int = 96, width: int = 128) -> None:
    array = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


@pytest.mark.parametrize("flag", ["--max-dets", "--max-det"])
def test_max_dets_override_is_validated(flag, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("sys.argv", ["profile_stages.py", "--imgsz", "320", flag, "0"])
    with pytest.raises(SystemExit) as exit_info:
        profile_stages.main()
    assert exit_info.value.code == 2
    message = capsys.readouterr().err.replace("\n", " ")
    assert "--max-dets" in message and "must be >= 1" in message


def test_instrument_times_every_stage_of_light_rcnn_and_restores_methods():
    torch.manual_seed(0)
    model = light_rcnn_from_config(dict(tiny_config(), image_size=128)).eval()
    assert isinstance(model.roi_heads.box_head, LightRoIHead)
    timer = profile_stages.StageTimer(torch.device("cpu"))
    with torch.inference_mode(), profile_stages.instrument(model, timer):
        record, wall, outputs = profile_stages.profile_call(
            model, torch.randint(0, 256, (3, 96, 128), dtype=torch.uint8), torch.device("cpu"),
            timer)
    assert set(record) == {*STAGE_LABELS, *profile_stages.SUBSTAGES}
    assert sum(record[stage] for stage in STAGE_LABELS) <= wall
    assert record["backbone.body"] + record["backbone.neck"] <= record["backbone"]
    for stage, (parent, _) in profile_stages.SUBSTAGES.items():
        assert record[stage] <= record[parent]
    assert set(outputs[0]) == {"boxes", "labels", "scores"}
    for name, methods in PATCHED.items():
        assert not methods & set(vars(getattr(model, name)))
    assert "forward" not in vars(model.backbone.body) and "forward" not in vars(model.backbone.neck)


def test_instrument_splits_the_torchvision_backbone_into_body_and_fpn():
    model = random_mobilenet(128).eval()
    parts = profile_stages.backbone_parts(model)
    assert parts == {"backbone.body": model.backbone.body, "backbone.neck": model.backbone.fpn}
    timer = profile_stages.StageTimer(torch.device("cpu"))
    with torch.inference_mode(), profile_stages.instrument(model, timer):
        record, _, outputs = profile_stages.profile_call(
            model, torch.randint(0, 256, (3, 96, 128), dtype=torch.uint8), torch.device("cpu"),
            timer)
    assert set(record) == {*STAGE_LABELS, *profile_stages.SUBSTAGES}
    assert record["backbone.body"] + record["backbone.neck"] <= record["backbone"]
    for stage, (parent, _) in profile_stages.SUBSTAGES.items():
        assert record[stage] <= record[parent]
    assert set(outputs[0]) == {"boxes", "labels", "scores"}
    for part in (model.backbone.body, model.backbone.fpn):
        assert "forward" not in vars(part)


def run_profile(monkeypatch, tmp_path, model_kind: str, build) -> dict:
    """profile_stages.main for --model model_kind with the loader replaced by build(image_size)."""
    monkeypatch.chdir(ROOT)
    built = {}

    def fake_load(kind, config, weights=None, image_size=None):
        built.update(kind=kind, weights=weights, image_size=image_size)
        return build(image_size)

    monkeypatch.setattr(profile_stages, "load_comparison_model", fake_load)
    image = tmp_path / "one.jpg"
    write_image(image)
    output = tmp_path / f"{model_kind}_stages.json"
    monkeypatch.setattr("sys.argv", [
        "profile_stages.py", "--model", model_kind, "--device", "cpu", "--imgsz", "128",
        "--source", str(image), "--warmup", "0", "--iterations", "2", "--output", str(output),
    ])
    profile_stages.main()
    return {"built": built, "report": json.loads(output.read_text(encoding="utf-8"))}


def test_mobilenet_baseline_stages_report_times_backbone_fpn_rpn_and_roi_heads(
        tmp_path, monkeypatch, capsys):
    result = run_profile(monkeypatch, tmp_path, "mobilenet_faster_rcnn", random_mobilenet)
    printed = json.loads(capsys.readouterr().out)
    assert result["built"] == {"kind": "mobilenet_faster_rcnn", "weights": None,
                               "image_size": 128}
    report = result["report"]
    assert (report["kind"], report["model"]) == ("stages", "mobilenet_faster_rcnn")
    assert report["weights"] == "torchvision_coco_v1"
    section = report["stages"]
    assert [item["stage"] for item in section["items"]] == list(STAGE_LABELS)
    assert [(item["stage"], item["parent"]) for item in section["substages"]] == [
        (stage, parent) for stage, (parent, _) in profile_stages.SUBSTAGES.items()]
    assert section["unavailable_substages"] == [] and printed["unavailable_substages"] == []
    assert section["conv_bn_fused"] is True
    means = {item["stage"]: item["mean_ms"] for item in [*section["items"], *section["substages"]]}
    assert all(means[stage] > 0 for stage in ("backbone", "rpn", "roi_heads", *BACKBONE_SUBSTAGES))


def test_a_backbone_without_body_or_fpn_records_what_is_missing(tmp_path, monkeypatch, capsys):
    result = run_profile(monkeypatch, tmp_path, "faster_rcnn", plain_detector)
    printed = json.loads(capsys.readouterr().out)
    section = result["report"]["stages"]
    assert [item["stage"] for item in section["items"]] == list(STAGE_LABELS)
    assert [item["stage"] for item in section["substages"]] == [
        "rpn.filter_proposals", "roi_heads.postprocess_detections"]
    missing = section["unavailable_substages"]
    assert [(item["stage"], item["parent"]) for item in missing] == [
        (stage, "backbone") for stage in BACKBONE_SUBSTAGES]
    assert missing[0]["reason"] == ("백본에 body 모듈이 없어 '백본 + FPN/BiFPN' 시간에만 들어 "
                                    "있습니다")
    assert "neck/fpn 모듈이 없어" in missing[1]["reason"]
    assert printed["unavailable_substages"] == list(BACKBONE_SUBSTAGES)


def test_stages_report_uses_checkpoint_architecture(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    trained = tiny_config()
    torch.manual_seed(0)
    weights = tmp_path / "tiny.pth"
    torch.save({"model": light_rcnn_from_config(trained).state_dict(), "config": trained}, weights)
    image = tmp_path / "one.jpg"
    write_image(image)
    output = tmp_path / "run" / "light_rcnn_stages.json"
    monkeypatch.setattr("sys.argv", [
        "profile_stages.py", "--weights", str(weights), "--device", "cpu", "--imgsz", "128",
        "--max-det", "7", "--source", str(image), "--image-count", "4", "--warmup", "0",
        "--iterations", "2", "--output", str(output),
    ])
    profile_stages.main()

    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == SCHEMA_VERSION
    assert (report["kind"], report["model"], report["image_size"]) == ("stages", "light_rcnn", 128)
    assert report["weights_sha256"] and report["environment"]["device"] == "cpu"
    section = report["stages"]
    assert section["scope_id"] == profile_stages.SCOPE_ID
    assert (section["image_count"], section["iterations"], section["warmup"]) == (1, 2, 0)
    assert section["images"] == [str(image)] and section["max_detections"] == 7
    assert section["conv_bn_fused"] is True
    assert [item["stage"] for item in section["items"]] == list(STAGE_LABELS)
    assert sum(item["share"] for item in section["items"]) == pytest.approx(1, abs=1e-3)
    assert [(item["stage"], item["parent"]) for item in section["substages"]] == [
        (stage, parent) for stage, (parent, _) in profile_stages.SUBSTAGES.items()]
    assert section["unavailable_substages"] == []  # BiFPN timed apart from the MobileNetV3 body
    assert section["total_mean_ms"] <= section["wall_mean_ms"]
    assert section["overhead_note"] == profile_stages.OVERHEAD_NOTE


def test_overhead_note_avoids_internal_names():
    for name in ("total", "share", "substages", "wall_", "GeneralizedRCNN", "synchronize"):
        assert name not in profile_stages.OVERHEAD_NOTE
