import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from light_rcnn.model.detector import LightRoIHead, light_rcnn_from_config
from light_rcnn.reporting import SCHEMA_VERSION, STAGE_LABELS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("profile_stages_script",
                                               ROOT / "scripts/profile_stages.py")
profile_stages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_stages)
PATCHED = {"transform": {"forward", "postprocess"}, "backbone": {"forward"},
           "rpn": {"forward", "filter_proposals"},
           "roi_heads": {"forward", "postprocess_detections"}}


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
    for stage, (parent, _) in profile_stages.SUBSTAGES.items():
        assert record[stage] <= record[parent]
    assert set(outputs[0]) == {"boxes", "labels", "scores"}
    for name, methods in PATCHED.items():
        assert not methods & set(vars(getattr(model, name)))


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
    assert section["total_mean_ms"] <= section["wall_mean_ms"]
    assert section["overhead_note"] == profile_stages.OVERHEAD_NOTE


def test_overhead_note_avoids_internal_names():
    for name in ("total", "share", "substages", "wall_", "GeneralizedRCNN", "synchronize"):
        assert name not in profile_stages.OVERHEAD_NOTE
