"""CPU integration checks on real detector paths and a tiny synthetic COCO set."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from light_rcnn.data import CocoTensorDataset
from light_rcnn.distillation import DistillationTrainer
from light_rcnn.evaluation import evaluate_coco
from light_rcnn.model.detector import LightRCNN, build_vanilla_faster_rcnn

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def batch():
    return (
        [torch.rand(3, 64, 80), torch.rand(3, 80, 64)],
        [
            {"boxes": torch.tensor([[4.0, 6.0, 40.0, 42.0]]), "labels": torch.tensor([1])}
            for _ in range(2)
        ],
    )


def student():
    return LightRCNN(
        pretrained_backbone=False,
        num_classes=3,
        bifpn_channels=16,
        roi_channels=8,
        min_size=64,
        max_size=64,
        rpn_pre_nms_topk_test=20,
        rpn_post_nms_topk_test=10,
    )


@pytest.fixture
def coco(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    records, annotations = [], []
    for index in range(2):
        name = f"{index:012d}.jpg"
        Image.new("RGB", (80, 64), (40 + index * 30, 80, 120)).save(images / name)
        records.append({"id": index, "file_name": name, "width": 80, "height": 64})
        annotations.append(
            {
                "id": index + 1,
                "image_id": index,
                "category_id": 1,
                "bbox": [4, 6, 36, 36],
                "area": 1296,
                "iscrowd": 0,
            }
        )
    path = tmp_path / "instances.json"
    path.write_text(
        json.dumps(
            {
                "info": {},
                "images": records,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "object"}],
            }
        )
    )
    return images, path


def test_detector_forward_backward_and_inference():
    model = student().train()
    images, targets = batch()
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.roi_heads.box_roi_pool.projections["0"][0].weight.grad is not None
    with torch.no_grad():
        output = model.eval()(images)
    assert len(output) == 2
    assert set(output[0]) == {"boxes", "scores", "labels"}


def test_kd_aligns_rois_freezes_teacher_and_backpropagates():
    teacher = build_vanilla_faster_rcnn(num_classes=3, pretrained=False, min_size=128, max_size=128)
    trainer = DistillationTrainer(student(), teacher).train()
    seen = {}

    def capture(name):
        def hook(module, args):
            seen[name] = (args[1], args[2])

        return hook

    trainer.student.roi_heads.box_roi_pool.register_forward_pre_hook(capture("student"))
    teacher.roi_heads.box_roi_pool.register_forward_pre_hook(capture("teacher"))
    teacher.transform.register_forward_pre_hook(
        lambda *_: pytest.fail("Teacher must not independently transform images")
    )
    losses = trainer(*batch())
    assert {"loss_kd_feature", "loss_kd_roi"} <= losses.keys()
    assert seen["student"][1] == seen["teacher"][1]
    assert all(torch.equal(a, b) for a, b in zip(seen["student"][0], seen["teacher"][0]))
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert trainer.adapters["0"].weight.grad.abs().sum() > 0
    assert trainer.student.roi_heads.box_predictor.cls_score.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training
    assert not teacher.backbone.body.bn1.training


def test_zero_kd_matches_supervised_losses():
    model = student().train()
    reference = deepcopy(model)
    teacher = build_vanilla_faster_rcnn(num_classes=3, pretrained=False)
    wrapper = DistillationTrainer(model, teacher, feature_weight=0, roi_kl_weight=0).train()
    images, targets = batch()
    torch.manual_seed(123)
    expected = reference(images, targets)
    torch.manual_seed(123)
    actual = wrapper(images, targets)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_empty_predictions_are_valid_zero_ap(coco):
    class EmptyDetector(torch.nn.Module):
        def forward(self, images):
            return [
                {
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.long),
                }
                for _ in images
            ]

    model = EmptyDetector().train()
    metrics = evaluate_coco(model, CocoTensorDataset(*coco), "cpu")
    assert metrics["AP_50_95"] == 0
    assert metrics["images_evaluated"] == 2
    assert model.training


def run_script(*args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_train_resume_best_and_saved_architecture(coco, tmp_path):
    images, annotations = coco
    config = yaml.safe_load((ROOT / "configs/coco.yaml").read_text())
    config.update(device="cpu", num_classes=3, image_size=64)
    config["data"] = {
        "train_images": str(images),
        "train_annotations": str(annotations),
        "val_images": str(images),
        "val_annotations": str(annotations),
    }
    config["model"].update(
        pretrained=False,
        bifpn_channels=16,
        roi_channels=8,
        rpn_pre_nms_topk_test=20,
        rpn_post_nms_topk_test=10,
    )
    config["train"].update(epochs=2, batch_size=2, workers=0, amp=False, warmup_steps=2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run_script("scripts/train.py", "--config", config_path, "--output-dir", full)
    run_script("scripts/train.py", "--config", config_path, "--epochs", 1, "--output-dir", resumed)
    run_script("scripts/train.py", "--resume", resumed / "last.pth", "--epochs", 2)
    expected = torch.load(full / "last.pth", weights_only=True)
    actual = torch.load(resumed / "last.pth", weights_only=True)
    assert actual["epoch"] == 2
    assert actual["global_step"] == expected["global_step"] == 2
    assert actual["scheduler"] == expected["scheduler"]
    for name in actual["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
    rows = [json.loads(row) for row in (resumed / "history.jsonl").read_text().splitlines()]
    best = torch.load(resumed / "best.pth", weights_only=True)
    assert len(rows) == 2
    assert best["metrics"]["AP_50_95"] == max(row["validation"]["AP_50_95"] for row in rows)
    assert best["epoch"] == actual["best_epoch"]
    report = tmp_path / "metrics.json"
    run_script(
        "scripts/evaluate_coco.py",
        "--model",
        "light_rcnn",
        "--weights",
        resumed / "best.pth",
        "--device",
        "cpu",
        "--output",
        report,
    )
    assert json.loads(report.read_text(encoding="utf-8"))["images_evaluated"] == 2
    run_script(
        "scripts/benchmark.py",
        "--model",
        "light_rcnn",
        "--weights",
        resumed / "best.pth",
        "--device",
        "cpu",
        "--source",
        images,
        "--size",
        64,
        "--warmup",
        1,
        "--iterations",
        2,
        "--output",
        tmp_path / "timing.json",
    )

    # Random teacher weights are strictly a no-download integration fixture.
    teacher_path = tmp_path / "teacher.pth"
    teacher = build_vanilla_faster_rcnn(num_classes=3, pretrained=False)
    torch.save(teacher.state_dict(), teacher_path)
    del teacher
    config["distillation"]["teacher"]["checkpoint"] = str(teacher_path)
    config_path.write_text(yaml.safe_dump(config))
    distilled = tmp_path / "distilled"
    run_script(
        "scripts/train.py",
        "--config",
        config_path,
        "--init-weights",
        resumed / "best.pth",
        "--distill",
        "--epochs",
        1,
        "--output-dir",
        distilled,
    )
    first = torch.load(distilled / "last.pth", weights_only=True)
    run_script("scripts/train.py", "--resume", distilled / "last.pth", "--epochs", 2)
    second = torch.load(distilled / "last.pth", weights_only=True)
    assert second["epoch"] == 2
    assert not any("teacher" in key for key in second["model"])
    assert any(
        not torch.equal(value, second["distillation_adapters"][key])
        for key, value in first["distillation_adapters"].items()
    )
    kd_history = [json.loads(row) for row in (distilled / "history.jsonl").read_text().splitlines()]
    assert all(
        "loss_kd_feature" in row["losses"] and "loss_kd_roi" in row["losses"] for row in kd_history
    )

    continuous_kd = tmp_path / "continuous_kd"
    run_script(
        "scripts/train.py",
        "--config",
        config_path,
        "--init-weights",
        resumed / "best.pth",
        "--distill",
        "--output-dir",
        continuous_kd,
    )
    uninterrupted = torch.load(continuous_kd / "last.pth", weights_only=True)
    for section in ("model", "distillation_adapters"):
        for key, value in second[section].items():
            torch.testing.assert_close(value, uninterrupted[section][key], rtol=0, atol=0)
