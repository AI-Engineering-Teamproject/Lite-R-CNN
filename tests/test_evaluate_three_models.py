"""scripts/evaluate_three_models.py on a tiny synthetic COCO set with stub R-CNN detectors.

The stubs run torchvision's own GeneralizedRCNN pipeline (transform, hooks, postprocess) with an
RPN that proposes the ground-truth box in the resized frame and RoI heads that turn proposals
into confident detections, so AP and proposal recall are exactly 1 when the frames are handled
right.
"""

import csv
import importlib.util
import json
import runpy
import sys
from collections import OrderedDict
from pathlib import Path
from urllib.error import URLError

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.misc import FrozenBatchNorm2d

from light_rcnn.evaluation import evaluation_subset
from light_rcnn.model.detector import build_detector
from light_rcnn.reporting import percentile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_three_models.py"
GPU_SCRIPT = ROOT / "scripts/evaluate_three_models_gpu_full.py"
WIDTH, HEIGHT = 80, 64
BOX = [4.0, 6.0, 40.0, 42.0]  # ground truth, original image frame
IMAGE_SIZE = 128  # 80x64 images are resized to 128x102: the two frames differ
IMAGE_IDS = list(range(101, 113))
EVALUATED, TIMING, WARMUP, ITERATIONS = 8, 3, 1, 6
# summary.csv header written by the 2026-11-01 runs (outputs/evaluation_runs/20261101_053245).
LEGACY_COLUMNS = [
    "model", "checkpoint", "images_evaluated", "image_size", "AP", "mAP_50_95", "AP50", "AP75",
    "AP_S", "AP_M", "AP_L", "AR@1", "AR@10", "AR@100", "AR_S", "AR_M", "AR_L",
    "Precision_IoU50_score0.5", "Recall_IoU50_score0.5", "precision_recall_definition",
    "parameters", "parameters_M", "trainable_parameters", "checkpoint_size_MB",
    "model_state_size_MB_fp32", "model_size_MB", "peak_VRAM_allocated_MB", "latency_p50_ms",
    "latency_p95_ms", "fps", "latency_iterations", "latency_p50_100_proposals_ms",
    "latency_p50_300_proposals_ms", "speedup_vs_faster_rcnn", "delta_AP_vs_faster_rcnn_pp",
    "backbone_latency_ms", "bifpn_latency_ms", "rpn_latency_ms", "roi_head_latency_ms",
    "proposal_recall_images_with_gt", "FLOPs_G_supported_ops", "FLOPs_coverage", "FLOPs_G",
    "evaluation_wall_time_seconds", "device", "precision_mode", "throughput_scope",
    "stage_backbone_ms", "stage_bifpn_ms", "stage_rpn_ms", "stage_roi_head_ms",
    "proposal_R@100_IoU50", "proposal_R@100_IoU75", "proposal_R@300_IoU50", "proposal_R@300_IoU75",
]
NEW_COLUMNS = ["latency_p99_ms", "latency_p95_of_image_medians_ms", "rpn_post_nms_top_n_test",
               "rpn_post_nms_top_n_native", "conv_bn_fused", "conv_bn_fused_layers"]
JSON_ONLY = {"latency_runs", "latency_timing_image_ids", "per_category_ap", "stage_latency_ms",
             "proposal_recall", "rpn_settings"}
THREE_MODELS = ["Faster R-CNN", "Lite R-CNN v2", "Lite R-CNN v3"]
FOUR_MODELS = ["Faster R-CNN", "MobileNetV3 Faster R-CNN", "Lite R-CNN v2", "Lite R-CNN v3"]


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4, padding=1, bias=False),
                                  nn.BatchNorm2d(4), nn.ReLU())
        self.neck = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), FrozenBatchNorm2d(4))
        self.out_channels = 4

    def forward(self, images):
        return OrderedDict(p3=self.neck(self.body(images)))


class StubRPN(nn.Module):
    """Proposes the ground-truth box in the resized frame; records the test budget of each call."""

    def __init__(self):
        super().__init__()
        self._pre_nms_top_n = {"training": 2000, "testing": 1000}
        self._post_nms_top_n = {"training": 2000, "testing": 1000}  # torchvision Faster R-CNN
        self.budgets = []

    def forward(self, images, features, targets=None):
        self.budgets.append(self._post_nms_top_n["testing"])
        scales = [torch.tensor([width / WIDTH, height / HEIGHT] * 2)
                  for height, width in images.image_sizes]
        return [torch.tensor([BOX]) * scale for scale in scales], {}


class StubRoIHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.score_thresh, self.nms_thresh, self.detections_per_img = 0.05, 0.5, 100

    def forward(self, features, proposals, image_shapes, targets=None):
        return [{"boxes": boxes, "scores": torch.full((len(boxes),), 0.9),
                 "labels": torch.ones(len(boxes), dtype=torch.int64)} for boxes in proposals], {}


def stub_detector(image_size: int) -> GeneralizedRCNN:
    transform = GeneralizedRCNNTransform(image_size, image_size, [0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    return GeneralizedRCNN(StubBackbone(), StubRPN(), StubRoIHeads(), transform)


def has_norm(model: nn.Module) -> bool:
    return any(isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)) for m in model.modules())


def load_script(path: Path = SCRIPT, name: str = "evaluate_three_models"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco")
    images = root / "images"
    images.mkdir()
    records, annotations = [], []
    for image_id in IMAGE_IDS:
        name = f"{image_id:012d}.jpg"
        Image.new("RGB", (WIDTH, HEIGHT), (image_id, 90, 140)).save(images / name)
        records.append({"id": image_id, "file_name": name, "width": WIDTH, "height": HEIGHT})
        annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                            "bbox": [4, 6, 36, 36], "area": 36 * 36, "iscrowd": 0})
    path = root / "instances.json"
    path.write_text(json.dumps({"images": records, "annotations": annotations,
                                "categories": [{"id": 1, "name": "thing"}]}))
    for version in ("v2", "v3", "mobilenet"):
        (root / f"{version}.pth").write_bytes(b"stub checkpoint")
    return root


def run(module, coco: Path, out: Path, *extra: str) -> dict:
    module.main(["--val-images", str(coco / "images"), "--annotations", str(coco / "instances.json"),
                 "--v2-weights", str(coco / "v2.pth"), "--v3-weights", str(coco / "v3.pth"),
                 "--image-size", str(IMAGE_SIZE), "--images", str(EVALUATED),
                 "--timing-images", str(TIMING), "--warmup", str(WARMUP),
                 "--iterations", str(ITERATIONS), "--stage-repeats", "3",
                 "--output-dir", str(out), *extra])
    return json.loads((out / "evaluation.json").read_text(encoding="utf-8"))


def patched(monkeypatch, fail_download: bool | str = False, real_mobilenet: bool = False):
    """Stub detectors for every kind. fail_download: the kind whose torchvision download fails
    (True = Faster R-CNN); real_mobilenet: the real MobileNetV3 Faster R-CNN with random weights
    (weights=None, weights_backbone=None, so nothing is downloaded) instead of its stub."""
    module = load_script()
    built = []
    offline = "faster_rcnn" if fail_download is True else fail_download

    def fake_load(kind, config, weights, image_size):
        if kind == offline and weights is None:
            raise URLError("Tunnel connection failed: 403 Forbidden")
        if real_mobilenet and kind == "mobilenet_faster_rcnn":
            torch.manual_seed(0)
            model = build_detector(kind, pretrained=False, num_classes=config["num_classes"],
                                   min_size=image_size, max_size=image_size)
        else:
            model = stub_detector(image_size)
        built.append({"kind": kind, "weights": weights, "image_size": image_size, "model": model})
        return model

    monkeypatch.setattr(module, "load_comparison_model", fake_load)
    return module, built


@pytest.fixture(scope="module")
def default_run(coco, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        module, built = patched(monkeypatch)
        out = tmp_path_factory.mktemp("run") / "default"
        payload = run(module, coco, out)
    return module, built, out, payload


@pytest.fixture(scope="module")
def mobilenet_run(coco, tmp_path_factory):
    """--with-mobilenet with a local MobileNet state dict (stub detectors throughout)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        module, built = patched(monkeypatch)
        out = tmp_path_factory.mktemp("run") / "mobilenet"
        payload = run(module, coco, out, "--with-mobilenet", "--mobilenet-weights",
                      str(coco / "mobilenet.pth"))
    return module, built, out, payload


def read_summary(out: Path) -> tuple[list[str], list[dict]]:
    with (out / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames, rows


def test_ap_and_proposal_recall_are_scored_in_the_original_frame(default_run):
    module, built, _, payload = default_run
    assert [row["model"] for row in payload["results"]] == list(module.MODELS)
    assert [entry["kind"] for entry in built] == ["faster_rcnn", "light_rcnn", "light_rcnn"]
    assert {entry["image_size"] for entry in built} == {IMAGE_SIZE}
    for row in payload["results"]:
        assert row["images_evaluated"] == EVALUATED
        assert row["AP"] == pytest.approx(1.0) and row["AP50"] == pytest.approx(1.0)
        assert row["Precision_IoU50_score0.5"] == row["Recall_IoU50_score0.5"] == 1.0
        assert row["proposal_recall"] == {key: 1.0 for key in row["proposal_recall"]}
        assert row["proposal_recall_images_with_gt"] == EVALUATED
        assert row["proposal_recall_frame"] == "original_image"
    # The old comparison (resized-frame proposals vs original-frame boxes) misses every box here.
    resized = torch.tensor([BOX]) * torch.tensor([IMAGE_SIZE / WIDTH, 102 / HEIGHT] * 2)
    assert module.proposal_recall(resized, torch.tensor([BOX]))["R@300_IoU50"] == 0.0


def test_every_model_is_scored_and_timed_with_300_proposals(default_run):
    _, built, out, payload = default_run
    warmup = max(WARMUP, TIMING)
    for entry, row in zip(built, payload["results"], strict=True):
        budgets = entry["model"].rpn.budgets
        assert budgets[:EVALUATED] == [300] * EVALUATED  # AP and proposal recall
        assert budgets.count(100) == warmup + ITERATIONS
        assert budgets[-4:] == [300] * 4  # stage profile and FLOPs after the 100-proposal run
        assert set(budgets) == {300, 100}
        assert row["rpn_post_nms_top_n_test"] == 300 and row["rpn_post_nms_top_n_native"] == 1000
        assert row["rpn_pre_nms_top_n_test"] == 1000
        assert [run_["proposals"] for run_ in row["latency_runs"]] == [300, 100]
        assert row["latency_p50_300_proposals_ms"] == row["latency_p50_ms"]
    assert payload["metadata"]["rpn_post_nms_top_n_test"] == 300
    assert "| 300 (native 1000) |" in (out / "REPORT.md").read_text(encoding="utf-8")


def test_latency_times_a_seeded_sample_of_the_evaluated_images(default_run):
    _, _, _, payload = default_run
    metadata = payload["metadata"]
    subset = evaluation_subset(IMAGE_IDS, EVALUATED, 42)
    order = torch.randperm(EVALUATED, generator=torch.Generator().manual_seed(42))[:TIMING]
    expected = [IMAGE_IDS[subset[k]] for k in order.tolist()]
    assert metadata["latency_timing_image_ids"] == expected
    assert set(expected) <= set(metadata["subset_image_ids"]) and len(set(expected)) == TIMING
    assert metadata["latency_warmup"] == max(WARMUP, TIMING)
    assert metadata["latency_iterations"] == ITERATIONS
    for row in payload["results"]:
        assert row["latency_timing_image_ids"] == expected
        assert row["latency_warmup"] == max(WARMUP, TIMING)
        for run_ in row["latency_runs"]:
            assert run_["image_ids"] == expected
            assert run_["sample_image_ids"] == expected * (ITERATIONS // TIMING)
            assert len(run_["samples_ms"]) == run_["iterations"] == ITERATIONS
            assert len(run_["image_medians_ms"]) == TIMING
            assert run_["p50_ms"] <= run_["p95_ms"] <= run_["p99_ms"]
        assert row["latency_p99_ms"] == row["latency_runs"][0]["p99_ms"]
        assert row["latency_p95_of_image_medians_ms"] == row["latency_runs"][0]["p95_of_image_medians_ms"]
        assert row["latency_p95_100_proposals_ms"] == row["latency_runs"][1]["p95_ms"]


def test_latency_stats_reports_tails_and_per_image_medians():
    module = load_script()
    samples, ids = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0], [7, 8, 9, 7, 8, 9]
    stats = module.latency_stats(samples, ids, [7, 8, 9])
    assert stats["p50_ms"] == 3.5
    assert stats["p95_ms"] == pytest.approx(17.5) and stats["p99_ms"] == pytest.approx(19.5)
    assert stats["image_medians_ms"] == [2.5, 6.0, 11.5]
    assert stats["p95_of_image_medians_ms"] == pytest.approx(percentile([2.5, 6.0, 11.5], 95))
    assert stats["samples_ms"] == samples and stats["sample_image_ids"] == ids


def test_conv_bn_folding_is_on_by_default_and_recorded(default_run, coco, tmp_path, monkeypatch):
    _, built, _, payload = default_run
    assert payload["metadata"]["fuse_conv_bn"] is True
    for entry, row in zip(built, payload["results"], strict=True):
        assert not has_norm(entry["model"])  # BatchNorm2d and FrozenBatchNorm2d both folded
        assert row["conv_bn_fused"] is True and row["conv_bn_fused_layers"] == 2

    module, unfused = patched(monkeypatch)
    payload = run(module, coco, tmp_path / "unfused", "--no-fuse")
    assert payload["metadata"]["fuse_conv_bn"] is False
    for entry, row in zip(unfused, payload["results"], strict=True):
        assert has_norm(entry["model"])
        assert row["conv_bn_fused"] is False and row["conv_bn_fused_layers"] == 0
        assert row["AP"] == pytest.approx(1.0)
    assert "미적용 (--no-fuse)" in (tmp_path / "unfused" / "REPORT.md").read_text(encoding="utf-8")


def test_outputs_keep_the_original_columns_first(default_run):
    _, _, out, payload = default_run
    with (out / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames
    assert header[:len(LEGACY_COLUMNS)] == LEGACY_COLUMNS
    assert set(NEW_COLUMNS) <= set(header[len(LEGACY_COLUMNS):])
    assert not JSON_ONLY & set(header)  # JSON only
    assert set(payload["results"][0]["rpn_settings"]) == {"score_thresh", "nms_thresh",
                                                          "anchor_sizes", "aspect_ratios"}
    assert [row["model"] for row in rows] == [row["model"] for row in payload["results"]]
    # The fields the dashboard's detection race page reads.
    for row in rows:
        assert float(row["AP"]) == pytest.approx(1.0)
        assert float(row["latency_p95_ms"]) >= float(row["latency_p50_ms"]) > 0
        assert row["device"] == "cpu"
    assert float(rows[0]["speedup_vs_faster_rcnn"]) == 1.0
    assert float(rows[0]["delta_AP_vs_faster_rcnn_pp"]) == 0.0
    for key in ("latency_warmup", "latency_iterations", "subset_image_ids", "image_size"):
        assert key in payload["metadata"]
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert report.startswith("# Three-model detection evaluation\n")
    assert ("\nFaster R-CNN has no BiFPN module; its BiFPN latency is null. VRAM is peak allocated "
            "CUDA memory.\n") in report
    assert "MobileNet" not in report and payload["metadata"]["with_mobilenet"] is False
    assert [row["model"] for row in payload["results"]] == THREE_MODELS
    assert ("| Model | AP/mAP | AP50 | AP75 | AP_S | AP_M | AP_L | Precision | Recall | p50 ms | FPS "
            "| Params M | FLOPs G* | p95 ms | p99 ms |") in report
    assert report.count("| Lite R-CNN v") == 2
    assert (out / "partial_results.json").is_file()


def test_profiler_failure_keeps_the_column_layout(coco, tmp_path, monkeypatch):
    def broken_profile(*args, **kwargs):
        raise RuntimeError("no profiler in this build")

    monkeypatch.setattr(torch.profiler, "profile", broken_profile)
    module, _ = patched(monkeypatch)
    payload = run(module, coco, tmp_path / "noflops")
    with (tmp_path / "noflops" / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames[:len(LEGACY_COLUMNS)] == LEGACY_COLUMNS
    assert rows[0]["FLOPs_G"] == "" and "no profiler" in rows[0]["FLOPs_error"]
    assert payload["results"][0]["FLOPs_G_supported_ops"] is None
    assert "| N/A |" in (tmp_path / "noflops" / "REPORT.md").read_text(encoding="utf-8")


def test_skip_faster_rcnn_evaluates_lite_models_only(coco, tmp_path, monkeypatch):
    module, built = patched(monkeypatch, fail_download=True)
    payload = run(module, coco, tmp_path / "skip", "--skip-faster-rcnn")
    assert [entry["kind"] for entry in built] == ["light_rcnn", "light_rcnn"]
    assert [row["model"] for row in payload["results"]] == ["Lite R-CNN v2", "Lite R-CNN v3"]
    assert payload["metadata"]["skipped_models"] == ["Faster R-CNN"]
    for row in payload["results"]:
        assert row["speedup_vs_faster_rcnn"] is None and row["delta_AP_vs_faster_rcnn_pp"] is None
    report = (tmp_path / "skip" / "REPORT.md").read_text(encoding="utf-8")
    assert "제외한 모델: Faster R-CNN" in report
    # The title and notes follow the models that ran: two, and no torchvision baseline.
    assert report.startswith("# Two-model detection evaluation\n") and "no BiFPN" not in report


@pytest.mark.parametrize("with_mobilenet", [False, True])
def test_failed_baseline_download_explains_the_options(coco, tmp_path, monkeypatch,
                                                       with_mobilenet):
    module, _ = patched(monkeypatch, fail_download=True)
    with pytest.raises(SystemExit) as error:
        run(module, coco, tmp_path / "offline", *(["--with-mobilenet"] if with_mobilenet else []))
    message = str(error.value)
    assert "--skip-faster-rcnn" in message and "--faster-weights" in message
    assert "partial_results.json" not in message  # Faster R-CNN runs first: nothing finished
    if with_mobilenet:  # skipping Faster R-CNN still leaves MobileNet's download
        assert "MobileNetV3 Faster R-CNN과 Lite R-CNN v2/v3만 평가하려면" in message
        assert "--mobilenet-weights" in message
    else:
        assert "Lite R-CNN v2/v3만 평가하려면" in message and "MobileNet" not in message
    assert not (tmp_path / "offline").exists()


@pytest.mark.parametrize("extra", [("--iterations", "2"), ("--timing-images", "0"),
                                   ("--warmup", "-1"), ("--images", "0")])
def test_invalid_counts_are_rejected(coco, tmp_path, monkeypatch, extra):
    module, _ = patched(monkeypatch)
    with pytest.raises(SystemExit) as error:
        run(module, coco, tmp_path / "invalid", *extra)
    assert error.value.code == 2


def test_with_mobilenet_adds_a_fourth_model_under_the_same_protocol(mobilenet_run, default_run,
                                                                    coco):
    module, built, out, payload = mobilenet_run
    assert [row["model"] for row in payload["results"]] == FOUR_MODELS
    assert list(module.MODELS_WITH_MOBILENET) == FOUR_MODELS and list(module.MODELS) == THREE_MODELS
    assert [entry["kind"] for entry in built] == ["faster_rcnn", "mobilenet_faster_rcnn",
                                                  "light_rcnn", "light_rcnn"]
    local = str((coco / "mobilenet.pth").resolve())  # --mobilenet-weights, a local state dict
    assert built[1]["weights"] == payload["results"][1]["checkpoint"] == local
    warmup = max(WARMUP, TIMING)
    for entry, row in zip(built, payload["results"], strict=True):
        budgets = entry["model"].rpn.budgets
        assert budgets[:EVALUATED] == [300] * EVALUATED and budgets.count(100) == warmup + ITERATIONS
        assert row["rpn_post_nms_top_n_test"] == 300 and row["rpn_post_nms_top_n_native"] == 1000
        assert row["AP"] == pytest.approx(1.0)  # proposals and boxes mapped to the original frame
        assert row["proposal_recall"] == {key: 1.0 for key in row["proposal_recall"]}
        assert row["conv_bn_fused"] is True and not has_norm(entry["model"])
        assert row["latency_timing_image_ids"] == payload["results"][0]["latency_timing_image_ids"]
    metadata = payload["metadata"]
    assert metadata["with_mobilenet"] is True and metadata["models"] == FOUR_MODELS
    assert metadata["skipped_models"] == []
    assert metadata["subset_image_ids"] == default_run[3]["metadata"]["subset_image_ids"]
    # Rows are added, the columns stay exactly those of the three-model run.
    header, rows = read_summary(out)
    assert header == read_summary(default_run[2])[0]
    assert [row["model"] for row in rows] == FOUR_MODELS
    assert float(rows[1]["speedup_vs_faster_rcnn"]) > 0
    assert float(rows[1]["delta_AP_vs_faster_rcnn_pp"]) == pytest.approx(0.0)
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert report.startswith("# Four-model detection evaluation\n")
    assert report.count("| MobileNetV3 Faster R-CNN |") == 1 and report.count("| Lite R-CNN v") == 2
    assert "Faster R-CNN and MobileNetV3 Faster R-CNN have no BiFPN module" in report
    # The note names the local state dict that ran, not torchvision's COCO_V1 numbers.
    assert module.mobilenet_note(local) in report and f"(로컬 가중치 {local})" in report
    assert "COCO_V1" not in report and "32.8" not in report
    assert "feat/personal-comparison-ci" not in report
    assert json.loads((out / "partial_results.json").read_text(encoding="utf-8"))["metadata"][
        "models_completed"] == FOUR_MODELS


def test_real_mobilenet_runs_the_whole_protocol_without_downloads(coco, tmp_path, monkeypatch,
                                                                 capsys):
    module, built = patched(monkeypatch, real_mobilenet=True)
    payload = run(module, coco, tmp_path / "real", "--with-mobilenet", "--skip-faster-rcnn")
    console = capsys.readouterr().out
    assert "MobileNetV3 Faster R-CNN도 같은 조건에서 함께 평가합니다" in console
    assert "네 번째" not in console  # three models ran
    assert [row["model"] for row in payload["results"]] == FOUR_MODELS[1:]
    assert payload["metadata"]["skipped_models"] == ["Faster R-CNN"]
    entry, row = built[0], payload["results"][0]
    assert entry["kind"] == "mobilenet_faster_rcnn" and entry["weights"] is None
    model = entry["model"]
    assert row["rpn_post_nms_top_n_native"] == 1000 and row["rpn_post_nms_top_n_test"] == 300
    assert model.rpn._post_nms_top_n["testing"] == 300  # the latency runs restored it
    # torchvision defaults that stay as they are, recorded in evaluation.json only.
    assert row["rpn_settings"]["score_thresh"] == pytest.approx(0.05)
    assert row["rpn_settings"]["anchor_sizes"] == [[32.0, 64.0, 128.0, 256.0, 512.0]] * 3
    assert row["rpn_settings"]["aspect_ratios"] == [[0.5, 1.0, 2.0]] * 3
    assert row["conv_bn_fused"] is True and row["conv_bn_fused_layers"] > 0 and not has_norm(model)
    # BackboneWithFPN: the backbone time includes the FPN; there is no BiFPN to time.
    stages = row["stage_latency_ms"]
    assert stages["bifpn"] is None and all(stages[key] > 0 for key in ("backbone", "rpn",
                                                                        "roi_head"))
    assert row["proposal_recall_images_with_gt"] == EVALUATED and row["image_size"] == IMAGE_SIZE
    assert row["latency_p50_ms"] > 0 and row["checkpoint"] == "torchvision COCO_V1"
    assert row["speedup_vs_faster_rcnn"] is None  # no Faster R-CNN in this run
    report = (tmp_path / "real" / "REPORT.md").read_text(encoding="utf-8")
    assert "제외한 모델: Faster R-CNN" in report and "| MobileNetV3 Faster R-CNN |" in report
    # Title and notes follow the three models that ran; torchvision's COCO_V1 weights are named.
    assert report.startswith("# Three-model detection evaluation\n")
    assert "MobileNetV3 Faster R-CNN has no BiFPN module" in report
    assert "Faster R-CNN and" not in report
    assert module.mobilenet_note("torchvision COCO_V1") in report and "box AP 32.8" in report


def test_with_mobilenet_blocked_download_explains_the_options(coco, tmp_path, monkeypatch):
    module, built = patched(monkeypatch, fail_download="mobilenet_faster_rcnn")
    with pytest.raises(SystemExit) as error:
        run(module, coco, tmp_path / "offline", "--with-mobilenet")
    message = str(error.value)
    assert message.startswith("MobileNetV3 Faster R-CNN 기본 가중치(torchvision COCO_V1)를 내려받지")
    assert "--mobilenet-weights" in message and "--with-mobilenet" in message
    partial_path = tmp_path / "offline" / "partial_results.json"
    assert message.endswith(f"앞서 끝난 모델(Faster R-CNN)의 결과는 {partial_path}에 있습니다.")
    # Faster R-CNN ran first and stays in the partial results.
    partial = json.loads(partial_path.read_text("utf-8"))
    assert partial["metadata"]["models_completed"] == ["Faster R-CNN"]
    assert [entry["kind"] for entry in built] == ["faster_rcnn"]


def test_with_mobilenet_first_blocked_download_points_at_no_partial_results(coco, tmp_path,
                                                                           monkeypatch):
    """--skip-faster-rcnn puts MobileNet first: when its download fails nothing has finished."""
    module, built = patched(monkeypatch, fail_download="mobilenet_faster_rcnn")
    with pytest.raises(SystemExit) as error:
        run(module, coco, tmp_path / "offline", "--with-mobilenet", "--skip-faster-rcnn")
    message = str(error.value)
    assert message.startswith("MobileNetV3 Faster R-CNN 기본 가중치(torchvision COCO_V1)를 내려받지")
    assert "--mobilenet-weights" in message and "--with-mobilenet" in message
    assert "partial_results.json" not in message and "앞서 끝난" not in message
    assert not (tmp_path / "offline").exists() and built == []


def test_gpu_full_script_differs_only_in_defaults(monkeypatch, capsys):
    shared = load_script()
    gpu = load_script(GPU_SCRIPT, "evaluate_three_models_gpu_full")
    cpu_defaults = vars(shared.build_parser().parse_args([]))
    gpu_defaults = vars(gpu.build_parser().parse_args([]))
    changed = {key for key in cpu_defaults if cpu_defaults[key] != gpu_defaults[key]}
    assert changed == {"device", "images"}
    assert (gpu_defaults["device"], gpu_defaults["images"]) == ("cuda", 5000)
    assert (cpu_defaults["device"], cpu_defaults["images"]) == ("cpu", 1000)
    assert gpu_defaults["fuse"] is True and gpu_defaults["timing_images"] == 64
    assert gpu_defaults["warmup"] >= 64 and gpu_defaults["iterations"] == 640
    assert gpu_defaults["with_mobilenet"] is cpu_defaults["with_mobilenet"] is False
    assert vars(gpu.build_parser().parse_args(["--with-mobilenet"]))["with_mobilenet"] is True

    monkeypatch.setattr(sys, "argv", [str(GPU_SCRIPT), "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(GPU_SCRIPT), run_name="__main__")
    assert error.value.code == 0
    text = capsys.readouterr().out
    assert "all COCO val2017 images" in text and "default: cuda" in text and "default: 5000" in text
    skip = next(action for action in gpu.build_parser()._actions
                if "--skip-faster-rcnn" in action.option_strings)
    assert "Lite R-CNN v2/v3 (and MobileNetV3 Faster R-CNN with --with-mobilenet)" in skip.help
