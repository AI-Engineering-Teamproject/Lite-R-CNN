"""Validate code-linked Lite R-CNN v3 docs, run values, and architecture assets."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path

import yaml
from PIL import Image


DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/coco_v3.yaml").read_text(encoding="utf-8"))
    run_cfg = yaml.safe_load((ROOT / "outputs/coco_4090_v3/config.yaml").read_text(encoding="utf-8"))
    train_report = (ROOT / "outputs/coco_4090_v3/RESULTS.md").read_text(encoding="utf-8")
    history = [json.loads(line) for line in (ROOT / "outputs/coco_4090_v3/history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gpu = json.loads((ROOT / "outputs/evaluation_runs/20261101_053245/evaluation.json").read_text(encoding="utf-8"))
    v3_result = next(item for item in gpu["results"] if item["model"] == "Lite R-CNN v3")

    model = config["model"]
    kd = config["distillation"]
    require(model["architecture"] == "light_rcnn_v3", "v3 architecture id")
    require(config["image_size"] == 640, "configured input size")
    require(config["num_classes"] == 91, "COCO classifier slot count")
    require(model["fpn_channels"] == 64 and model["roi_channels"] == 32, "FPN/RoI channels")
    require(model["rpn_anchor_sizes"] == [[8], [16], [32], [64], [128], [256]], "RPN anchor scales")
    require(model["rpn_pre_nms_topk_test"] == 1000 and model["rpn_post_nms_topk_test"] == 300, "test proposal limits")
    require(model["box_score_thresh"] == 0.05 and model["box_nms_thresh"] == 0.5, "detector thresholds")
    require(model["box_detections_per_img"] == 100, "final detection cap")
    require(kd["enabled"] is True and kd["feature_weight"] == 0.5, "feature KD settings")
    require(kd["roi_kl_weight"] == 1.0 and kd["temperature"] == 2.0, "RoI KD settings")
    require(run_cfg["initialization"]["type"] == "mobilenet_v3_backbone_only", "v2 initialization scope")

    best_epoch_line = next(line for line in train_report.splitlines() if "최고 AP@" in line)
    require("26.80%" in best_epoch_line and "epoch 42" in best_epoch_line, "training best metric context")
    epoch1 = next(item for item in history if item["epoch"] == 1)
    require("loss_kd_feature" in epoch1["losses"] and "loss_kd_roi" in epoch1["losses"], "KD losses present from epoch 1")
    require(history[-1]["epoch"] == 48, "recorded training reaches epoch 48")

    docs = {path.name: path.read_text(encoding="utf-8") for path in DOCS.glob("*.md")}
    for name in ("LITE_RCNN_V3_FULL.md", "LITE_RCNN_V3_QUICK.md", "PROGRESS.md"):
        require(name in docs, f"missing {name}")
    combined = "\n".join(docs.values())
    for key, expected in (("AP", f"{100 * v3_result['AP']:.4f}%"),
                          ("AP50", f"{100 * v3_result['AP50']:.4f}%"),
                          ("AP75", f"{100 * v3_result['AP75']:.4f}%")):
        require(expected in combined, f"documented GPU {key} matches saved evaluation")
    require("post-Lite-FPN" in combined and "P2-P5" in combined, "KD outputs identified as post-neck P2-P5")
    require(re.search(r"\bnot recorded\b|\bunrecorded\b|\bdoes not record\b", combined, re.IGNORECASE),
            "unrecorded environment fields remain explicit")
    require("p-value" in combined.lower(), "statistical p-value caveat present")
    require("0.25" in combined and "0.5" in combined and "0.05" in combined, "threshold distinctions present")
    require(not re.search(r"distill(?:ation|ed|s)?[^\n]{0,70}\braw C2-C5\b", combined, re.IGNORECASE),
            "no claim that raw C2-C5 are the feature KD targets")

    # Check every local Markdown link is resolvable from its file.
    checked_links = 0
    for name, text in docs.items():
        source = DOCS / name
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"^[a-z]+://", target, re.IGNORECASE):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (source.parent / target).resolve()
            require(resolved.exists(), f"broken local link in {name}: {target}")
            checked_links += 1

    svg_path = DOCS / "lite_rcnn_v3_architecture.svg"
    png_path = DOCS / "lite_rcnn_v3_architecture.png"
    require(svg_path.is_file() and png_path.is_file(), "architecture SVG/PNG exist")
    ET.parse(svg_path)  # fail on malformed SVG XML
    svg_text = unescape(svg_path.read_text(encoding="utf-8"))
    require(not re.search(r"[\uac00-\ud7a3]", svg_text), "architecture diagram contains no Hangul labels")
    generator_text = (DOCS / "generate_architecture.py").read_text(encoding="utf-8")
    for phrase in (
        "pre-NMS top-k <= 1000 / level",
        "post-NMS <= 300 / image",
        "score > 0.05",
        "P2-P5 feature maps",
    ):
        require(phrase in generator_text and phrase in svg_text, f"diagram includes source-backed label: {phrase}")
    with Image.open(png_path) as image:
        width, height = image.size
    require(width >= 2500 and height >= 1300, f"PNG resolution adequate ({width}x{height})")

    print("PASS: v3 config/run values, distillation losses, doc metrics, local links, SVG language, and PNG resolution")
    print(f"PASS: checked {checked_links} local Markdown links; diagram {width}x{height}")


if __name__ == "__main__":
    main()
