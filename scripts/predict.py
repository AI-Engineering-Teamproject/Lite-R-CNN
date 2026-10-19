"""Run a trained Lite R-CNN checkpoint on one image or a folder of images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from light_rcnn.model.detector import light_rcnn_from_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def coco_category_names(config: dict) -> dict[int, str]:
    annotation_path = Path(config["data"]["val_annotations"])
    if not annotation_path.exists():
        return {}
    with annotation_path.open(encoding="utf-8") as file:
        annotations = json.load(file)
    return {int(item["id"]): item["name"] for item in annotations.get("categories", [])}


def suppress_overlapping_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    nms_thresh: float = 0.45,
    containment_thresh: float = 0.75,
) -> torch.Tensor:
    """Filter duplicate overlapping boxes and sub-part detections per class."""
    import torchvision.ops

    if not boxes.numel():
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep_indices = []
    unique_labels = torch.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        cls_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        cls_nms_keep = torchvision.ops.nms(cls_boxes, cls_scores, nms_thresh)
        survived_indices = cls_indices[cls_nms_keep]
        survived_boxes = boxes[survived_indices]

        if containment_thresh < 1.0 and len(survived_boxes) > 1:
            filtered = []
            for i in range(len(survived_boxes)):
                box_i = survived_boxes[i]
                area_i = (box_i[2] - box_i[0]).clamp(min=0) * (box_i[3] - box_i[1]).clamp(min=0)
                if area_i <= 0:
                    continue
                contained = False
                for j in filtered:
                    box_j = survived_boxes[j]
                    x1 = torch.max(box_i[0], box_j[0])
                    y1 = torch.max(box_i[1], box_j[1])
                    x2 = torch.min(box_i[2], box_j[2])
                    y2 = torch.min(box_i[3], box_j[3])
                    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                    coverage = inter / (area_i + 1e-6)
                    if coverage >= containment_thresh:
                        contained = True
                        break
                if not contained:
                    filtered.append(i)
            survived_indices = survived_indices[filtered]
        keep_indices.append(survived_indices)
    if not keep_indices:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.cat(keep_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="outputs/coco_4090/best.pth")
    parser.add_argument("--source", required=True, help="Image path or directory of images")
    parser.add_argument("--output-dir", default="outputs/coco_4090/predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--nms-threshold", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.75,
        help="Suppress smaller sub-boxes heavily contained inside a larger box of the same class (default: 0.75; 1.0 to disable)",
    )
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = light_rcnn_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.roi_heads.score_thresh = min(0.05, args.score_threshold)
    model.roi_heads.nms_thresh = args.nms_threshold
    names = coco_category_names(config)

    source = Path(args.source)
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )[: args.max_images]
    if not paths:
        raise SystemExit(f"No supported images found: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for path in paths:
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        tensor = to_tensor(image).to(device)
        with torch.inference_mode():
            prediction = model([tensor])[0]

        pred_boxes = prediction["boxes"].detach()
        pred_scores = prediction["scores"].detach()
        pred_labels = prediction["labels"].detach()

        keep_idx = suppress_overlapping_detections(
            pred_boxes,
            pred_scores,
            pred_labels,
            nms_thresh=args.nms_threshold,
            containment_thresh=args.containment_threshold,
        )
        if keep_idx.numel():
            pred_boxes = pred_boxes[keep_idx]
            pred_scores = pred_scores[keep_idx]
            pred_labels = pred_labels[keep_idx]
        else:
            pred_boxes = pred_boxes[:0]
            pred_scores = pred_scores[:0]
            pred_labels = pred_labels[:0]

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        detections = []
        for box, label, score in zip(
            pred_boxes.cpu().tolist(),
            pred_labels.cpu().tolist(),
            pred_scores.cpu().tolist(),
        ):
            if score < args.score_threshold:
                continue
            name = names.get(int(label), f"class_{label}")
            text = f"{name} {score:.2f}"
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline=(255, 48, 48), width=3)
            bounds = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bounds, fill=(255, 48, 48))
            draw.text((x1, y1), text, fill="white", font=font)
            detections.append({"class": name, "score": round(score, 4), "box_xyxy": [round(v, 1) for v in box]})

        output_path = output_dir / f"{path.stem}_predictions.jpg"
        canvas.save(output_path, quality=95)
        print(json.dumps({
            "image": str(path),
            "output": str(output_path),
            "score_threshold": args.score_threshold,
            "detections": detections,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
