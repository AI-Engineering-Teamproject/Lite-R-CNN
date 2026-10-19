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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="outputs/coco_4090/best.pth")
    parser.add_argument("--source", required=True, help="Image path or directory of images")
    parser.add_argument("--output-dir", default="outputs/coco_4090/predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = light_rcnn_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
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

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        detections = []
        for box, label, score in zip(
            prediction["boxes"].detach().cpu().tolist(),
            prediction["labels"].detach().cpu().tolist(),
            prediction["scores"].detach().cpu().tolist(),
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
