"""Evaluate the YOLO26 reference on an Ultralytics COCO dataset YAML."""

import argparse
import json
import os
from pathlib import Path

# Keep Ultralytics settings inside the project so read-only user profiles work.
_ultralytics_config = Path("outputs/.ultralytics").resolve()
_ultralytics_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_config))

from ultralytics import YOLO
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--images", default="data/coco/images/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit; 0 evaluates full val set")
    parser.add_argument("--output", default="outputs/yolo26_coco_metrics.json")
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    annotations_path = Path(args.annotations).resolve()
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_paths = sorted(
        path.resolve() for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit(f"No images found under {images_dir}")
    categories = sorted(coco["categories"], key=lambda category: category["id"])
    category_to_index = {category["id"]: index for index, category in enumerate(categories)}
    names = {index: category["name"] for index, category in enumerate(categories)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    coco_root = annotations_path.parent.parent
    labels_dir = coco_root / "labels" / "val2017"
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_by_image = {}
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0) == 0 and annotation["category_id"] in category_to_index:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
    for info in coco["images"]:
        label_path = labels_dir / f"{Path(info['file_name']).stem}.txt"
        lines = []
        width, height = info["width"], info["height"]
        for annotation in annotations_by_image.get(info["id"], []):
            x, y, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            lines.append(f"{category_to_index[annotation['category_id']]} {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")
    image_list = output.parent / "val2017.txt"
    image_list.write_text("\n".join(str(path) for path in image_paths), encoding="utf-8")
    dataset_yaml = output.parent / "coco_val.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(coco_root),
        # Ultralytics requires both keys even when this script only calls model.val().
        "train": str(image_list.resolve()),
        "val": str(image_list.resolve()),
        "names": names,
    }, sort_keys=False), encoding="utf-8")

    model = YOLO(args.weights)
    metrics = model.val(data=str(dataset_yaml), imgsz=args.imgsz, device=args.device, batch=args.batch,
                        workers=args.workers, half=False, plots=False, verbose=False)
    report = {
        "model": "yolo26",
        "model_name": "YOLO26",
        "weights": args.weights,
        "data": str(dataset_yaml),
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "images_evaluated": len(image_paths),
        "accuracy": {
            "AP_50_95": float(metrics.box.map),
            "AP_50": float(metrics.box.map50),
            "AP_75": float(metrics.box.map75),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
