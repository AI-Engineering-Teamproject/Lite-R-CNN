"""Evaluate YOLO26 on COCO val with the same pycocotools path as the R-CNN models (schema-v2)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keep Ultralytics settings inside the project so read-only user profiles work.
_ultralytics_config = ROOT / "outputs/.ultralytics"
_ultralytics_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_config))

# torch must load before ultralytics, whose import sets OMP_NUM_THREADS=1 for later libraries.
import torch
import yaml
from pycocotools.coco import COCO
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import nms, ops

from light_rcnn.bench import resolve_device
from light_rcnn.coco_eval import summarize_detections
from light_rcnn.data import CocoTensorDataset
from light_rcnn.evaluation import evaluation_subset, subset_record
from light_rcnn.reporting import (
    build_report,
    collect_environment,
    dataset_info,
    default_report_path,
    display_path,
    file_sha256,
    write_report,
)

# Configs saved before comparison.score_threshold/max_detections existed lack them.
DEFAULT_SCORE_THRESHOLD = 0.001
DEFAULT_MAX_DETECTIONS = 100


class MultiLabelPredictor(DetectionPredictor):
    """Predict-mode NMS with multi_label=True, as in Ultralytics' COCO validator and RoIHeads."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(
            preds, self.args.conf, kwargs.pop("iou", self.args.iou), self.args.classes,
            self.args.agnostic_nms, multi_label=True, max_det=self.args.max_det,
            end2end=getattr(self.model, "end2end", False))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(preds, img, orig_imgs, **kwargs)


def ultralytics_device(device: torch.device) -> str:
    """Ultralytics takes 'cpu' or a bare CUDA index."""
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return str(device.index or 0)
    raise SystemExit(f"Unsupported --device {device}; use cpu, 0, cuda or cuda:0")


def select_images(dataset: CocoTensorDataset, limit: int, seed: int) -> list[int]:
    """The image ids evaluate_coco.py scores for the same annotations, --limit and seed."""
    return [dataset.ids[index] for index in evaluation_subset(dataset.ids, limit, seed)]


def category_map(names: dict[int, str], coco: COCO) -> dict[int, int]:
    """Map YOLO class indices to COCO category ids by name, else by the standard 80->91 table."""
    by_name = {category["name"]: category_id for category_id, category in coco.cats.items()}
    fallback = coco80_to_coco91_class()
    mapping, unmatched = {}, []
    for index, name in names.items():
        if name in by_name:
            mapping[int(index)] = by_name[name]
        elif int(index) < len(fallback):
            mapping[int(index)] = fallback[int(index)]
            unmatched.append(f"{index}:{name}->{fallback[int(index)]}")
        else:
            unmatched.append(f"{index}:{name}->skipped")
    if unmatched:
        print(f"Classes not matched by name: {', '.join(unmatched)}", file=sys.stderr)
    return mapping


def inference_mode(model: YOLO) -> str:
    """Report the head mode Ultralytics actually ran: the predictor backend first, then the head."""
    layers = getattr(model.model, "model", None)
    head = layers[-1] if layers is not None and len(layers) else None
    backend = getattr(getattr(model, "predictor", None), "model", None)
    for source in (backend, model.model, head):
        flag = getattr(source, "end2end", None)
        if isinstance(flag, bool):
            return "end2end (NMS-free)" if flag else "NMS"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco.yaml",
                        help="Shared settings (data, seed, image_size, device, comparison.*)")
    parser.add_argument("--weights", help="Defaults to comparison.control.weights or yolo26n.pt")
    parser.add_argument("--images", help="Defaults to the config val_images")
    parser.add_argument("--annotations", help="Defaults to the config val_annotations")
    parser.add_argument("--imgsz", type=int, help="Override config image_size")
    parser.add_argument("--device", help="cpu, 0, cuda or cuda:0; defaults to the config device")
    parser.add_argument("--nms-free", action="store_true",
                        help="Run the one-to-one (NMS-free) head instead of the default NMS path")
    parser.add_argument("--limit", type=int, default=0,
                        help="Seeded random subset (config seed), the same images "
                             "evaluate_coco.py picks; 0 evaluates the full val set")
    parser.add_argument("--score-thresh", type=float, help="Override comparison.score_threshold")
    parser.add_argument("--max-dets", type=int, help="Override comparison.max_detections")
    parser.add_argument("--output", help="Defaults to outputs/reports/yolo26_accuracy.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    comparison = config.get("comparison") or {}
    imgsz = int(args.imgsz if args.imgsz is not None else config["image_size"])
    score_threshold = (args.score_thresh if args.score_thresh is not None
                       else comparison.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    max_detections = (args.max_dets if args.max_dets is not None
                      else comparison.get("max_detections", DEFAULT_MAX_DETECTIONS))
    weights = args.weights or (comparison.get("control") or {}).get("weights") or "yolo26n.pt"
    if args.limit < 0 or max_detections < 1 or imgsz < 1:
        parser.error("--limit must be >= 0; --max-dets and --imgsz must be >= 1")
    images_dir = Path(args.images or config["data"]["val_images"])
    annotations = Path(args.annotations or config["data"]["val_annotations"])
    seed = config.get("seed", 0)

    # The dataset class evaluate_coco.py uses, so the ids, their order and the subset are identical.
    dataset = CocoTensorDataset(images_dir, str(annotations))
    coco = dataset.coco
    image_ids = select_images(dataset, args.limit, seed)
    paths = [images_dir / coco.imgs[image_id]["file_name"] for image_id in image_ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} annotated images are missing, e.g. {missing[0]}")

    try:
        torch_device = resolve_device(str(args.device or config.get("device", "cuda")), "evaluating")
    except ValueError as exc:
        parser.error(f"--device: {exc}")
    yolo_device, device = ultralytics_device(torch_device), str(torch_device)
    # select_device() lowers torch threads to cpu_count - 1 on the first CPU predict; keep the
    # process default so YOLO runs with the same thread count as the R-CNN scripts.
    threads = torch.get_num_threads()
    model = YOLO(weights)
    categories = category_map(model.names, coco)
    predict_kwargs = {"imgsz": imgsz, "device": yolo_device, "conf": score_threshold,
                      "max_det": max_detections, "verbose": False}
    # Recent 8.4.x releases replaced half/end2end with quantize/nms (fp32 is the default there);
    # passing the old keys warns. --nms-free maps exactly as in benchmark_yolo26.py. Accuracy uses
    # multi-label NMS like the R-CNN RoIHeads; benchmark_yolo26.py keeps the deployment predictor.
    cfg_keys = DEFAULT_CFG_DICT.keys()
    if "half" in cfg_keys:
        predict_kwargs["half"] = False
    if args.nms_free:
        predict_kwargs.update({"end2end": True} if "end2end" in cfg_keys else {"nms": False})

    detections = []
    start = time.perf_counter()
    for image_id, path in zip(tqdm(image_ids, desc="COCO val: yolo26"), paths):
        results = model.predict(source=str(path), predictor=MultiLabelPredictor, **predict_kwargs)
        if torch.get_num_threads() != threads:  # only the first predict runs select_device()
            torch.set_num_threads(threads)
        boxes = results[0].boxes
        for box, label, score in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(),
                                     boxes.conf.cpu().tolist()):
            if int(label) not in categories:
                continue
            x1, y1, x2, y2 = box
            detections.append({
                "image_id": int(image_id),
                "category_id": categories[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    eval_seconds = time.perf_counter() - start
    if not detections:
        print(f"WARNING: yolo26 returned no detections above score {score_threshold} on "
              f"{len(image_ids)} images; writing an all-zero AP report.", file=sys.stderr)
    summary = summarize_detections(coco, detections, list(image_ids))

    fp16 = bool(getattr(getattr(model.predictor, "model", None), "fp16", False))
    mode = inference_mode(model)
    weights_file = getattr(model, "ckpt_path", None) or weights
    accuracy = {
        "evaluator": "pycocotools",
        "inference_mode": mode,
        "nms_multi_label": mode == "NMS",
        **subset_record(image_ids, len(dataset.ids), seed),
        "score_threshold": score_threshold,
        "max_detections": max_detections,
        "eval_seconds": eval_seconds,
        "dataset": dataset_info(annotations, images_dir),
        **summary,
    }
    report = build_report(
        "accuracy",
        "yolo26",
        weights=display_path(weights_file) if Path(weights_file).is_file() else str(weights),
        weights_sha256=file_sha256(weights_file),
        config=display_path(args.config),
        image_size=imgsz,
        batch_size=1,
        precision="fp16" if fp16 else "fp32",
        environment=collect_environment(device),
        accuracy=accuracy,
        # Pre-v2 reports kept this at the top level; older readers still look for it there.
        images_evaluated=accuracy["images_evaluated"],
    )
    output = write_report(report, args.output or default_report_path("accuracy", "yolo26"))
    hidden = {"per_category_ap", "image_ids"}
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "config": report["config"],
        "device": device,
        "image_size": report["image_size"],
        "precision": report["precision"],
        "accuracy": {key: value for key, value in accuracy.items() if key not in hidden},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
