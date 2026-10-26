"""Measure YOLO26 latency from decoded host images to final detections (schema-v2 report)."""

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
import cv2
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from light_rcnn.bench import IMAGE_EXTENSIONS, resolve_device
from light_rcnn.reporting import (
    COLD_START_FIRST_PREDICT,
    SCOPE_HOST_TO_DETECTIONS,
    SCOPE_LABELS,
    build_report,
    collect_environment,
    default_report_path,
    display_path,
    file_sha256,
    latency_summary,
    write_report,
)


def ultralytics_device(device: torch.device) -> str:
    """Ultralytics takes 'cpu' or a bare CUDA index."""
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return str(device.index or 0)
    raise SystemExit(f"Unsupported --device {device}; use cpu, 0, cuda or cuda:0")


def load_images(source: str | Path, count: int) -> tuple[list[Path], list]:
    """Decode one image file, or the first ``count`` images of a folder, into uint8 BGR arrays."""
    source = Path(source)
    candidates = [source] if source.is_file() else source.rglob("*")
    paths = sorted(path for path in candidates if path.suffix.lower() in IMAGE_EXTENSIONS)[:count]
    if not paths:
        raise SystemExit(f"No benchmark images found under {source}")
    images = [cv2.imread(str(path)) for path in paths]
    failed = [str(path) for path, image in zip(paths, images) if image is None]
    if failed:
        raise SystemExit(f"Failed to decode benchmark images: {', '.join(failed)}")
    return paths, images


def timed_call(model: YOLO, image, predict_kwargs: dict, device: str) -> tuple[float, int]:
    start = time.perf_counter()
    results = model.predict(source=image, **predict_kwargs)
    if device != "cpu":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000, len(results[0].boxes)


def detection_head(model: YOLO) -> torch.nn.Module | None:
    layers = getattr(model.model, "model", None)
    return layers[-1] if layers is not None and len(layers) else None


def inference_mode(model: YOLO) -> str:
    """Report the head mode Ultralytics actually ran: the predictor backend first, then the head."""
    backend = getattr(getattr(model, "predictor", None), "model", None)
    for source in (backend, model.model, detection_head(model)):
        flag = getattr(source, "end2end", None)
        if isinstance(flag, bool):
            return "end2end (NMS-free)" if flag else "NMS"
    return "unknown"


def count_flops(net: torch.nn.Module | None, imgsz: int) -> float | None:
    try:
        from ultralytics.utils.torch_utils import get_flops

        value = get_flops(net, imgsz)  # returns 0.0 when thop is missing or profiling fails
    except Exception as exc:  # noqa: BLE001 - FLOPs are context only, never fatal
        print(f"FLOP count unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return float(value) if value else None


def predictor_graph(model: YOLO) -> torch.nn.Module | None:
    """The fused nn.Module the predictor ran (AutoBackend keeps it on its PyTorch backend)."""
    backend = getattr(getattr(model, "predictor", None), "model", None)
    net = getattr(getattr(backend, "backend", None), "model", None)
    net = net if isinstance(net, torch.nn.Module) else getattr(backend, "model", None)
    return net if isinstance(net, torch.nn.Module) else None


def graph_layout(net: torch.nn.Module | None) -> dict:
    """Memory format and Conv+BN fusion state of the graph that ran."""
    if net is None:
        return {"memory_format": None, "conv_bn_fused": None}
    # 1x1 and single-channel kernels are contiguous in both layouts, so they cannot tell.
    channels_last = any(
        module.weight.is_contiguous(memory_format=torch.channels_last)
        and not module.weight.is_contiguous()
        for module in net.modules() if isinstance(module, torch.nn.Conv2d)
    )
    return {
        "memory_format": "channels_last" if channels_last else "contiguous",
        "conv_bn_fused": not any(isinstance(module, torch.nn.BatchNorm2d)
                                 for module in net.modules()),
    }


def model_stats(model: YOLO, imgsz: int) -> dict:
    """Parameters and FLOPs of the fused graph the predictor ran; checkpoint_* keep the .pt counts."""
    net = model.model
    deployed = predictor_graph(model)
    checkpoint_params = sum(param.numel() for param in net.parameters()) / 1e6
    deployed_params = (sum(param.numel() for param in deployed.parameters()) / 1e6
                       if deployed is not None else None)
    gflops = count_flops(deployed, imgsz) if deployed is not None else None
    method = ("Ultralytics get_flops (thop, multiply-add = 2) on the fused graph that ran, "
              f"{imgsz}x{imgsz} input" if gflops is not None
              else "FLOP count unavailable for the fused graph that ran (thop failed)")
    return {
        "parameters_m": deployed_params if deployed_params is not None else checkpoint_params,
        "checkpoint_parameters_m": checkpoint_params,
        "parameters_method": ("fused graph that ran (BatchNorm folded, inactive detection head "
                              "removed)" if deployed_params is not None
                              else "unfused checkpoint (predictor graph unavailable)"),
        "gflops": gflops,
        "checkpoint_gflops": count_flops(net, imgsz),
        "flops_input": [imgsz, imgsz],
        "flops_method": method,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/coco.yaml",
                        help="Shared comparison settings (comparison.*, image_size, device)")
    parser.add_argument("--weights", help="Defaults to comparison.control.weights or yolo26n.pt")
    parser.add_argument("--source", help="Image file or folder shared by all compared models; "
                                         "defaults to the config val_images")
    parser.add_argument("--imgsz", type=int, help="Override config image_size")
    parser.add_argument("--device", help="cpu, 0, cuda or cuda:0; defaults to the config device")
    parser.add_argument("--half", action="store_true", help="FP16 (CUDA only; ignored on cpu)")
    parser.add_argument("--nms-free", action="store_true",
                        help="Run the one-to-one (NMS-free) head instead of the default NMS path")
    parser.add_argument("--image-count", type=int, help="Maximum preloaded images to cycle through")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--score-thresh", type=float,
                        help="Override comparison.latency_score_threshold")
    parser.add_argument("--max-dets", "--max-det", dest="max_dets", type=int,
                        help="Override comparison.max_detections")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction,
                        help="Override comparison.channels_last (shared by every benchmark; "
                             "unset = Ultralytics auto rule)")
    parser.add_argument("--output", help="Defaults to outputs/reports/yolo26_latency.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    comparison = config["comparison"]
    imgsz = args.imgsz if args.imgsz is not None else int(config["image_size"])
    warmup = args.warmup if args.warmup is not None else comparison["warmup"]
    iterations = args.iterations if args.iterations is not None else comparison["iterations"]
    image_count = args.image_count if args.image_count is not None else comparison["image_count"]
    score_threshold = (args.score_thresh if args.score_thresh is not None
                       else comparison["latency_score_threshold"])
    max_detections = args.max_dets if args.max_dets is not None else comparison["max_detections"]
    channels_last = (args.channels_last if args.channels_last is not None
                     else comparison.get("channels_last"))
    weights = args.weights or (comparison.get("control") or {}).get("weights") or "yolo26n.pt"
    if warmup < 0 or iterations < 1 or image_count < 1 or max_detections < 1 or imgsz < 1:
        parser.error("--warmup must be >= 0; --iterations, --image-count, --max-dets and --imgsz "
                     "must be >= 1")

    # COCO images vary in shape, so cuDNN autotuning would re-tune per shape inside the timings.
    torch.backends.cudnn.benchmark = False
    try:
        torch_device = resolve_device(str(args.device or config.get("device", "cuda")),
                                      "benchmarking")
    except (RuntimeError, ValueError) as exc:
        parser.error(f"--device: {exc}")
    yolo_device, device = ultralytics_device(torch_device), str(torch_device)
    paths, images = load_images(args.source or config["data"]["val_images"], image_count)
    # select_device() lowers torch threads to cpu_count - 1 on the first CPU predict; keep the
    # process default so YOLO runs with the same thread count as the R-CNN scripts.
    threads = torch.get_num_threads()
    model = YOLO(weights)

    predict_kwargs = {"imgsz": imgsz, "device": yolo_device, "conf": score_threshold,
                      "max_det": max_detections, "verbose": False}
    # Recent 8.4.x releases replaced half/end2end with quantize/nms; passing the old keys warns.
    cfg_keys = DEFAULT_CFG_DICT.keys()
    # Without an explicit value Ultralytics turns channels_last on by itself on x86 CPUs.
    if "channels_last" in cfg_keys and channels_last is not None:
        predict_kwargs["channels_last"] = channels_last
    if args.half:
        predict_kwargs.update({"half": True} if "half" in cfg_keys else {"quantize": 16})
    if args.nms_free:
        predict_kwargs.update({"end2end": True} if "end2end" in cfg_keys else {"nms": False})

    # The first call also builds the predictor and fuses the model; that is the cold start.
    cold_start_ms, _ = timed_call(model, images[0], predict_kwargs, device)
    torch.set_num_threads(threads)
    for index in range(warmup):
        timed_call(model, images[index % len(images)], predict_kwargs, device)
    samples, detections = [], []
    for index in range(iterations):
        elapsed, count = timed_call(model, images[index % len(images)], predict_kwargs, device)
        samples.append(elapsed)
        detections.append(count)

    backend = getattr(model.predictor, "model", None)
    fp16 = bool(getattr(backend, "fp16", args.half))
    if args.half and not fp16:
        print("Ultralytics ignored --half on this device; recording fp32.", file=sys.stderr)
    layout = graph_layout(predictor_graph(model))
    if channels_last is not None and layout["memory_format"] not in (
            None, "channels_last" if channels_last else "contiguous"):
        print(f"Ultralytics ran {layout['memory_format']} although channels_last={channels_last}; "
              "recording what ran.", file=sys.stderr)
    weights_file = getattr(model, "ckpt_path", None) or weights
    has_one2one = getattr(detection_head(model), "one2one_cv2", None) is not None

    report = build_report(
        "latency",
        "yolo26",
        weights=display_path(weights_file) if Path(weights_file).is_file() else str(weights),
        weights_sha256=file_sha256(weights_file),
        image_size=imgsz,
        batch_size=1,
        precision="fp16" if fp16 else "fp32",
        environment=collect_environment(device),
        latency={
            "scope_id": SCOPE_HOST_TO_DETECTIONS,
            "timed_scope": SCOPE_LABELS[SCOPE_HOST_TO_DETECTIONS],
            "inference_mode": inference_mode(model),
            "nms_free_head_available": has_one2one,
            **layout,
            "warmup": warmup,
            "iterations": iterations,
            "image_count": len(paths),
            "score_threshold": score_threshold,
            "max_detections": max_detections,
            "images": [display_path(path) for path in paths[:20]],
            "mean_detections": sum(detections) / len(detections),
            **latency_summary(samples, cold_start_ms=cold_start_ms),
            "cold_start_scope": COLD_START_FIRST_PREDICT,
        },
        model_stats=model_stats(model, imgsz),
    )
    output = write_report(report, args.output or default_report_path("latency", "yolo26"))
    hidden = {"samples_ms", "images", "timed_scope"}
    print(json.dumps({
        "output": str(output),
        "model": report["model"],
        "weights": report["weights"],
        "device": device,
        "image_size": report["image_size"],
        "precision": report["precision"],
        "latency": {key: value for key, value in report["latency"].items() if key not in hidden},
        "model_stats": report["model_stats"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
