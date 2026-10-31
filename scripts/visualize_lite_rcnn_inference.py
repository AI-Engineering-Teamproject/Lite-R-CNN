"""Run the same image through Lite R-CNN v2/v3 and save inference-stage figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

from light_rcnn.model.detector import light_rcnn_from_config

ROOT = Path(__file__).resolve().parents[1]


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def capture_model(model: torch.nn.Module, image: torch.Tensor) -> tuple[dict, dict]:
    captured: dict[str, Any] = {}
    handles = []

    def save_output(key: str):
        def hook(_module, _inputs, output):
            captured[key] = cpu_copy(output)
        return hook

    def save_transform(_module, _inputs, output):
        image_list = output[0]
        captured["transformed"] = image_list.tensors[0].detach().cpu()
        captured["transformed_size"] = tuple(int(v) for v in image_list.image_sizes[0])

    handles.append(model.transform.register_forward_hook(save_transform))
    backbone = model.backbone
    if hasattr(backbone, "body"):
        handles.append(backbone.body.register_forward_hook(save_output("body")))
    if hasattr(backbone, "neck"):
        handles.append(backbone.neck.register_forward_hook(save_output("neck")))
    handles.append(backbone.register_forward_hook(save_output("backbone")))
    handles.append(model.rpn.head.register_forward_hook(save_output("rpn_head")))
    handles.append(model.rpn.register_forward_hook(save_output("rpn")))
    handles.append(model.roi_heads.box_roi_pool.register_forward_hook(save_output("roi_pool")))

    def save_roi_boxes(_module, inputs):
        captured["roi_boxes"] = cpu_copy(inputs[1][0])

    handles.append(model.roi_heads.box_roi_pool.register_forward_pre_hook(save_roi_boxes))
    handles.append(model.roi_heads.box_head.register_forward_hook(save_output("roi_head")))
    handles.append(model.roi_heads.box_predictor.register_forward_hook(save_output("predictor")))
    try:
        with torch.inference_mode():
            prediction = model([image])[0]
    finally:
        for handle in handles:
            handle.remove()
    return captured, cpu_copy(prediction)


def normalized_rgb(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    return (tensor * std_tensor + mean_tensor).clamp(0, 1)


def save_figure(fig, stem: Path) -> None:
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=230, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def activation_map(feature: torch.Tensor, valid_size: tuple[int, int] | None = None) -> torch.Tensor:
    value = feature.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3:
        value = value.abs().mean(dim=0)
    if valid_size is not None:
        h, w = valid_size
        value = value[:max(1, h), :max(1, w)]
    return value


def plot_feature_overlay(image: Image.Image, feature: torch.Tensor, stride: int,
                         title: str, path: Path, resized_size: tuple[int, int]) -> None:
    import matplotlib.pyplot as plt

    feature_map = activation_map(feature)
    resized_h, resized_w = resized_size
    valid_h = min(feature_map.shape[0], math.ceil(resized_h / stride))
    valid_w = min(feature_map.shape[1], math.ceil(resized_w / stride))
    feature_map = feature_map[:valid_h, :valid_w]
    feature_map = F.interpolate(feature_map[None, None], size=(image.height, image.width),
                                mode="bilinear", align_corners=False)[0, 0].numpy()
    vmax = max(float(torch.tensor(feature_map).quantile(0.99)), 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[1].imshow(image)
    axes[1].imshow(feature_map, cmap="magma", alpha=0.58, vmin=0, vmax=vmax)
    axes[1].set_title(f"{title}\nmean absolute activation, stride {stride}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    save_figure(fig, path)


def draw_boxes(image: Image.Image, boxes: torch.Tensor, *, labels=None, scores=None,
               limit: int = 50, color: str = "#00d084", title: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    for index, box in enumerate(boxes[:limit].tolist()):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=1.05 if labels is None else 1.7))
        if labels is not None:
            caption = labels[index]
            if scores is not None:
                caption += f" {float(scores[index]):.2f}"
            ax.text(x1, max(0, y1 - 4), caption, color="white", fontsize=8,
                    bbox={"facecolor": color, "alpha": 0.8, "pad": 1.5, "edgecolor": "none"})
    fig.tight_layout()
    return fig


def resized_boxes_to_original(boxes: torch.Tensor, original_size: tuple[int, int],
                              resized_size: tuple[int, int]) -> torch.Tensor:
    original_w, original_h = original_size
    resized_h, resized_w = resized_size
    scale = torch.tensor([original_w / resized_w, original_h / resized_h,
                          original_w / resized_w, original_h / resized_h])
    mapped = boxes.detach().cpu() * scale
    mapped[:, 0::2].clamp_(0, original_w)
    mapped[:, 1::2].clamp_(0, original_h)
    return mapped


def map_feature_names(name: str, maps: dict) -> list[tuple[str, torch.Tensor, int]]:
    values = list(maps.values())
    if name == "Lite R-CNN v2":
        stages = ["P3", "P4", "P5"]
        strides = [8, 16, 32]
    else:
        stages = ["P2", "P3", "P4", "P5", "P6", "P7"]
        strides = [4, 8, 16, 32, 64, 128]
    return [(stage, value, strides[index]) for index, (stage, value) in enumerate(zip(stages, values))]


def save_rpn_head(image: Image.Image, outputs, out_dir: Path, prefix: str,
                  resized_size: tuple[int, int]) -> None:
    import matplotlib.pyplot as plt

    objectness_levels, deltas_levels = outputs
    level_count = min(len(objectness_levels), len(deltas_levels))
    fig, axes = plt.subplots(2, level_count, figsize=(3.6 * level_count, 6.2), squeeze=False)
    for index in range(level_count):
        obj = objectness_levels[index][0].float().sigmoid().amax(dim=0)
        delta = deltas_levels[index][0].float().abs().mean(dim=0)
        for row, (heat, caption) in enumerate(((obj, "max anchor objectness"),
                                               (delta, "mean absolute box delta"))):
            resized_h, resized_w = resized_size
            stride = 4 * (2 ** index) if level_count == 6 else 8 * (2 ** index)
            heat = heat[:math.ceil(resized_h / stride), :math.ceil(resized_w / stride)]
            heat = F.interpolate(heat[None, None], size=(image.height, image.width),
                                 mode="bilinear", align_corners=False)[0, 0].numpy()
            axes[row, index].imshow(image)
            axes[row, index].imshow(heat, cmap="viridis", alpha=0.58)
            axes[row, index].set_title(f"RPN level {index + 1}\n{caption}")
            axes[row, index].axis("off")
    fig.suptitle(f"{prefix} RPN head outputs", fontsize=15)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{prefix}_RPN_head")


def save_roi_align(image: Image.Image, pooled: torch.Tensor, roi_boxes: torch.Tensor,
                   original_size: tuple[int, int], resized_size: tuple[int, int],
                   out_dir: Path, prefix: str, count: int = 12) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = min(count, pooled.shape[0], roi_boxes.shape[0])
    if n == 0:
        return
    mapped = resized_boxes_to_original(roi_boxes[:n], original_size, resized_size)
    columns, rows = 4, math.ceil(n / 4)
    fig, axes = plt.subplots(rows, columns * 2, figsize=(16, 3.3 * rows), squeeze=False)
    for index in range(n):
        row, col = divmod(index, columns)
        crop_ax, heat_ax = axes[row, col * 2], axes[row, col * 2 + 1]
        x1, y1, x2, y2 = [int(v) for v in mapped[index].tolist()]
        crop_ax.imshow(image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))))
        crop_ax.set_title(f"Proposal {index + 1}: image crop")
        heat = activation_map(pooled[index:index + 1]).numpy()
        heat_ax.imshow(heat, cmap="magma")
        heat_ax.set_title("RoIAlign 7×7 activation")
        crop_ax.axis("off")
        heat_ax.axis("off")
    for index in range(n, rows * columns):
        row, col = divmod(index, columns)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")
    fig.suptitle(f"{prefix}: proposal image crops and RoIAlign outputs", fontsize=15)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{prefix}_RoIAlign")


def save_roi_head(features: torch.Tensor, out_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    if features.ndim != 2 or features.shape[0] == 0:
        return
    values = features[:min(32, features.shape[0])].float().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(values, cmap="coolwarm", aspect="auto", interpolation="nearest")
    ax.set_title(f"{prefix} RoI head embeddings (proposals × {features.shape[1]} dimensions)")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("RPN proposal index")
    fig.colorbar(im, ax=ax, label="feature value")
    fig.tight_layout()
    save_figure(fig, out_dir / f"{prefix}_RoIHead")


def save_classifier(logits: torch.Tensor, out_dir: Path, prefix: str, class_names: dict[int, str]) -> None:
    import matplotlib.pyplot as plt

    if logits.ndim != 2 or logits.shape[0] == 0:
        return
    probs = logits.float().softmax(dim=1)
    count = min(8, probs.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.axis("off")
            continue
        scores, labels = probs[index].topk(min(6, probs.shape[1]))
        labels = labels.tolist()
        names = [class_names.get(label, f"class_{label}") for label in labels]
        ax.barh(range(len(labels)), scores.tolist(), color="#5aa9e6")
        ax.set_yticks(range(len(labels)), names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.25, float(scores.max()) * 1.15))
        ax.set_title(f"Proposal {index + 1}")
    fig.suptitle(f"{prefix} class probabilities before final NMS", fontsize=15)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{prefix}_classifier")


def save_pipeline(image: Image.Image, feature_image: Path, rpn_image: Path,
                  detection_image: Path, out_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt

    panels = [(image, "Input"), (Image.open(feature_image.with_suffix(".png")), "Backbone / pyramid feature"),
              (Image.open(rpn_image.with_suffix(".png")), "RPN proposals"),
              (Image.open(detection_image.with_suffix(".png")), "Final detections")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{prefix}: same input through Lite R-CNN inference", fontsize=16)
    fig.tight_layout()
    save_figure(fig, out_dir / f"{prefix}_pipeline_overview")


def visualize_model(name: str, checkpoint_path: Path, config_path: Path, image: Image.Image,
                    out_dir: Path, class_names: dict[int, str], device: torch.device,
                    max_detections: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = checkpoint.get("config", config)
    model = light_rcnn_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint.get("model", checkpoint))
    prefix = "v2" if name.endswith("v2") else "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_tensor = to_tensor(image).to(device)
    captured, prediction = capture_model(model, input_tensor)
    resized_size = captured["transformed_size"]
    input_path = out_dir / f"{prefix}_input"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(f"{prefix} original RGB input | {image.width}×{image.height}")
    ax.axis("off")
    save_figure(fig, input_path)

    mean = list(model.transform.image_mean)
    std = list(model.transform.image_std)
    transformed = normalized_rgb(captured["transformed"][:, :resized_size[0], :resized_size[1]], mean, std)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(transformed.permute(1, 2, 0).numpy())
    ax.set_title(f"{prefix} transformed model input | {resized_size[1]}×{resized_size[0]}")
    ax.axis("off")
    save_figure(fig, out_dir / f"{prefix}_transform")

    if isinstance(captured.get("body"), (tuple, list)):
        body_names = ["C3", "C4", "C5"] if prefix == "v2" else ["C2", "C3", "C4", "C5"]
        body_strides = {"C2": 4, "C3": 8, "C4": 16, "C5": 32}
        for stage, value in zip(body_names, captured["body"]):
            plot_feature_overlay(image, value, body_strides[stage], f"{prefix} backbone {stage}",
                                 out_dir / f"{prefix}_backbone_{stage}", resized_size)

    pyramid = captured.get("backbone", {})
    pyramid_items = list(pyramid.items()) if isinstance(pyramid, dict) else []
    if prefix == "v2":
        pyramid_stages = ["P3", "P4", "P5"]
        pyramid_strides = [8, 16, 32]
    else:
        pyramid_stages = ["P2", "P3", "P4", "P5", "P6", "P7"]
        pyramid_strides = [4, 8, 16, 32, 64, 128]
    feature_for_overview = None
    feature_stride = 8 if prefix == "v2" else 4
    for (key, feature), stage, stride in zip(pyramid_items, pyramid_stages, pyramid_strides):
        stage_group = "BiFPN" if prefix == "v2" else "FPN"
        path = out_dir / f"{prefix}_{stage_group}_{stage}"
        plot_feature_overlay(image, feature, stride, f"{prefix} {stage_group} {stage}", path, resized_size)
        if feature_for_overview is None and stage in ("P3", "P2"):
            feature_for_overview = path

    if "rpn_head" in captured:
        save_rpn_head(image, captured["rpn_head"], out_dir, prefix, resized_size)
    proposals = captured.get("rpn", ([], {}))[0][0]
    original_proposals = resized_boxes_to_original(proposals, image.size, resized_size)
    fig = draw_boxes(image, original_proposals, limit=50, color="#ffcc00",
                     title=f"{prefix} RPN proposals (first 50 of {len(proposals)})")
    rpn_path = out_dir / f"{prefix}_RPN"
    save_figure(fig, rpn_path)

    if "roi_pool" in captured and "roi_boxes" in captured:
        save_roi_align(image, captured["roi_pool"], captured["roi_boxes"], image.size,
                       resized_size, out_dir, prefix)
    if "roi_head" in captured:
        save_roi_head(captured["roi_head"], out_dir, prefix)
    if "predictor" in captured:
        logits = captured["predictor"][0]
        save_classifier(logits, out_dir, prefix, class_names)

    detections = prediction["boxes"]
    labels = prediction["labels"].tolist()
    scores = prediction["scores"]
    captions = [class_names.get(int(label), f"class_{int(label)}") for label in labels]
    fig = draw_boxes(image, detections, labels=captions, scores=scores,
                     limit=max_detections, color="#21d4fd",
                     title=f"{prefix} final detections (score ≥ {model.roi_heads.score_thresh:.2f})")
    detection_path = out_dir / f"{prefix}_detections"
    save_figure(fig, detection_path)

    if feature_for_overview is None:
        feature_for_overview = input_path
    save_pipeline(image, feature_for_overview, rpn_path, detection_path, out_dir, prefix)
    metadata = {
        "model": name, "checkpoint": str(checkpoint_path), "input_width": image.width,
        "input_height": image.height, "transformed_width": resized_size[1],
        "transformed_height": resized_size[0], "device": str(device),
        "rpn_proposals": len(proposals), "final_detections": len(detections),
        "feature_visualization": "per-location mean absolute activation, normalized to overlay on the original image",
        "roi_visualization": "image crops paired with channel-mean absolute 7x7 RoIAlign activations",
        "files": sorted(path.name for path in out_dir.glob("*.png")),
    }
    (out_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    del model, captured
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="One RGB image passed to both models")
    parser.add_argument("--v2-weights", default="outputs/coco_4090_v2/best.pth")
    parser.add_argument("--v3-weights", default="outputs/coco_4090_v3/best.pth")
    parser.add_argument("--v2-config", default="configs/coco.yaml")
    parser.add_argument("--v3-config", default="configs/coco_v3.yaml")
    parser.add_argument("--labels-json", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--output-dir", default="outputs/lite_rcnn_inference")
    parser.add_argument("--device", default="cuda", help="Visualization is intended to run on GPU")
    parser.add_argument("--max-detections", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This visualization script is configured for GPU inference; use --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build and retry.")
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise SystemExit(f"Input image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    label_path = Path(args.labels_json)
    if not label_path.is_absolute():
        label_path = ROOT / label_path
    class_names = {}
    if label_path.is_file():
        categories = json.loads(label_path.read_text(encoding="utf-8")).get("categories", [])
        class_names = {int(item["id"]): str(item["name"]) for item in categories}
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root = output_root / image_path.stem
    jobs = [
        ("Lite R-CNN v2", ROOT / args.v2_weights, ROOT / args.v2_config, output_root / "v2_image"),
        ("Lite R-CNN v3", ROOT / args.v3_weights, ROOT / args.v3_config, output_root / "v3_image"),
    ]
    for name, checkpoint, config, _folder in jobs:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found for {name}: {checkpoint}")
        if not config.is_file():
            raise SystemExit(f"Config not found for {name}: {config}")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Shared input image: {image_path}", flush=True)
    results = []
    for name, checkpoint, config, folder in jobs:
        print(f"\n=== {name} ===", flush=True)
        results.append(visualize_model(name, checkpoint, config, image, folder,
                                       class_names, device, args.max_detections))
        print(f"Saved visualizations: {folder}", flush=True)
    (output_root / "comparison_metadata.json").write_text(
        json.dumps({"shared_image": str(image_path), "device": str(device), "models": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nPipeline comparison: {output_root}", flush=True)


if __name__ == "__main__":
    main()
