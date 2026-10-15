"""Export the Light R-CNN Faster R-CNN path at fixed batch=1 and image size."""

import argparse
from pathlib import Path

import torch
from torch import nn

from light_rcnn.model import LightRCNN
from light_rcnn.model.detector import light_rcnn_from_config


class DetectionExport(nn.Module):
    def __init__(self, model: LightRCNN):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        detections = self.model([images[0]])[0]
        return detections["boxes"], detections["scores"], detections["labels"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="outputs/light_rcnn.onnx")
    parser.add_argument(
        "--size", type=int, help="Defaults to saved input size, or 640 for raw weights"
    )
    args = parser.parse_args()

    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    size = args.size or state.get("config", {}).get("image_size", 640)
    if "config" in state:
        model = light_rcnn_from_config(dict(state["config"], image_size=size)).eval()
    else:
        model = LightRCNN(pretrained_backbone=False, min_size=size, max_size=size).eval()
    model.load_state_dict(state.get("model", state))
    wrapper = DetectionExport(model)
    sample = torch.zeros(1, 3, size, size)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample,
        args.output,
        opset_version=17,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        do_constant_folding=True,
    )
    print(
        f"Exported fixed-shape Faster R-CNN graph to {args.output}; validate NMS and RoIAlign support in TensorRT."
    )


if __name__ == "__main__":
    main()
