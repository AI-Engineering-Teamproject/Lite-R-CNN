"""Export the Light R-CNN Faster R-CNN path at fixed batch=1 and image size."""

import argparse

import torch
from torch import nn

from light_rcnn.model import LightRCNN


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
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    model = LightRCNN(pretrained_backbone=False, min_size=args.size, max_size=args.size).eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))
    wrapper = DetectionExport(model)
    sample = torch.zeros(1, 3, args.size, args.size)
    torch.onnx.export(
        wrapper,
        sample,
        args.output,
        opset_version=17,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        do_constant_folding=True,
    )
    print(f"Exported fixed-shape Faster R-CNN graph to {args.output}; validate NMS and RoIAlign support in TensorRT.")


if __name__ == "__main__":
    main()
