# Project roadmap

## Current scaffold

- MobileNetV3-Large features at strides 8/16/32.
- Repeated normalized bidirectional BiFPN fusion with depthwise separable convolutions.
- Standard Faster R-CNN anchor-based RPN and two-stage RoI heads.
- MobileNetV3 + BiFPN student backbone, smaller RPN proposal budget, 10-channel projection before multi-scale RoIAlign, and GAP RoI head.
- COCO dataset adapter and AP evaluator, distillation loss primitives, fixed-shape ONNX export entry point, and PyTorch timing scripts.

## Comparison roles

- **Vanilla baseline:** torchvision Faster R-CNN ResNet-50-FPN with COCO weights.
- **Candidate under development:** Light R-CNN, an optimized Faster R-CNN that retains the anchor-based RPN and RoI head but uses a MobileNetV3 + BiFPN backbone, fewer test proposals and a compressed RoI head.
- **Real-time reference:** Ultralytics YOLO26 nano (`yolo26n.pt`) at 640 input resolution by default. YOLO26 has multiple sizes and supports optional end-to-end NMS-free inference, so report the exact variant and inference mode.
- Compare on identical COCO val2017 images, 640-pixel maximum dimension, batch=1, T4 and FP32 for the initial framework baseline. Framework resize/padding details differ, so record them alongside the metrics. The starter scripts preload images and include model transforms and postprocessing in timed calls while excluding file decode. Report AP@[.50:.95], AP50, AP75, parameters, FLOPs, p50/p95 latency, and NMS time. Report end-to-end latency separately from model-only latency.

## Milestones

1. **Baseline and protocol:** compare vanilla Faster R-CNN ResNet-50-FPN, Light R-CNN and YOLO26n. Freeze checkpoint hashes, input resolution, COCO split, batch size and T4 software versions. Profile preprocessing, backbone, RPN, RoIAlign, head and NMS separately.
2. **Trainable detector:** use torchvision's Faster R-CNN RPN and sampled RoI losses. A COCO AP evaluator is included; add checkpoint resume and augmentation.
3. **Distillation:** use vanilla Faster R-CNN as teacher; align pyramid features and sampled RoIs; combine supervised loss, feature loss and temperature-scaled RoI KL loss.
4. **Accuracy/compute tuning:** compare AP, AP50, AP75, per-size AP, parameters and FLOPs to teacher. Tune image size, BiFPN width/repeats and proposal budget.
5. **Deployment:** export tensor-only graph, add TensorRT plugin(s) for proposal decode/RoIAlign/NMS, validate FP16 then calibrate INT8 if accuracy allows.
6. **Acceptance:** record warm and cold latency, p50/p95 end-to-end latency, NMS time, batch=1 T4 precision, model hash, and COCO val2017 AP. Claim target success only from these recorded measurements.

## Known gaps

This starter is an architecture and training scaffold, not a trained detector. The trainer currently saves the last epoch and reports training loss; it does not resume runs or invoke validation automatically. The separate COCO evaluator reports AP for vanilla Faster R-CNN and Light R-CNN. Distillation primitives are present, while vanilla Faster R-CNN teacher inference and proposal/logit alignment still need to be connected in a dedicated training path. The student retains standard RPN proposal generation and two-stage RoI refinement. TensorRT support for RoIAlign and NMS must be validated on the target version. The initial configuration uses 640-pixel images; expected AP and latency require measurement and may require tuning.
