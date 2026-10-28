# Project roadmap

## Current scaffold and candidate

- v2 uses MobileNetV3-Large features at strides 8/16/32 and repeated normalized BiFPN fusion.
- v3 candidate adds the stride-4 C2/P2 level and uses a single-pass lightweight FPN over P2-P5.
- Standard Faster R-CNN anchor-based RPN and two-stage RoI heads.
- v2 full COCO validation reached AP@[.50:.95] 28.82%. Its 10.99% AP-small motivates testing P2 in v3.
- v3 narrows the RoI projection, transfers the v2 MobileNetV3 feature extractor, and enables Faster R-CNN teacher feature/logit distillation.
- COCO dataset adapter and shared official AP evaluator, resumable training with best-AP selection, aligned teacher/student distillation, fixed-shape ONNX export entry point, and PyTorch timing scripts.

## Comparison roles

- **Vanilla baseline:** torchvision Faster R-CNN ResNet-50-FPN with COCO weights.
- **Candidate under development:** Lite R-CNN v3 retains Faster R-CNN's anchor-based RPN and two-stage RoI detector with a MobileNetV3 + four-level lightweight FPN.
- **Real-time reference:** Ultralytics YOLO26 nano (`yolo26n.pt`) at 640 input resolution by default. YOLO26 has multiple sizes and supports optional end-to-end NMS-free inference, so report the exact variant and inference mode.
- Compare on identical COCO val2017 images, 640-pixel maximum dimension, batch=1, T4 and FP32 for the initial framework baseline. Framework resize/padding details differ, so record them alongside the metrics. The starter scripts preload images and include model transforms and postprocessing in timed calls while excluding file decode. Report AP@[.50:.95], AP50, AP75, parameters, FLOPs, p50/p95 latency, and NMS time. Report end-to-end latency separately from model-only latency.

## Milestones

1. **Baseline and protocol:** compare vanilla Faster R-CNN ResNet-50-FPN, Light R-CNN and YOLO26n. Freeze checkpoint hashes, input resolution, COCO split, batch size and T4 software versions. Profile preprocessing, backbone, RPN, RoIAlign, head and NMS separately.
2. **Trainable detector:** use torchvision's Faster R-CNN RPN and sampled RoI losses. Validation, best/last checkpoints, optimizer/scheduler/scaler/RNG resume and loss history are implemented; augmentation remains future work.
3. **Distillation:** use vanilla Faster R-CNN as teacher; align pyramid features and sampled RoIs; combine supervised loss, feature loss and temperature-scaled RoI KL loss.
4. **Accuracy/compute tuning:** compare AP, AP50, AP75, per-size AP, parameters and FLOPs to teacher. Tune image size, BiFPN width/repeats and proposal budget.
5. **Deployment:** export tensor-only graph, add TensorRT plugin(s) for proposal decode/RoIAlign/NMS, validate FP16 then calibrate INT8 if accuracy allows.
6. **Acceptance:** record warm and cold latency, p50/p95 end-to-end latency, NMS time, batch=1 T4 precision, model hash, and COCO val2017 AP. Claim target success only from these recorded measurements.

## Known gaps

The v2 checkpoint and full-val results are stored in `outputs/coco_4090_v2/`. The v3 architecture and training preset are implemented, but no v3 checkpoint or full-val result exists yet. Its AP gain and the latency cost of P2 remain hypotheses until a full 48-epoch COCO run and matched batch-1 profiling are completed. Optional distillation supports both the v2 stride 8/16/32 maps and v3 stride 4/8/16/32 maps plus shared sampled RoIs. TensorRT support for RoIAlign and NMS still requires validation on the target version. See [TRAINING.md](TRAINING.md) and [the v3 experiment record](experiments/2026-10-28-v3-design.md).
