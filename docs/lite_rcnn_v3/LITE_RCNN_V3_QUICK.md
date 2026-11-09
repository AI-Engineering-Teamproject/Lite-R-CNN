# Lite R-CNN v3: Quick Guide

**Flow:** Image → MobileNetV3-Large → Lite FPN → anchor RPN → RoIAlign and compact box head → detections.

| Component | Implemented setting |
|---|---|
| Backbone and neck | C2-C5 MobileNetV3-Large; project Lite FPN produces 64-channel P2-P5; max-pooled P6-P7 feed RPN |
| RPN | Anchor sizes 8/16/32/64/128/256 px, ratios 0.5/1/2; up to 300 post-NMS proposals at inference |
| RoI head | RoIAlign 7×7; Conv 64→32; FC 1568→256; 91 logits for background + COCO category IDs (80 foreground classes; IDs have gaps) |
| Final detections | Score threshold 0.05; class-wise NMS IoU 0.5; maximum 100 boxes/image |

**Distillation:** During training, frozen COCO Faster R-CNN ResNet50-FPN supervises the student's post-FPN P2-P5 features (Smooth L1, weight 0.5) and its class logits on the student's proposals (KL, temperature 2, weight 1). The teacher is absent at inference.

**Recorded results:** The 48-epoch RTX 4090 run selected epoch 42 at AP@[.50:.95] 26.80% with validation batch 4. A separate standardized COCO val2017 GPU evaluation on 5,000 images at batch 1 measured AP@[.50:.95] 26.11% and median latency (p50) 5.91 ms. Its report records Torch 2.7.1+cu128/CUDA 12.8, but not Python or TorchVision versions. The separate local `.venv312` is Python 3.12.14 with CPU-only Torch 2.14.1 and TorchVision 0.29.1.

**Thresholds and limits:** Detector cutoff is 0.05; `predict.py` displays scores ≥0.25 by default; custom Precision/Recall use score ≥0.5 and IoU ≥0.5. COCO AP averages IoU 0.50-0.95. No statistical p-value was calculated, and no matched v3 run without distillation is recorded.

See the [full implementation record](LITE_RCNN_V3_FULL.md) and its [architecture figure](lite_rcnn_v3_architecture.png) for detailed layers, training setup, evidence links, and interpretation notes.
