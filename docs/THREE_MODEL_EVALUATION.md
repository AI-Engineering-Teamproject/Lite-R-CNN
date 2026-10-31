# Three-model validation evaluation

Run from the repository root after COCO val2017 images and annotations are in
`data/coco/`, and after v2/v3 checkpoints exist in their default output folders:

```powershell
python scripts/evaluate_three_models.py --device cpu
```

The command compares torchvision COCO-pretrained Faster R-CNN with the saved
Lite R-CNN v2 and v3 checkpoints. It samples 1,000 images once with seed 42 and
uses that exact image ID list for all three models. The default image size is
640, batch size is one, and latency is measured after warmup. CPU is the
default; pass `--device cuda` only when GPU evaluation is intended.

The run saves `REPORT.md`, `summary.csv`, and `evaluation.json` in a timestamped
folder under `outputs/evaluation_runs/`. It includes COCO AP/mAP, AP50, AP75,
AP_S/M/L, AR@100, score-threshold precision/recall, parameters, model size,
checkpoint size, FLOPs, peak CUDA allocation, end-to-end latency/FPS, speedup
and AP change against Faster R-CNN, backbone/BiFPN/RPN/RoI head timing, RPN
proposal recall at 100/300, and latency with 100/300 proposals.

To select other checkpoints or tune timing repetitions:

```powershell
python scripts/evaluate_three_models.py --device cpu `
  --v2-weights outputs/coco_4090_v2/best.pth `
  --v3-weights outputs/coco_4090_v3/best.pth `
  --images 1000 --seed 42 --warmup 10 --iterations 100
```

Precision and recall count class-aware detections with score at least 0.5 and
IoU at least 0.5. AP is COCO AP@[.50:.95], so it is the same quantity as the
reported mAP. FLOPs use PyTorch profiler counts for supported convolution and
matrix multiplication operators; torchvision detection operations such as
RoIAlign and NMS may be omitted. The report marks the value as a supported-op
estimate. VRAM is reported as peak allocated CUDA memory; on CPU it is null.
Faster R-CNN has no BiFPN module, so its BiFPN timing is null.
