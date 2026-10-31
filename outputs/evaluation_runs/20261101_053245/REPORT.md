# Three-model detection evaluation

- Images: 5000 shared validation images (seed 42)
- Device: cuda; image size: 640; batch size: 1

| Model | AP/mAP | AP50 | AP75 | AP_S | AP_M | AP_L | Precision | Recall | p50 ms | FPS | Params M | FLOPs G* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.3237 | 0.5201 | 0.3438 | 0.1271 | 0.3493 | 0.4977 | 0.5816 | 0.5534 | 7.11 | 140.56 | 41.755 | 136.385 |
| Lite R-CNN v2 | 0.2817 | 0.4677 | 0.2888 | 0.1023 | 0.3090 | 0.4366 | 0.6721 | 0.4595 | 5.34 | 187.20 | 4.188 | 4.477 |
| Lite R-CNN v3 | 0.2611 | 0.4715 | 0.2629 | 0.1192 | 0.2809 | 0.3737 | 0.6687 | 0.4617 | 5.91 | 169.13 | 3.623 | 4.964 |

* FLOPs sums operators counted by PyTorch profiler; ROIAlign, NMS and other unsupported ops can be omitted.
Precision/Recall use score >= 0.5 and class-aware IoU >= 0.5. AP is COCO AP@[.50:.95] and equals mAP_50_95.
Faster R-CNN has no BiFPN module; its BiFPN latency is null. VRAM is peak allocated CUDA memory.
See evaluation.json and summary.csv for stage times, proposal recall, AR@100, speedup and AP deltas.
