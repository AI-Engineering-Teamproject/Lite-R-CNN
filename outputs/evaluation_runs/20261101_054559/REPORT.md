# Three-model detection evaluation

- Images: 1000 shared validation images (seed 42)
- Device: cpu; image size: 640; batch size: 1

| Model | AP/mAP | AP50 | AP75 | AP_S | AP_M | AP_L | Precision | Recall | p50 ms | FPS | Params M | FLOPs G* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 0.3312 | 0.5185 | 0.3536 | 0.1472 | 0.3774 | 0.5285 | 0.5763 | 0.5465 | 1756.04 | 0.57 | 41.755 | 144.082 |
| Lite R-CNN v2 | 0.2902 | 0.4768 | 0.2994 | 0.1085 | 0.3230 | 0.4907 | 0.6667 | 0.4533 | 278.49 | 3.59 | 4.188 | 4.757 |
| Lite R-CNN v3 | 0.2724 | 0.4798 | 0.2864 | 0.1283 | 0.2903 | 0.4333 | 0.6699 | 0.4612 | 257.31 | 3.89 | 3.623 | 5.292 |

* FLOPs sums operators counted by PyTorch profiler; ROIAlign, NMS and other unsupported ops can be omitted.
Precision/Recall use score >= 0.5 and class-aware IoU >= 0.5. AP is COCO AP@[.50:.95] and equals mAP_50_95.
Faster R-CNN has no BiFPN module; its BiFPN latency is null. VRAM is peak allocated CUDA memory.
See evaluation.json and summary.csv for stage times, proposal recall, AR@100, speedup and AP deltas.
