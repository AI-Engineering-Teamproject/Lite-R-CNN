# CPU inference comparison: Faster R-CNN, Lite R-CNN v2, Lite R-CNN v3

## Summary

CPU batch-1 latency was measured with PyTorch FP32 at 640 px. The benchmark used the first 16 sorted COCO val2017 images, 5 warmup inferences, and 30 timed inferences per model. Image decoding was excluded; torchvision transforms, backbone, RPN, RoI heads, and NMS were timed.

| Model | Full-val AP@[.50:.95] | AP-small | CPU p50 | CPU p95 | Estimated raw inference time for 5,000 images |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN (COCO-pretrained ResNet-50-FPN) | **32.39%** | 12.77% | 1,978.6 ms | 3,899.9 ms | about 2 h 45 min |
| Lite R-CNN v2 | 28.82% | 10.99% | **228.5 ms** | 670.6 ms | about 19 min |
| Lite R-CNN v3 | 26.80% | **12.15%** | 299.9 ms | **648.5 ms** | about 25 min |

Full-val accuracy values are existing COCO val2017 results over all 5,000 images. The accuracy column was not recomputed on CPU. CPU latency is a short local benchmark; the 5,000-image times are linear extrapolations from p50 and exclude image decoding and COCO metric aggregation.

## Interpretation

- Faster R-CNN has the highest full-val AP, but its measured CPU p50 is about 8.7 times v2 and 6.6 times v3.
- v2 is the fastest by p50 in this run. v3 p50 is about 31% higher than v2, while v3 has a slightly lower measured p95; the 30-sample tail estimates are noisy.
- These latency measurements describe this machine and this image sample. They should not be treated as universal CPU timings.

## Source records

- `faster_rcnn_latency.json`
- `lite_rcnn_v2_latency.json`
- `lite_rcnn_v3_latency.json`
- Accuracy: `../../coco_4090_v2/RESULTS.md`, `../../coco_4090_v3/RESULTS.md`, and `../20261031_144558/faster_rcnn_coco_metrics.json`
