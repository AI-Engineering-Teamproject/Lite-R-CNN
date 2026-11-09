# Lite R-CNN v3 Documentation — Progress and Handoff

Updated: 2026-11-08 (Asia/Seoul)

## Status

Complete. The work documents the current Lite R-CNN v3 implementation; no model implementation, training, configuration, or checkpoint files were modified.

## Deliverables

- [`LITE_RCNN_V3_FULL.md`](LITE_RCNN_V3_FULL.md) — source-linked implementation and experiment guide, including input/data transforms, inference stages, thresholds, training recipe, distillation losses, run metrics, limitations, and the locally scoped MobileNetV3 feature-stack table.
- [`LITE_RCNN_V3_QUICK.md`](LITE_RCNN_V3_QUICK.md) — concise overview and main results.
- [`lite_rcnn_v3_architecture.png`](lite_rcnn_v3_architecture.png) and [`lite_rcnn_v3_architecture.svg`](lite_rcnn_v3_architecture.svg) — English-only, code-derived architecture figure; SVG is editable.
- [`ENVIRONMENT_SNAPSHOT.md`](ENVIRONMENT_SNAPSHOT.md) — selected local package versions and the separately recorded/unknown RTX 4090 runtime fields.
- [`TASKS.md`](TASKS.md), [`REVIEW_HARNESS.md`](REVIEW_HARNESS.md), [`REVIEW_LOG.md`](REVIEW_LOG.md), [`validate_docs.py`](validate_docs.py) — acceptance criteria, source map, review record, and validation script.

## Review and validation

- Three full source-to-document review loops completed. The final fact audit found no material remaining factual errors.
- Two readability reviewers discussed the quick guide; their clarifications were applied.
- Three diagram reviewers checked source correspondence, flow, labels, overlap, and reduced/full-size readability. Final checks found no clipping or meaning-changing crossings. Place the figure at full page/column width in print; small annotations shrink when the figure is reduced.
- The validation harness passed: config/run values, distillation settings, documented GPU metrics, 80 local Markdown links, English-only SVG labels, and a 3661×2084 PNG.
- Read-only Git status for the model, distillation, training, and v3 config files was empty.

## Facts to preserve in later edits

- Feature distillation compares the student's post-Lite-FPN P2–P5 tensors with teacher FPN tensors; it does not compare raw MobileNet C2–C5.
- The teacher is frozen and absent at inference; its RPN is skipped, and teacher RoI logits use student-sampled training proposals.
- The training report's best epoch 42 AP (validation batch 4) and the standardized batch-1 GPU evaluation on 5,000 COCO val2017 images are separate measurements.
- The RTX 4090 report records PyTorch 2.7.1+cu128 and CUDA 12.8, but not Python or TorchVision versions. Do not substitute this machine's CPU-only `.venv312` versions.
- The training report marks code revision `9465b59` with a dirty worktree; the current source is documented alongside, but byte-for-byte identity with every training-time source file is not proven by the archived report.
- No statistical p-value was calculated, and no matched v3 training run without distillation is recorded.

## Commands

From the repository root on Windows:

```powershell
.\.venv312\Scripts\python.exe docs\lite_rcnn_v3\generate_architecture.py
.\.venv312\Scripts\python.exe docs\lite_rcnn_v3\validate_docs.py
```

For CUDA training, activate a CUDA-capable environment first; the local `.venv312` is CPU-only. The exact command and training caveat are in the full guide.
