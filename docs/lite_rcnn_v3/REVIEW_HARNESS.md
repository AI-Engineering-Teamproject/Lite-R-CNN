# Review Harness: Lite R-CNN v3

## Rule for evidence

Treat the repository source, the saved configuration, and the recorded run outputs as the evidence set. Do not infer implementation details from the original project proposal, older drafts, diagram conventions, or model names. Label every unrecorded value as **not recorded**. If code and an old report disagree, document the code path and identify the report as stale.

## Source map

| Claim family | Primary evidence |
|---|---|
| v3 architecture, C2-C5 taps, FPN, P6/P7, RPN and RoI wiring | [`src/light_rcnn/model/v3.py`](../../src/light_rcnn/model/v3.py) |
| RoI projection and Faster R-CNN baseline construction | [`src/light_rcnn/model/detector.py`](../../src/light_rcnn/model/detector.py) |
| Depthwise separable convolution implementation | [`src/light_rcnn/model/layers.py`](../../src/light_rcnn/model/layers.py) |
| Training and inference hyperparameters | [`configs/coco_v3.yaml`](../../configs/coco_v3.yaml) |
| Teacher, feature/logit losses and gradient boundaries | [`src/light_rcnn/distillation.py`](../../src/light_rcnn/distillation.py) |
| Optimizer, checkpoint initialization, loss summation and checkpointing | [`scripts/train.py`](../../scripts/train.py) |
| Input/target loading | [`src/light_rcnn/data.py`](../../src/light_rcnn/data.py) |
| COCO AP and custom Precision/Recall calculation | [`src/light_rcnn/evaluation.py`](../../src/light_rcnn/evaluation.py), [`scripts/evaluate_three_models_gpu_full.py`](../../scripts/evaluate_three_models_gpu_full.py) |
| Training-run facts and epoch-best result | [`outputs/coco_4090_v3/RESULTS.md`](../../outputs/coco_4090_v3/RESULTS.md), [`history.jsonl`](../../outputs/coco_4090_v3/history.jsonl), [`config.yaml`](../../outputs/coco_4090_v3/config.yaml) |
| Standardized GPU and CPU comparisons | [`GPU evaluation.json`](../../outputs/evaluation_runs/20261101_053245/evaluation.json), [`CPU evaluation.json`](../../outputs/evaluation_runs/20261101_054559/evaluation.json) |
| Python dependency floor and optional groups | [`pyproject.toml`](../../pyproject.toml) |
| Local venv state | `.venv312` and `.venv` interpreter/package probes; these are local snapshots, not the RTX 4090 run environment |

## Agent tasks

### Documentation review agents

1. **Fact auditor:** trace every architecture, threshold, environment, training, and metric statement to the source map. Report missing evidence and incorrect mappings. In particular, distinguish the raw MobileNet C2-C5 taps from the post-neck P2-P5 tensors used by feature distillation.
2. **Readability reviewer A:** identify duplicated explanations, jargon, long sections, and details that can be removed from the quick guide.
3. **Readability reviewer B:** independently propose a compact outline, then discuss A's recommendations and produce a jointly prioritized cut/rewrite list. Preserve technical distinctions that change the meaning.

### Diagram review agents

1. **Code correspondence:** check every box, layer, tensor dimension, branch, threshold, and count against source.
2. **Flow and semantics:** follow the arrows from input through RPN, proposals, RoI path, and outputs; verify train-only teacher paths cannot be mistaken for inference paths.
3. **Visual inspection:** inspect the rendered PNG at full size and reduced size for readable labels, clipping, overlap, arrow crossings, and English-only text.

## Two complete review loops

For each loop, reviewers restart from the source map rather than only checking the previous review notes.

1. Read the implementation/configuration/run artifacts anew.
2. Audit the full guide and quick guide against those sources.
3. Inspect the SVG source and rendered PNG independently.
4. Record each issue with evidence, resolution, or an explicit unknown.
5. Re-render the diagram after changes and rerun `validate_docs.py`.

Loop 1 catches initial factual and layout issues. Loop 2 verifies the corrected artifacts from the beginning and checks that fixes did not introduce new errors. Log reviewer names/roles, findings, changes, and validation result in `REVIEW_LOG.md`.

## Local commands

Run from the repository root on Windows. The first command regenerates only the documentation image assets; the second validates saved config/run values, document metrics, local links, diagram labels, and SVG/PNG files.

```powershell
.\.venv312\Scripts\python.exe docs\lite_rcnn_v3\generate_architecture.py
.\.venv312\Scripts\python.exe docs\lite_rcnn_v3\validate_docs.py
```

The generator requires Matplotlib. The validation harness requires PyYAML and Pillow. Neither command modifies model implementation files or checkpoints.

## Non-negotiable distinctions

- v3 feature distillation compares its **post-Lite-FPN P2-P5 outputs** with teacher FPN feature maps; it does not compare the raw C2-C5 tensors.
- The teacher is frozen and train-only. The teacher RPN is not executed; teacher RoI logits use the student's sampled training proposals.
- v3 run-best checkpoint AP (validation batch size 4) and standardized batch-size-1 AP are separate measurements and must keep their contexts.
- Model output score cutoff (`0.05`), `predict.py` drawing cutoff (default `0.25`), and custom P/R score cutoff (`0.5`) are different settings. COCO AP is not measured at a single fixed score cutoff.
- RTX run Python version and training TorchVision version are not present in the saved run report; do not copy the local `.venv312` versions into the training-environment field.
- A p-value was not calculated. Do not call AP a p-value.
