# Review Log

This log records reviewer findings and resolutions. It is part of the handoff record; unresolved issues remain visible until resolved.

## Initial source audit

- Reviewer: `v3_fact_audit` (independent code/artifact inventory; no file edits).
- Result: verified backbone taps and dimensions, FPN/RPN/RoI settings, thresholds, training recipe, environments, and separate evaluation contexts.
- Correction made during root trace: the agent initially described the student KD maps as raw C2-C5. Direct tracing of `distillation_feature_names`, `MobileFPNBackboneV3.forward`, and `DistillationTrainer.forward` shows that KD selects the **post-Lite-FPN P2-P5 outputs**. The full guide and diagram use the code-derived P2-P5 mapping. This item must be rechecked in both full review loops.
- Environment caveat retained: RTX run Python/TorchVision versions are unrecorded; local `.venv312` is a separate CPU environment.
- Result-context caveat retained: training best epoch 42 at batch 4 and standardized batch-1 GPU AP on 5,000 images are different measurements.

## Quick-guide readability discussion

- Reviewers: `quick_readability_a`, `quick_readability_b`.
- Method: independent pass followed by agent-to-agent discussion; no edits by reviewers.
- Joint findings: the first draft was too dense and repeated the proposal count. The quick guide was revised to use one pipeline line, a compact settings table, one KD paragraph, and concise results/threshold caveats. Per-stage details and the full environment inventory remain in the full guide.
- Status: incorporated; must recheck in both full review loops.

## Visual review pass 1

- `diagram_code_review`, `diagram_flow_review`, and `diagram_layout_review`: completed independent reviews, then re-reviewed after the first redesign.
- Initial root inspection and reviewers found crowded labels, overlapping routes, the P6/P7 note crossed by a route, and KD arrows entering the inference area. The first redesign made feature/proposal routes orthogonal and split the inference and training lanes; labels moved outside feature cuboids.
- Re-review findings: RPN top-k needed an explicit per-level scope; final detection uses strict `score > 0.05`; the P2-P5 RoI route looked as though it began at P7. The diagram now says `pre-NMS top-k <= 1000 / level`, `post-NMS <= 300 / image`, and `score > 0.05`, and a bracket shows the P2-P5-only RoI tap.
- Final visual pass: all three reviewers confirmed the bracket separates P2-P5 from RPN-only P6/P7, labels are English-only, and no clipping/overlap remains. Small annotations need full-width placement in print.

## Full review loop 1

Completed by `v3_fact_audit` with root source reread. Restarted from v3 model/config/distillation/training source, run report/history, evaluation JSON, both guides, generator, and rendered PNG. The reviewer confirmed the P2-P5 KD mapping; best-epoch vs batch-1 full-evaluation contexts; AP/P/R values; separate score thresholds; RPN proposal-limit scopes; and no unsupported p-value claim.

- Corrections applied: clarified that the 640 transform preserves aspect ratio and resizes the long edge to approximately 640; called P6/P7 max-pooled; clarified configured LR milestones 32/42 versus recorded LR changes first appearing in epochs 33/43; documented loader filtering/conversion and the exact KD loss alignment/temperature-squared multiplier; added a captured local environment snapshot.
- The reviewer found no material false claim. It confirmed the RTX 4090 Python/TorchVision versions remain unrecorded.
- Diagram reviewers' exactness findings above were applied before the final render.
- Validation after corrections: `validate_docs.py` passed, including 51 local links and 3661×2084 PNG resolution.

## Full review loop 2

Completed by `v3_fact_audit`, `quick_readability_a/b`, and all three diagram reviewers, each rereading current artifacts independently. The fact audit confirmed the architecture, KD, AP/P/R, evaluation context, threshold scopes, training limits, local environment probe, and current rendered image against source/config/artifacts.

- Readability reviewers recommended six compact clarifications: enumerate the six discrete anchor scales; explain 91 output logits as background plus original COCO category IDs with 80 foreground classes and gaps; distinguish unrecorded RTX Python/TorchVision versions from local versions; identify COCO val2017 in the 5,000-image evaluation; expand p50 as median latency; state separately that no p-value was calculated and no matched no-KD v3 run exists. These were applied in `LITE_RCNN_V3_QUICK.md`.
- The source audit found that the local environment file lists selected packages, not the entire package inventory; the guide was changed to say “selected local package versions.” It also flagged a CUDA training command tied to this machine's CPU-only `.venv312`; the guide now says to activate a CUDA-capable environment and uses `python` without the local interpreter path.
- The source audit noted that MobileNetV3 internal bottleneck stages were collapsed into one diagram block. The full guide now includes a 17-index local TorchVision 0.29.1 feature-stack table, explicitly scoped as the local dependency because the RTX run's TorchVision version was not recorded. The diagram remains a readable end-to-end graph and the guide explains this abstraction.
- Diagram reviewers found no remaining crossings/clipping; the P2-P5 bracket, strict score threshold, RPN scopes, and training-only paths remain clear. Full-width placement is recommended for printing.
- Harness after these fixes: PASS, including 56 local links across the two guides and 3661×2084 PNG resolution.

## Final post-fix source loop 3

Completed by `v3_fact_audit`, with a final code-correspondence pass by `diagram_code_review`. Both reopened the current files after Loop 2 corrections. The fact reviewer also re-probed `.venv312`, checked all 17 MobileNetV3 feature entries against installed TorchVision `0.29.1+cpu`, and rechecked the latest SVG/PNG. No material factual or diagram-flow issues remain. The review confirms that the MobileNet table is scoped to the local TorchVision build and is not claimed as the unrecorded RTX run version.

- Final output keeps the diagram readable as an end-to-end model view; the FULL guide supplies the dependency-defined MobileNetV3 feature stack table.
- The final harness run passed after all changes: 80 local links across all Markdown files, config/run values, distillation losses, documented GPU metrics, English-only SVG labels, and 3661×2084 PNG resolution.
- Read-only Git status for `src/light_rcnn/model/v3.py`, `detector.py`, `layers.py`, `distillation.py`, `scripts/train.py`, and `configs/coco_v3.yaml` was empty; the model implementation, training code, and v3 config were not modified.

## Harness result

Latest run: PASS — config/run values, distillation losses, documented metrics, 80 local links across all Markdown files, SVG language check, and 3661×2084 PNG resolution.
