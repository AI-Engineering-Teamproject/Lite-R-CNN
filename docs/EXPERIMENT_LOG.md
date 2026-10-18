# Lite R-CNN 실험 기록

## 2026-10-19 — COCO 객체 크기 분포와 validation 처리량

로컬 COCO 2017 train annotation 849,947개의 non-crowd 유효 box를 입력 최대 변 640 기준으로 정규화해 `sqrt(width × height)`를 요약했습니다. 백분위수는 10%=14.4px, 25%=26.8px, 50%=58.6px, 75%=133.3px, 90%=257.2px, 95%=347.4px, 99%=505.1px였습니다. 현재 RPN anchor base sizes 32–256px는 중앙 분포에는 맞지만 작은 25% 구간과 큰 10% 구간에는 확장 여지가 있습니다. 후속 후보로 P3에 16px, P5에 512px anchor를 추가하는 ablation을 제안합니다. 이 통계는 anchor/proposal recall이나 AP가 아니므로 기본 anchor를 아직 바꾸지 않았고, GPU 학습 후 full-val AP와 proposal recall로 판단해야 합니다.

매 epoch COCO validation의 작은 batch 호출 오버헤드를 줄일 수 있도록 평가 batch size를 설정화했습니다. CUDA 기본 설정은 batch 4, CPU 설정은 batch 1이며 결과 JSON/history에 `evaluation_batch_size`를 기록합니다. 배치 추론의 실제 시간 개선과 AP 차이는 아직 측정하지 않았습니다.

## RTX 4090 학습 인계 상태 (2026-10-19)

- 구현·학습 파이프라인과 설정, COCO 준비 스크립트, 이전 소규모 진단 결과를 현재 상태로 저장했습니다.
- 전체 COCO 학습은 아직 실행하지 않았고, 새 spatial RoI head의 전체 validation AP 성과도 없습니다. GPU 환경에서 새 run으로 시작해야 합니다.
- 권장 시작 명령: `python scripts/train.py --config configs/coco.yaml --device cuda --epochs 36 --output-dir outputs/coco_4090`
- 재개 명령: `python scripts/train.py --resume outputs/coco_4090/last.pth --epochs 36`
- 현재 CUDA 미사용 CPU 기준 36 epoch 시간은 대략 70일 이상으로 추정했으며, RTX 4090의 실제 시간은 아직 측정하지 않았습니다.
- COCO 데이터, 가상환경, 체크포인트와 실행 산출물은 Git에 넣지 않습니다. GPU 환경에 데이터 준비 스크립트로 데이터셋을 내려받고, 학습 후 `best.pth`, `history.jsonl`, config를 별도 결과물로 보존합니다.

이 문서는 Lite R-CNN 설계와 학습 실험만 기록합니다. Faster R-CNN 및 YOLOv26 공통 벤치마크는 팀원의 결과와 별도로 관리합니다.

## 2026-10-18 — RoI head 초기 ablation

**질문:** 7×7 RoI feature를 10차원 GAP로 바로 줄이는 현재 방식이 학습 표현력을 제한하는가?

**비교 설정:** CPU, 같은 COCO train/val 앞 16장, seed 42, image size 640, batch size 1, 1 epoch, learning rate `2e-4`, subset에 맞춘 3-step warmup. 데이터 전체는 COCO train 118,287장, validation 5,000장이지만 이 실험에서는 각 16장만 사용했습니다.

| RoI head | Student parameters | Train loss | COCO AP@[.50:.95] | 평가 범위 |
|---|---:|---:|---:|---|
| GAP 1×1, FC 없음, 10차원 출력 | 3,255,367 | 5.6678 | 0.000% | validation subset 16장 |
| Adaptive pool 3×3 + 단일 FC 128 | 3,320,705 | 5.3757 | 0.000% | validation subset 16장 |

**해석:** 공간 head의 마지막 train loss가 이 한 epoch에서 약 5.15% 낮았습니다. 파라미터는 65,338개(약 2.01%) 늘었습니다. 두 head 모두 AP가 0이고 validation이 16장뿐이므로, 정확도 개선을 입증하지 못했습니다. 구조 후보로만 유지하고 전체 validation 정확도 결과로 승격하지 않습니다.

**구현 당시 결정:** 3×3 pooling + 단일 FC 128 head를 ablation 후보로 추가하고 `roi_head_pool_size`와 `roi_head_dim`을 설정으로 노출했습니다. 초기에는 subset 결과만으로 정확도를 판정할 수 없어 GAP(`roi_head_pool_size: 1`, `roi_head_dim: 0`)를 기본으로 유지했습니다. 제한된 학습 subset에서 500-step warmup이 한 epoch 동안 학습률을 거의 0으로 만드는 것을 확인해, subset 크기에 맞게 warmup step을 자동 축소했습니다. 전체 데이터 학습의 warmup 설정은 바꾸지 않습니다.

**다음 실험:** 전체 COCO validation을 고정한 채 더 많은 train image와 여러 epoch로 두 RoI head를 비교합니다. 그 뒤에만 accuracy와 checkpoint 크기를 근거로 기본 head를 선택합니다. 이번 측정의 CPU 처리량(약 1.4초/train image)을 그대로 환산하면 train 118,287장 1 epoch에 약 47시간, 36 epoch에 약 70일이며 실제 시간은 다를 수 있습니다. 전체 데이터 반복 학습에는 팀 GPU 환경이 필요합니다.

상세 결과: `outputs/roi_head_ablation/REPORT.md`, `outputs/roi_head_ablation/results.json` (로컬 실행 산출물; Git ignore 대상).

## 2026-10-18 — 중단된 200/500 subset 확인

CPU GAP head를 train 200장, validation 500장, 3 epoch로 확인하던 중 사용자가 전체 학습은 직접 실행하기로 해 실험을 중단했습니다. 저장된 두 epoch의 train loss는 5.4146 → 4.1826으로 감소했지만 두 번 모두 AP@[.50:.95], AP50이 0이었습니다. 이 결과는 3 epoch 실험도 아니고, spatial head와의 비교도 완료되지 않았으므로 아키텍처 성능 결론으로 사용하지 않습니다. COCO 사전학습 분류기/RPN 없이 ImageNet backbone과 새 detection head로 단 200장만 학습한 초기 상태에서 validation 매칭이 아직 나오지 않은 것으로 해석합니다.

후속 실행에서 원인을 구분할 수 있도록 공식 검증 JSON에 예측 수, 예측이 하나 이상 나온 이미지 수, 최고 confidence를 기록합니다. COCO crowd 영역은 학습 positive에서 제외하고, 학습 시작 때 네 데이터 경로를 사전 검사하도록 보완했습니다. 산출물은 `outputs/roi_head_ablation_200x500/gap/history.jsonl`이며 체크포인트는 이어서 사용하지 않고 새 전체 COCO run을 시작합니다.

중단 checkpoint를 새 진단 evaluator로 추가 점검한 결과, validation 앞 16장에서는 AP50/AP@[.50:.95]가 모두 0이었지만 961개 예측을 냈고 15/16 이미지에 예측이 있었습니다. 최고 confidence는 0.0758이었습니다. 따라서 문제는 빈 prediction이 아니라, 초기 head의 낮은 confidence 및/또는 잘못된 클래스·박스 매칭입니다. 평가 범위가 16장이라 COCO 성능 추정치로 해석하지 않습니다. 출력 JSON은 임시 경로 `lite_rcnn_diag16.json`에 두었습니다.

후속 전체 학습을 위한 작업 가설로 두 기본 COCO 설정의 RoI head를 3×3 adaptive pooling + FC 128로 변경했습니다. 근거는 16장×1 epoch 초기 ablation에서 GAP보다 train loss가 낮았고(5.6678 → 5.3757), 파라미터 증가는 약 2.01%였다는 점입니다. AP는 두 설정 모두 0이어서 공간형 head가 더 정확하다고 입증된 것은 아닙니다. 사용자가 시작할 새 run에서 충분히 학습한 뒤 전체 val2017 AP로 선택을 재검토해야 합니다. 기존 GAP checkpoint는 head 차원 불일치로 재개할 수 없으며, 새 output directory에서 시작합니다.

전체 학습 기본 파이프라인에 입력 RGB 정규화와 train 전용 random horizontal flip(probability 0.5)을 추가했습니다. 이미지와 COCO xyxy box의 x1/x2를 함께 좌우 변환하며 validation은 변경하지 않습니다. 증강 사용 여부는 checkpoint 설정에 저장되고 resume 시 동일해야 합니다. 이 증강의 AP 효과는 다음 학습 결과에서 검증할 예정입니다.

현재 로컬 annotation JSON을 직접 읽어 데이터 매핑을 확인했습니다. train은 118,287 images / 860,001 annotations, validation은 5,000 images / 36,781 annotations이며, 두 split 모두 80개 COCO category ID와 annotation category ID가 일치하고 범위는 1–90입니다. 따라서 학생의 `num_classes: 91` 및 원래 COCO category ID 유지와 호환됩니다. annotation category 불일치는 기존 AP 0의 원인으로 보이지 않습니다.

저장된 200장×2 epoch GAP checkpoint로 confidence threshold 민감도를 추가 점검했습니다(동일한 validation 앞 16장). 기본 threshold 0.05에서 961 detections, 0.001에서 최대 1,600 detections가 나왔지만 양쪽 모두 AP50/AP@[.50:.95]=0이었습니다. 예측을 더 허용하는 것만으로는 현재 AP 0을 해결하지 못했고, 아직 box/class 자체가 정답과 맞지 않는 초기 상태라는 해석을 강화합니다. 임의로 score threshold를 낮추지 않고 충분한 지도학습을 먼저 수행합니다.

같은 checkpoint와 subset에서 클래스 무관/일치 박스 겹침도 분리했습니다. 이미지마다 모든 예측-정답 쌍 중 최고 IoU의 평균은 0.567이었고 16장 중 10장에는 어떤 클래스든 IoU≥0.5인 예측이 있었습니다. 예측 label까지 같은 쌍의 최고 IoU 평균은 0.021이었고, 16장 모두 같은 label로 IoU≥0.5인 예측이 없었습니다. 이는 이 초기 checkpoint에서는 localization보다 RoI 분류가 더 큰 병목임을 시사합니다. 단 16장·2 epoch 진단이므로 일반화된 성능 결론은 아니며, 다음 학습에서 분류 loss와 full-val AP를 함께 확인합니다.

COCO category ID에 빈 번호가 있는 것이 분류 실패 원인인지도 확인했습니다. threshold 0.001의 예측 1,600개 모두 실제 COCO ID(80개)에 속했고 빈 category ID 출력은 0개였습니다. 따라서 클래스 head를 91→81로 단순 축소하는 것만으로 관찰된 오분류를 해결할 근거는 없습니다. 분류 head의 충분한 지도학습 또는 RoI-logit distillation이 다음 후보이며, 정확도 효과는 별도 학습 후에만 판단합니다.

관찰된 분류 병목을 겨냥해 `--distill-roi-only` 학습 옵션을 추가했습니다. 이 옵션은 기존 teacher/student 동일 proposal 경로를 사용하고 feature KD만 끄므로 `loss_kd_roi`가 분류 증류의 효과를 분리합니다. teacher forward가 추가되어 CPU 비용이 크며, AP 개선은 아직 측정하지 않았습니다. supervised-only baseline 이후 GPU에서 같은 full-val 프로토콜로 비교하는 후속 연구입니다.

향후 validation 진단을 반복 가능하게 하려고 evaluator가 이미지별 best IoU를 클래스 무관/일치 기준으로 추가 집계합니다. 이 값은 best single-pair IoU의 단순 평균이며 recall 지표가 아닙니다. official COCOeval AP와 함께 진단용으로만 해석합니다.

새 evaluator를 기존 checkpoint에 적용해 출력 동작도 확인했습니다. 기본 score threshold 0.05의 앞 16장에서는 `detections=961`, 클래스 무관 best-IoU 평균 0.426, IoU≥0.5인 이미지 8/16, 클래스 일치 best-IoU 평균 0.000, 같은 클래스 IoU≥0.5 이미지 0/16이었고 official AP50은 0이었습니다. 앞서 threshold 0.001에서 나온 평균 0.567 / 10/16과 차이가 나는 것은 낮은 점수 box가 더 포함되기 때문이며 두 결과 모두 분류 일치 문제를 보여줍니다.

같은 200장×2 epoch GAP checkpoint를 validation 앞 500장으로 확대 점검했습니다. 기본 threshold 0.05에서 33,792개 예측, 490/500 이미지에 예측이 있었고, 이미지별 클래스 무관 최고 IoU 평균은 0.416, IoU≥0.5가 한 쌍 이상 있는 이미지는 200/500이었습니다. 클래스 일치 최고 IoU 평균은 0.000, 클래스 일치 IoU≥0.5 이미지는 0/500이었으며 official AP50/AP@[.50:.95]도 0이었습니다. 클래스 혼동 병목이 16장 subset에만 국한되지 않는다는 근거지만, checkpoint가 2 epoch뿐이고 validation도 500/5,000장이므로 충분한 학습 성능으로 해석하지 않습니다.

분류 문제를 proposal localization에서 분리하기 위해 같은 validation 500장의 3,504개 GT box를 그대로 RoI head에 입력했습니다. GT 위치에서 top-1 class 정확도는 0.0285%(background 포함/제외 동일), GT class 평균 확률은 0.00783, cross-entropy는 5.175였습니다. 따라서 이 2 epoch checkpoint의 분류 head는 정답 box에서도 거의 학습되지 않은 상태입니다. 낮은 AP를 더 학습된 모델의 구조 한계로 단정하지 않으며, 충분한 supervised 학습과 선택적 RoI-logit KD 후 같은 측정을 다시 해야 합니다.

같은 방식으로 기존 train 200장의 GT box 1,479개를 평가한 결과 top-1 정확도 0.000, GT class 평균 확률 0.00821, cross-entropy 5.124였습니다. top prediction은 background 1,253개로 가장 많았습니다. train/eval 양쪽 모두 정답 box 위치에서도 분류가 학습되지 않은 상태여서 validation domain shift만으로 설명되지 않습니다. 16장 train/val에 대해 BN batch statistics와 running statistics도 비교했으며 두 모드 모두 top-1 0이었습니다(CE는 train 5.040 vs 4.768, val 5.077 vs 4.964). 따라서 BN running statistics가 단독 원인이라는 증거는 없습니다. 이 결과는 200장×2 epoch GAP checkpoint의 진단이며 현재 공간형 head 전체 학습 결과는 아닙니다.

최신 모델 코드에서 parameter count를 다시 산출했습니다. 현재 공간형 head는 3,320,705 parameters, GAP 기준선은 3,255,367 parameters로 차이는 65,338(2.01%)입니다. 공간형 설정의 추가 파라미터 비용은 작지만 정확도 이점은 전체 학습 후 검증이 필요합니다.

subset 데이터 구성을 확인하니 `--train-limit 200`은 COCO image ID 정렬 기준 앞 200장을 사용했고, 이 표본의 non-crowd annotation은 1,479개 중 person(category 1)이 412개(27.9%)였습니다. 전체 train의 person 비율 30.3%와 크게 다르지 않았습니다. category 89가 0개였지만 전체 non-crowd annotation 중 198/849,949(0.023%)인 희귀 class라 200장 sample에서 누락되는 것은 예상 가능합니다. 따라서 first-N 표본이 전반적으로 심하게 편향됐다고 단정하지 않습니다. 이후 subset 학습과 validation은 config seed에 따른 고정 무작위 표본을 사용하도록 바꿨고, 같은 seed의 GAP/spatial ablation이 같은 이미지를 공유합니다. full split(limit 0)은 기존대로 전부 사용합니다. 과거 first-N subset 결과는 이전 표본 기준의 기록으로 남깁니다.

seeded-random validation 경로를 기존 checkpoint로 실행 확인했습니다. seed 42의 validation subset 16장에서 AP50/AP@[.50:.95]=0, detection 1,028개, 클래스 무관 best-IoU 평균 0.393, 클래스 일치 평균 0.000이 출력되고 결과에 `evaluation_subset_seed: 42`가 기록됐습니다. 이전 first-16 결과와 표본이 달라 직접 비교하지 않습니다. 이는 evaluator 경로 확인과 subset seed 기록 확인입니다.

정답 box 분류 진단의 top prediction histogram은 class 89가 3,504개 중 3,092개, class 43이 206개로 쏠렸습니다. class 89는 기존 200장 train subset에 없었지만 COCO 전체에서도 매우 드문 class이며, 단순 누락만으로 이 정도 collapse를 설명할 수 없습니다. 원인은 미확정이며, 충분한 학습과 새 seeded-random 실험에서 다시 확인합니다.

seed 42의 새 random train subset 200장은 non-crowd annotation 1,491개 중 person(category 1)이 512개(34.3%), category 89는 1개였습니다. 전체 분포(person 30.3%, category 89 0.023%)와 비교해 subset 변동은 남지만, 동일 seed paired ablation에는 재현 가능한 표본입니다. 이는 subset 대표성을 개선한다고 보장하지 않고 image ID 순서 편향을 제거하기 위한 변경입니다.

## 2026-10-18 — 지속 연구 실행기 확인

`continuous_research.py --cycles 1 --epochs 1 --train-limit 16 --val-limit 16`로 한 cycle을 실행했습니다. 실행기가 `round_001/{last,best}.pth`, `state.json`, `experiments.jsonl`, `RESEARCH_REPORT.md`를 생성하고 다음 cycle 설정(learning rate `1e-4`)을 기록했습니다. 이 실행도 16장 validation AP는 0.000%였으며 자동화 경로 확인용입니다. 기본 `--cycles 0`의 무제한 반복이나 전체 COCO 학습은 아직 실행하지 않았습니다.
