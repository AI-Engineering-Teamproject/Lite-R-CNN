# 세 모델 검증 평가 (`evaluate_three_models.py`)

torchvision COCO 사전학습 Faster R-CNN과 Lite R-CNN v2·v3를 **같은 COCO 이미지, 같은 설정**으로 평가해 표를 만듭니다.
`--with-mobilenet`을 붙이면 MobileNetV3 Faster R-CNN을 네 번째 모델로 같은 설정에서 함께 평가합니다 (아래 [MobileNet 비교군](#mobilenet-비교군---with-mobilenet)).
저장소 최상위 폴더에서, COCO val2017 이미지와 주석이 `data/coco/`에 있고 v2/v3 체크포인트가 기본 위치
(`outputs/coco_4090_v2/best.pth`, `outputs/coco_4090_v3/best.pth`)에 있을 때 실행합니다.

```
python scripts/evaluate_three_models.py --device cpu
python scripts/evaluate_three_models_gpu_full.py
python scripts/evaluate_three_models_gpu_full.py --with-mobilenet
```

- 첫 줄: CPU에서 seed 42로 고른 1000장을 평가합니다 (기본값).
- 둘째 줄: 같은 평가를 GPU(`--device cuda`)에서 val2017 전체 5000장으로 합니다. 기본값 두 개만 다르고 옵션은 같습니다.
  Colab 노트북 단계 C가 이 스크립트를 씁니다 (노트북 설정 `WITH_MOBILENET`이 켜져 있으면 셋째 줄처럼 `--with-mobilenet`을 붙임).
- 셋째 줄: 둘째 줄에 MobileNetV3 Faster R-CNN을 더해 네 모델을 평가합니다. 기본은 꺼져 있어 옵션 없이 돌리면 예전과 같은 세 모델 결과가 나옵니다.
- CPU에서 기본값으로 돌리면 Faster R-CNN의 지연 시간 측정에만 수십 분이 걸립니다 (4코어 CPU에서 약 40분).
  빨리 보려면 `--timing-images 16 --warmup 16 --iterations 160`처럼 줄이세요.

## 측정 방식

| 항목 | 내용 | 옵션 (기본값) |
|---|---|---|
| 평가 이미지 | seed로 고른 같은 이미지 목록을 모든 모델이 함께 씁니다. 전체 수 이상이면 전체 | `--images` (1000, GPU 스크립트 5000), `--seed` (42) |
| 입력 | 크기 640, 배치 1 (torchvision 두 기준 모델도 `min_size = max_size = 640`) | `--image-size` (640) |
| RPN 제안 수 | **모든 모델 300개**. Faster R-CNN(과 `--with-mobilenet`의 MobileNetV3 Faster R-CNN)의 기본 1000개도 300개로 맞춰 AP, 제안 재현율, 단계별 시간, FLOPs, 지연 시간을 같은 설정에서 잽니다. 지연 시간은 제안 100개로도 한 번 더 잽니다 | – |
| Conv+BN 융합 | **기본 켜짐**. 모든 모델의 BatchNorm(torchvision 사전학습 백본의 FrozenBatchNorm2d 포함)을 앞 Conv에 합친 배포용 그래프로 AP와 지연 시간을 잽니다. AP는 부동소수점 오차 범위에서 같고, 매개변수 수는 융합 전 기준입니다 | `--fuse` / `--no-fuse` (학습 그래프 그대로) |
| 검출 점수 하한 | 모델 기본값 0.05, 이미지당 최대 100개. `run_comparison.py` 비교 프로토콜(0.001)의 AP와 섞지 마세요 | – |
| 지연 시간 | 평가 이미지 중 seed로 고른 이미지들을 차례로 반복해 잽니다. 예열은 지연 시간 이미지 수보다 적으면 그만큼 늘리고, 측정 횟수는 이미지 수 이상이어야 합니다 (기본은 이미지당 10회) | `--timing-images` (64), `--warmup` (64), `--iterations` (640) |
| 지연 시간 통계 | p50/p95/p99는 전체 측정값의 백분위수(선형 보간)입니다. **p95 of image medians**는 이미지별 중앙값의 95번째 백분위수라, 원래 느린 이미지와 일시적인 지연을 구분해 줍니다. 원시 측정값과 이미지 ID는 `evaluation.json`의 `latency_runs`에 있습니다 | – |
| 단계별 시간 | 백본, BiFPN, RPN, RoI 헤드. 첫 지연 시간 이미지로 반복한 중앙값 (Faster R-CNN과 MobileNetV3 Faster R-CNN은 BiFPN이 없어 비어 있고, 백본 시간에 FPN이 들어 있음) | `--stage-repeats` (5) |
| 제안 재현율 | R@100·R@300 (IoU 0.5 / 0.75). RPN 제안을 **원본 이미지 좌표**로 되돌린 뒤(torchvision `resize_boxes`) 정답 상자와 비교합니다 | – |

다른 체크포인트를 쓰거나 측정 횟수를 바꿀 때 (한 줄로 입력):

```
python scripts/evaluate_three_models.py --device cpu --v2-weights outputs/coco_4090_v2/best.pth --v3-weights outputs/coco_4090_v3/best.pth --images 1000 --seed 42 --timing-images 64 --warmup 64 --iterations 640
```

### Faster R-CNN 가중치

기본은 torchvision COCO_V1 가중치를 내려받습니다. 인터넷이 막혀 있으면 받아 둔 state dict를 `--faster-weights 파일.pth`로 주거나,
`--skip-faster-rcnn`으로 Faster R-CNN을 빼고 Lite R-CNN v2/v3만(`--with-mobilenet`이면 MobileNetV3 Faster R-CNN도) 평가합니다. 이때 Faster R-CNN 대비 속도 향상과 AP 차이는 비어 있습니다.

## MobileNet 비교군 (`--with-mobilenet`)

MobileNetV3 Faster R-CNN은 torchvision `fasterrcnn_mobilenet_v3_large_fpn`(COCO_V1 가중치, 320 전용 모델 아님)입니다.
팀원 브랜치 `feat/personal-comparison-ci`의 독립 네 모델 비교에서 쓴 "MobileNet 벤치마크"와 같은 모델을 이 스크립트의 프로토콜로 잽니다.

- **왜 넣나**: Lite R-CNN과 같은 MobileNetV3-Large 계열 백본을 쓰는 기존 경량 Faster R-CNN이라, Lite R-CNN이 이미 있는 가벼운 R-CNN 옆에서
  어디쯤인지 보여 줍니다. 학습 과정·특징 융합(FPN과 BiFPN)·앵커가 모두 달라 BiFPN 하나의 효과를 가르는 ablation은 아닙니다.
- **같게 맞추는 것**: 평가·지연 시간 이미지, 입력 640, RPN 제안 300개, Conv+BN 융합, 원본 좌표의 제안 재현율, 지연 시간 측정 방식.
- **그대로 두는 것**: 앵커(세 레벨 모두 32~512 × 비율 0.5/1/2)와 RPN 객체성 점수 하한 0.05 등 torchvision 기본값. 모델마다 `evaluation.json`의
  `rpn_settings`(점수 하한, RPN NMS IoU, 앵커 크기·비율)에 기록합니다 (CSV 열은 그대로).
- **가중치**: torchvision 문서의 box AP 32.8은 800px(짧은 변 800, 긴 변 최대 1333) 기준이라 640에서는 더 낮게 나올 수 있습니다.
  인터넷이 막혀 있으면 `--mobilenet-weights 파일.pth`(state dict 또는 `{"model": ...}`)로 주세요. 내려받지 못하면 Faster R-CNN을 평가한 뒤
  멈추고, 끝난 모델의 결과는 `partial_results.json`에 남습니다 (`--skip-faster-rcnn`이면 끝난 모델이 없어 결과 폴더도 생기지 않음).
  MobileNet 없이 돌리려면 `--with-mobilenet`을 빼면 됩니다. `REPORT.md`의 설명 줄에는 쓴 가중치(torchvision COCO_V1 또는 로컬 파일 경로)가 적힙니다.
- 숫자는 Colab T4(단계 C) 또는 GPU에서 나옵니다. 이 저장소의 CPU 스모크 환경은 torchvision 가중치를 받을 수 없어 MobileNet을 재지 못했습니다.

## 결과

`outputs/evaluation_runs/<실행 시각>/`(`--output-dir`로 변경)에 저장합니다.

- `REPORT.md`: 모델별 AP/mAP, AP50, AP75, AP_S/M/L, Precision, Recall, p50, FPS, 매개변수, FLOPs, p95, p99, p95 of image medians, RPN 제안 수, Conv+BN 융합 여부
  (제목은 평가한 모델 수를 따름: 기본 "Three-model detection evaluation", `--with-mobilenet`이면 "Four-model ...", 여기에 `--skip-faster-rcnn`을
  붙이면 하나 적은 "Two-model ..."·"Three-model ...". `--with-mobilenet`이면 MobileNetV3 Faster R-CNN 행과 설명 한 줄이 더해짐)
- `summary.csv`: 위 값에 더해 AR@100, 체크포인트·모델 크기, 최대 VRAM, 제안 100/300개 지연 시간, Faster R-CNN 대비 속도 향상과 AP 차이, 단계별 시간, 제안 재현율.
  `--with-mobilenet`이면 행이 하나 늘고 열은 같습니다
- `evaluation.json`: 모든 값과 메타데이터 (평가·지연 시간 이미지 ID, 원시 측정값, 실행 환경, 커밋, 모델별 `rpn_settings`, `with_mobilenet`)
- 모델 하나가 끝날 때마다 `partial_results.json`을 갱신하므로, 중간에 멈춰도 끝난 모델의 결과는 남습니다.

지표 정의:

- Precision과 Recall은 점수 0.5 이상인 검출을 같은 클래스의 정답과 IoU 0.5 이상으로 짝지어 전체 개수로 셉니다.
- AP는 COCO AP@[.50:.95]이고 mAP와 같은 값입니다.
- FLOPs는 PyTorch profiler가 세는 Conv·MatMul 연산만 더한 추정치라 RoIAlign, NMS 같은 검출 연산은 빠질 수 있습니다.
- VRAM은 검증 추론(배치 1) 동안의 최대 CUDA 할당량이고, CPU에서는 비어 있습니다.

## 예전 결과(2026-11-01까지)와 비교할 때

예전 버전 스크립트로 만든 표와는 측정 방식이 달라 숫자를 그대로 비교하면 안 됩니다.

- Faster R-CNN의 AP를 예전에는 제안 1000개로 쟀습니다 (지연 시간은 300개). 지금은 모두 300개라 AP와 FLOPs가 조금 낮게 나올 수 있습니다.
- 예전 제안 재현율은 리사이즈된 좌표의 제안을 원본 좌표의 정답과 비교해, 크기가 바뀐 이미지에서 낮게 나왔습니다.
- 예전 지연 시간은 평가 이미지 앞 8장, 예열 10회, 측정 100회, Conv+BN 융합 없음이었습니다.
- p95는 선형 보간으로 계산합니다 (예전에는 정렬한 값의 `int(0.95 × 개수)`번째). p99와 p95 of image medians는 새로 넣었습니다.
