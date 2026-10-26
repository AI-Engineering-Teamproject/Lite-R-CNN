# 성능 측정과 그래프

이 문서는 세 모델(바닐라 Faster R-CNN, Light R-CNN, YOLO26)을 **같은 규칙으로** 재고, 결과를 그래프로 보는 방법을 설명합니다. 명령은 저장소 최상위 폴더에서 실행합니다.

## 빠른 시작

```powershell
# 1) 세 모델 전체 비교: COCO 정확도 + 지연 시간 + 단계별 시간 + 그래프 + 대시보드 (COCO val2017 필요)
python scripts/run_comparison.py --config configs/coco.yaml --device cuda --light-weights outputs/coco_4090_v2/best.pth --yolo-weights yolo26n.pt

# 2) annotation 없이 지연 시간만: 이미지 한 장(--single-image) 또는 이미지 폴더(--inference-only --source)
python scripts/run_comparison.py --single-image data/coco/images/val2017/000000129756.jpg --device cuda --light-weights outputs/coco_4090_v2/best.pth
python scripts/run_comparison.py --inference-only --source path/to/images --device cuda --light-weights outputs/coco_4090_v2/best.pth

# 3) 학습 중에 잰 Light R-CNN 검증 정확도를 결과 파일로 가져오기
python scripts/import_training_results.py outputs/coco_4090_v2

# 4) 그래프 보기
streamlit run dashboard/app.py
python scripts/plot_results.py --reports outputs/comparison_runs/<실행 시각> --out outputs/figures --theme both
```

## 한 번에 실행하기: `run_comparison.py`

`run_comparison.py`는 모델마다 **COCO 평가 → 지연 시간 측정 → (R-CNN 계열만) 단계별 시간 분석** 을 실행하고, 그래프 파일을 만든 뒤 대시보드를 엽니다. 모델 순서는 `--models`에 적은 순서와 상관없이 항상 Faster R-CNN → Light R-CNN → YOLO26입니다.

- 결과는 실행할 때마다 새 폴더 `outputs/comparison_runs/<실행 시각>/`에 `<모델>_<종류>.json`으로 저장되고, 그래프는 그 안의 `figures/`에 저장됩니다.
- 대시보드는 환경 변수 `LIGHT_RCNN_RESULTS_DIR`로 방금 끝난 실행 폴더를 열어 줍니다. `--no-dashboard`를 주면 대시보드를 띄우지 않습니다.
- 단계 하나가 실패하면 그 자리에서 멈추고, 그때까지 끝난 결과가 어느 폴더에 있는지 알려 줍니다.

| 옵션 | 뜻 |
|---|---|
| `--device cpu` / `cuda` / `cuda:0` | 측정 장치. **기본값은 `cpu`** 이므로 GPU에서 재려면 꼭 지정하십시오. CPU에서 `--config`를 생략하면 반복 횟수 등은 `configs/coco_cpu.yaml` 값(5/30/16)을 씁니다 |
| `--config` | 데이터 경로와 `comparison` 설정. 기본 `configs/coco.yaml`, CPU용은 `configs/coco_cpu.yaml` |
| `--light-weights` | Light R-CNN 체크포인트. 기본 `<train.output_dir>/best.pth`. 구조(백본, 채널 수 등)는 체크포인트 안에 저장된 학습 설정에서 읽습니다 |
| `--yolo-weights` | YOLO26 가중치. 기본 `comparison.control.weights`(`yolo26n.pt`) |
| `--faster-weights` | Faster R-CNN 체크포인트. 생략하면 torchvision COCO 가중치를 내려받습니다 |
| `--models light_rcnn yolo26` | 일부 모델만 측정 |
| `--single-image IMAGE` | 이 이미지 한 장으로 지연 시간만 측정 (정확도 평가 생략) |
| `--inference-only` + `--source` | 이미지 파일 또는 폴더로 지연 시간만 측정 (정확도 평가 생략) |
| `--source` | 지연 시간 측정에 쓸 이미지 파일/폴더. 기본은 config의 `val_images` |
| `--limit N` | COCO 평가를 N장으로 줄임 (세 모델이 같은 N장을 평가) |
| `--warmup`, `--iterations`, `--image-count` | 예열 횟수, 측정 반복 횟수, 돌려 쓸 이미지 수. 기본값은 config의 `comparison` |
| `--yolo-nms-free` | YOLO26을 NMS 없는(one-to-one) 헤드로 측정 (평가와 지연 시간 모두) |
| `--train-light` | 비교 전에 Light R-CNN을 먼저 학습 (COCO train split 필요) |

인터넷이 막혀 torchvision 가중치(download.pytorch.org)를 받을 수 없으면 Faster R-CNN 단계에서 실패합니다. 다른 곳에서 받아 둔 체크포인트를 `--faster-weights`로 주거나, `--models light_rcnn yolo26`으로 Faster R-CNN을 빼고 실행하십시오.

이미지 한 장 결과는 동작 확인용입니다. 같은 사진만 반복하므로 일반적인 지연 시간 분포를 대표하지 않습니다.

## 스크립트를 하나씩 실행하기

`run_comparison.py`가 부르는 스크립트를 직접 실행할 수도 있습니다. 직접 실행하면 기본 저장 위치는 `outputs/reports/<모델>_<종류>.json`입니다 (`--output`으로 변경).

| 스크립트 | 하는 일 | 자주 쓰는 옵션 |
|---|---|---|
| `evaluate_coco.py` | Light R-CNN / Faster R-CNN COCO 정확도 | `--model`, `--weights`, `--limit`, `--batch-size`(기본 `comparison.batch_size` = 1). `--config`를 생략하면 구조와 데이터 경로는 체크포인트에 저장된 학습 설정을 씁니다 |
| `evaluate_yolo26.py` | YOLO26 COCO 정확도 (같은 pycocotools 계산기) | `--weights`, `--limit`, `--nms-free` |
| `benchmark.py` | R-CNN 계열 지연 시간 | `--model`, `--weights`, `--source`, `--iterations`, `--channels-last` / `--no-channels-last` |
| `benchmark_yolo26.py` | YOLO26 지연 시간 | `--weights`, `--source`, `--iterations`, `--channels-last` / `--no-channels-last` |
| `profile_stages.py` | R-CNN 단계별 시간 (입력·전처리, 백본, RPN, RoI 헤드, 후처리) | `--model`, `--weights`, `--source` |
| `import_training_results.py` | 학습 중 검증 결과를 정확도 결과 파일로 변환 | 아래 설명 참고 |

```powershell
python scripts/evaluate_coco.py --model light_rcnn --weights outputs/coco_4090_v2/best.pth --device cuda
python scripts/benchmark.py --model light_rcnn --weights outputs/coco_4090_v2/best.pth --device cuda --source data/coco/images/val2017
```

## 학습 중 검증 결과 가져오기: `import_training_results.py`

`scripts/train.py`는 epoch마다 COCO val로 검증하고 `history.jsonl`, `RESULTS.md`에 기록합니다. 이 스크립트는 그중 **최고 epoch**(`best.pth`로 저장된 epoch)의 값을 대시보드가 읽는 schema v2 정확도 파일로 바꿉니다. GPU나 COCO 데이터가 없는 컴퓨터에서도 학습 결과를 그래프에 올릴 수 있습니다.

```powershell
python scripts/import_training_results.py outputs/coco_4090_v2
# 기본 저장 위치: outputs/reports/light_rcnn_accuracy_training.json (evaluate_coco.py 결과를 덮어쓰지 않음)
```

기록되는 내용과 주의할 점:

- **최고 epoch 선택**: `history.jsonl`의 `best_epoch` 값을 쓰고, 없으면 AP@[.50:.95]가 가장 높은 epoch를 씁니다. `best.pth`의 epoch가 다르면 체크포인트에 저장된 지표를 쓰고, 그것도 없으면 멈춥니다.
- **평가 방식**: 학습 검증도 pycocotools(COCOeval)로 계산하지만, 점수 임계값은 모델 설정 `box_score_thresh`(0.05)입니다. 비교용 평가(`score_threshold` 0.001)와 다르므로 대시보드에 "검출 점수 임계값이 다릅니다" 경고가 뜹니다. 임계값이 낮을수록 AP가 조금 유리합니다.
- **지표 범위**: 학습 기록에는 AP 6개(AP@[.50:.95], AP50, AP75, 소형/중형/대형)만 있어 AR과 클래스별 AP는 비어 있습니다.
- **평가 배치 크기**: 학습 검증은 `val_batch_size`(v2는 4장씩)로 계산했습니다. R-CNN은 여러 장을 묶으면 패딩 때문에 AP가 1%p 안팎 달라질 수 있어, 비교용 평가(1장씩)와 섞이면 대시보드가 경고합니다.
- **평가 이미지**: 검증 범위(전체 val 5000장 또는 고정 부분 집합)를 그대로 기록합니다.
- **환경·시각**: 장치 이름과 PyTorch 버전, 코드 revision은 `RESULTS.md`에서 읽고, 측정 시각은 학습이 기록한 시각입니다. 그래서 같은 조건의 `evaluate_coco.py` 결과를 나중에 만들면 대시보드는 그쪽을 우선 사용합니다.
- 공정한 세 모델 비교의 최종 숫자는 `evaluate_coco.py`(또는 `run_comparison.py`)로 다시 평가한 값을 쓰십시오.

## 측정 규칙 (공정성)

| 항목 | 규칙 | 이유 |
|---|---|---|
| AP 계산 | 세 모델 모두 `pycocotools`(`src/light_rcnn/coco_eval.py`)로 계산 | 계산기가 다르면 같은 예측도 AP가 달라집니다 |
| 평가 이미지 | `--limit`을 쓰면 config의 `seed`로 고른 같은 무작위 부분 집합을 세 모델과 학습 검증이 함께 사용 | 다른 사진으로 채점하면 비교가 안 됩니다 |
| 점수 임계값 | 정확도: `score_threshold` 0.001, 최대 100개 · 지연 시간: `latency_score_threshold` 0.25 | 임계값이 낮을수록 AP가 유리하고 NMS 시간이 늘어납니다 |
| 평가 배치 크기 | 정확도 평가는 세 모델 모두 1장씩(`comparison.batch_size`) | R-CNN은 여러 장을 묶으면 패딩 때문에 AP가 달라집니다 |
| 타이밍 범위 | 디코드된 uint8 이미지(호스트 메모리) → 최종 검출. 장치 복사, 정규화, 리사이즈, 모델, 후처리/NMS 포함. 파일 디코드 제외 | 한 모델만 준비 과정을 빼고 재면 그 모델이 유리합니다 |
| CPU 스레드 수 | 세 모델을 같은 torch 스레드 수로 측정. YOLO 스크립트는 Ultralytics가 낮춘 스레드 수를 되돌립니다. 바꾸려면 실행 전에 `OMP_NUM_THREADS`를 설정 | 스레드 수가 다르면 지연 시간이 달라집니다 (다르면 대시보드가 오류로 표시) |
| 그래프 설정 | 세 모델 모두 Conv+BN을 융합. 메모리 형식은 config의 `comparison.channels_last`(또는 각 벤치마크의 `--channels-last`/`--no-channels-last`)를 따르고, 지정하지 않으면 Ultralytics 자동 규칙(x86 CPU에서는 channels_last)을 세 스크립트가 똑같이 적용 | CPU에서는 channels_last 여부만으로 1.5배 안팎 차이가 납니다 |
| 첫 추론 | 예열 전에 따로 기록(`cold_start_ms`). R-CNN은 장치에 올린 모델의 첫 추론, YOLO26은 Ultralytics 예측기 준비 포함 | 참고용입니다. 모델 간 직접 비교에는 쓰지 마십시오 |
| 반복 횟수 | `configs/coco.yaml` 500회, `configs/coco_cpu.yaml` 30회 (100회 미만이면 대시보드가 경고) | 적게 재면 p95/p99가 우연에 좌우됩니다 |
| 환경 기록 | GPU/CPU 이름, CUDA, PyTorch 버전, 커밋, 가중치 해시를 모든 결과에 저장 | 나중에 같은 조건으로 다시 잴 수 있어야 합니다 |

## 결과 파일

모든 결과는 **schema v2** JSON입니다 (`src/light_rcnn/reporting.py`).

- `kind: "accuracy"` — `evaluate_coco.py`, `evaluate_yolo26.py`, `import_training_results.py`: COCO 지표, 클래스별 AP, 평가 이미지 수, 부분 평가 여부, 점수 임계값
- `kind: "latency"` — `benchmark.py`, `benchmark_yolo26.py`: p50/p90/p95/p99, 평균, 표준편차, FPS, 첫 추론, **원시 측정값 전체**(`samples_ms`), 파라미터 수, GFLOPs
- `kind: "stages"` — `profile_stages.py`: 단계별 시간과 NMS 세부 시간

같은 모델·종류의 결과가 여러 개면 대시보드와 그래프는 **새 형식(v2) > 전체 평가 > 이미지 수가 많은 평가 > 최신 결과** 순서로 하나를 고르고, 어떤 파일을 썼는지 "데이터·출처" 탭에 보여 줍니다.

예전 형식(v1) 파일(예: `outputs/comparison_runs/20261019_225947/`의 RTX 4090 단일 이미지 결과)도 읽습니다. 다만 원시 측정값이 없어 분포 그래프를 그릴 수 없고, 장치는 `cuda (기기명 미기록)`처럼 표시됩니다.

## 그래프 보기

```powershell
streamlit run dashboard/app.py                                               # 대화형 대시보드
python scripts/plot_results.py --reports outputs/comparison_runs/<실행 시각> --out outputs/figures --theme both   # PNG·SVG·HTML 보고서
```

대시보드 왼쪽 메뉴:

- **결과 폴더**: 하위 폴더의 JSON까지 모두 읽습니다. 기본값은 `LIGHT_RCNN_RESULTS_DIR` 또는 `outputs`. 특정 실행만 보려면 `outputs/comparison_runs/<실행 시각>`을 입력하십시오.
- **JSON 결과 파일 추가**: 다른 컴퓨터에서 받은 결과 파일을 바로 올릴 수 있습니다.
- **지연 시간 측정 장치**: CPU와 GPU 결과가 섞이지 않도록 지연 시간·단계별 그래프는 고른 장치의 결과만 씁니다. 정확도(AP)는 장치와 무관하므로 항상 표시합니다.
- **목표 기준**: 지연 시간 15 ms, AP 60% 목표(`comparison.targets`)를 바로 바꿀 수 있습니다. 지연 시간 목표는 T4 GPU 기준이라, T4가 아닌 장치(CPU, RTX 4090 등)의 측정값은 참고용으로 회색 표시됩니다.

`plot_results.py`도 `--reports` 아래 하위 폴더의 JSON을 모두 읽고, 지연 시간·단계별 그래프에는 장치 하나의 결과만 씁니다. 기본은 대시보드처럼 가장 최근에 측정한 장치이고, 다른 장치는 `--device "RTX 4090"`처럼 대시보드에 보이는 장치 이름으로 고릅니다. 고른 장치와 뺀 장치는 실행할 때 출력합니다. `outputs` 전체에는 예전 RTX 4090 결과(v1, 장치 이름 `cuda (기기명 미기록)`)도 있으므로 보통은 실행 폴더 하나를 `--reports`로 주십시오.

대시보드는 `streamlit`, `plotly`, `pandas`만 있으면 실행됩니다. PNG/SVG 저장에는 `kaleido`와 Chrome 또는 Edge가 필요하고, 없으면 HTML 보고서만 만듭니다. Chrome이 기본 위치에 없으면 환경 변수 `BROWSER_PATH`에 실행 파일 경로를 지정하십시오.

| 그래프 | 보는 법 |
|---|---|
| 정확도 대 속도 | 왼쪽 위로 갈수록 좋습니다. 음영이 목표 영역이고, 점에서 오른쪽으로 뻗은 수염 끝이 p95입니다 |
| 지연 시간 백분위수 | p50에서 p99까지의 선이 길수록 가끔 크게 느려집니다 (꼬리 지연) |
| 지연 시간 분포 | 상자는 가운데 50% 측정값, 점은 튀는 값입니다 |
| 측정 안정성 | 반복할수록 선이 오르내리면 발열이나 다른 작업 때문에 측정이 흔들린 것입니다 |
| COCO 정확도 세부 지표 | IoU 기준별(느슨함 ↔ 엄격함)과 물체 크기별 AP |
| 클래스별 AP | 모델 간 차이가 큰 클래스. 어디를 개선해야 하는지 찾을 때 씁니다 |
| 단계별 처리 시간 | 가장 긴 조각이 먼저 줄여야 할 병목입니다 |
| 모델 크기와 연산량 | 파라미터 수와 GFLOPs. 실제 속도는 지연 시간 그래프가 기준입니다 |

그래프 위에 빨간 상자("공정한 비교가 아닙니다")가 보이면 그 문제를 먼저 해결한 뒤 결과를 해석하십시오. 자주 보는 경고:

| 경고 | 해결 방법 |
|---|---|
| 평가한 이미지 수가 다릅니다 / 부분 평가 결과입니다 | 세 모델을 같은 `--limit`(또는 전체 val2017)으로 다시 평가 |
| 평가 세트가 COCO val2017 전체(5000장)가 아닙니다 | 작은 스모크 세트 결과입니다. 학습 이미지가 섞인 세트면 AP가 실제보다 높게 나옵니다 |
| 검출 점수 임계값이 다릅니다 | 학습 중 검증(0.05)과 비교용 평가(0.001)가 섞인 경우가 많습니다. `evaluate_coco.py`로 다시 평가 |
| 서로 다른 장치에서 측정했습니다 | 왼쪽 메뉴에서 장치를 하나 고르거나 같은 장치에서 다시 측정 |
| 반복 횟수가 100회 미만입니다 | `--iterations 500` 이상으로 다시 측정 |

## 색과 모양

모델마다 색과 점 모양이 모든 그래프에서 같습니다: Light R-CNN 파랑·원, Faster R-CNN 주황·마름모, YOLO26 청록·사각형. 세 색은 색각 이상 시뮬레이션에서도 구분되도록 골랐고, 모양과 이름표를 함께 써서 색만으로 구분하지 않게 했습니다.
