# Light R-CNN

Light R-CNN은 Faster R-CNN의 최적화 버전입니다. 표준 앵커 기반 RPN 및 RoI 탐지 흐름을 유지하면서, MobileNetV3 + BiFPN 백본, 축소된 RPN 제안 한도(proposal budget), RoIAlign 이전의 특징 압축, 그리고 GAP RoI 헤드를 사용합니다. 성능 비교는 기본(Vanilla) Faster R-CNN과 Light R-CNN 간에 이루어지며, YOLO26을 실시간 처리 기준선으로 활용합니다.

## 현재 상태

이 저장소는 경량 탐지 모델, 검증 AP 기반 최적 모델 저장, 학습 재개 및 선택적 교사/학생 지식 증류를 포함합니다. 학습된 Light R-CNN 가중치, TensorRT RoIAlign/NMS 플러그인 또는 검증된 COCO/T4 지표는 아직 포함되어 있지 않습니다. CPU 합성 데이터 통합 테스트로 학습·재개·증류 경로를 검증하며, 실제 COCO 성능은 별도 학습과 평가가 필요합니다. COCO AP ≥ 60 및 배치-1 T4 지연 시간 ≤ 15 ms는 프로젝트의 목표 수치이며, 현재 달성된 결과가 아닙니다. 자세한 내용은 [docs/ROADMAP.md](docs/ROADMAP.md)를 참조하십시오.

성능 비교를 위해 torchvision의 COCO 사전 학습 모델인 ResNet-50-FPN Faster R-CNN을 기본 베이스라인으로 사용하며, ImageNet 사전 학습 MobileNetV3 백본에서 학습된 Light R-CNN, 그리고 기본 참조선으로 YOLO26 nano(`yolo26n.pt`)를 사용합니다. 모든 모델은 기본적으로 640 픽셀 입력과 FP32 프레임워크 추론을 사용합니다. YOLO26은 NMS가 없는 엔드투엔드 추론을 선택적으로 지원하므로, 평가 시 선택한 체크포인트의 추론 모드를 기록해야 합니다. [공식 YOLO26 문서](https://docs.ultralytics.com/models/yolo26/)

## 설정

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,export,compare-yolo,dashboard]"
```

시스템에 Python 3.12가 설치되어 있지 않으면 먼저 Python 3.12를 설치하십시오. 이 프로젝트는 Python 3.10 이상이 필요합니다. Windows에서 `py` 런처를 사용할 수 없는 경우 설치된 Python 3.12의 실행 파일로 `-m venv .venv`를 실행하면 됩니다.

COCO 이미지는 `data/coco/images/{train2017,val2017}`, 주석은 `data/coco/annotations/instances_{train2017,val2017}.json` 구조로 배치하십시오. 패키지를 설치하기 전에 시스템의 CUDA 런타임 버전과 일치하는 PyTorch/torchvision 빌드를 설치해야 합니다.

## 현재 진입점 (Entry points)

COCO 검증 데이터와 `outputs/coco/best.pth`가 준비되어 있으면 아래 한 명령으로 Faster R-CNN → Light R-CNN → YOLO26 순서로 COCO 평가와 지연 시간 측정을 실행한 뒤 대시보드를 엽니다. 아직 Light R-CNN 가중치를 학습하지 않았다면 `--train-light`를 추가하십시오. CPU 학습과 전체 COCO 평가는 오래 걸릴 수 있습니다.

```powershell
python scripts/run_comparison.py --device cpu
# 체크포인트가 없고 COCO train split도 준비되어 있다면:
python scripts/run_comparison.py --device cpu --train-light
```

COCO 주석 없이 동일한 이미지 한 장(또는 이미지 폴더)으로 지연 시간만 먼저 확인하려면 `--inference-only --source`를 사용합니다. 이 모드는 mAP를 계산하지 않으므로 대시보드에는 지연 시간만 표시됩니다. Lite R-CNN 체크포인트가 없으면 ImageNet 사전학습 백본과 학습되지 않은 검출 헤드로 속도만 측정합니다. 이 경우 Lite R-CNN의 탐지 결과나 정확도는 비교할 수 없습니다.

```powershell
python scripts/run_comparison.py --device cpu --inference-only --source data/samples/test.jpg
```

실행 중 하나라도 실패하면 뒤 단계와 대시보드 실행은 중단됩니다. 반복 횟수와 예열 횟수는 `--iterations`, `--warmup`으로 조절하고, 빠른 부분 평가가 필요하면 `--limit`으로 세 모델의 평가 이미지 수를 함께 제한할 수 있습니다. 부분 평가 결과는 전체 COCO 평가 결과와 구분해서 해석하십시오. 결과 JSON은 실행별로 `outputs/comparison_runs/` 아래에 저장됩니다.

개별 학습, 평가, ONNX 내보내기 명령은 아래 스크립트를 직접 실행할 수 있습니다.

```powershell
python scripts/train.py --config configs/coco.yaml
python scripts/export_onnx.py --weights outputs/coco/best.pth
```

학습 진입점은 torchvision의 Faster R-CNN RPN/RoI 손실 함수를 사용합니다. 매 epoch `last.pth`를 저장하고, 검증 COCO AP@[.50:.95]가 개선되면 `best.pth`를 저장합니다. `--resume`으로 학습을 재개하고 `--distill`로 지식 증류를 활성화합니다. 실행 방법과 저장 상태는 [학습 안내](docs/TRAINING.md)를 참조하십시오. ONNX 내보내기는 고정 크기(fixed-shape)의 전체 탐지 경로를 지원합니다. 타겟 TensorRT 버전에 대한 RoIAlign 및 NMS 지원 검증은 [deploy/tensorrt/README.md](deploy/tensorrt/README.md)에 설명된 절차를 따르십시오.

YOLO26 스크립트는 JSON 타이밍 결과를 저장하고, 평가 스크립트는 COCO 주석에서 YOLO 형식 검증 레이블을 생성해 COCO 지표를 계산합니다. Faster R-CNN 벤치마크는 두 PyTorch 모델의 배치-1 p50/p95 지연 시간을 JSON으로 저장합니다. `streamlit run dashboard/app.py`로 세 모델 결과 대시보드를 실행할 수 있습니다. 최종 T4 비교는 가능한 경우 세 모델 모두 같은 정밀도와 측정 범위로 반복해야 합니다.

대시보드는 `outputs/**/*.json`을 자동으로 합쳐 세 모델의 AP-지연 시간 산점도, p50/p95 지연 막대 그래프, COCO AP 그래프와 수치 표를 한 화면에 보여줍니다. 평가와 벤치마크를 마친 뒤 대시보드를 실행하면 각 모델 결과가 같은 행으로 합쳐집니다.

## 설계 참고 사항

* 표준 앵커 기반 RPN 및 2단(two-stage) RoI 탐지 방식을 유지합니다.
* 추론 시 Light R-CNN의 RPN 제안(proposal) 개수는 300개로 제한됩니다(기본 Faster R-CNN은 표준 설정을 유지).
* BiFPN 및 깊이별 분리 합성곱(separable convolutions)을 적용하여 다중 스케일 특징 병합(feature-fusion) 연산 비용을 감소시킵니다.
* FPN 특징은 RoIAlign 단계 이전에 10채널로 투영되며, RoI 박스 헤드에는 전역 평균 풀링(global average pooling)이 사용됩니다.
* 지식 증류는 `src/light_rcnn/distillation.py`에서 같은 변환 이미지와 같은 sampled RoI를 사용합니다. stride 8/16/32 특징을 학습 가능한 채널 투영으로 정렬하고, 특징 손실과 temperature-scaled RoI KL 손실을 지도학습 손실에 더합니다. 교사 모델은 eval 상태와 gradient 비활성화를 유지하며 최종 추론 모델에 포함되지 않습니다.
