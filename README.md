# Light R-CNN

Light R-CNN은 Faster R-CNN의 최적화 버전입니다. 표준 앵커 기반 RPN 및 RoI 탐지 흐름을 유지하면서, MobileNetV3 + BiFPN 백본, 축소된 RPN 제안 한도(proposal budget), RoIAlign 이전의 특징 압축, 그리고 GAP RoI 헤드를 사용합니다. 성능 비교는 기본(Vanilla) Faster R-CNN과 Light R-CNN 간에 이루어지며, YOLO26을 실시간 처리 기준선으로 활용합니다.

## 현재 상태

이 저장소는 초기 아키텍처와 학습 스캐폴드(scaffold)를 포함하고 있습니다. 학습된 Light R-CNN 가중치, 연결된 교사/학생(teacher/student) 학습 파이프라인, TensorRT RoIAlign/NMS 플러그인 또는 검증된 COCO/T4 지표는 아직 포함되어 있지 않습니다. COCO AP 평가기가 포함되어 있습니다. COCO AP ≥ 60 및 배치-1 T4 지연 시간 ≤ 15 ms는 프로젝트의 목표 수치이며, 현재 달성된 결과가 아닙니다. 자세한 내용은 [docs/ROADMAP.md](https://www.google.com/search?q=docs/ROADMAP.md)를 참조하십시오.

성능 비교를 위해 torchvision의 COCO 사전 학습 모델인 ResNet-50-FPN Faster R-CNN을 기본 베이스라인으로 사용하며, ImageNet 사전 학습 MobileNetV3 백본에서 학습된 Light R-CNN, 그리고 기본 참조선으로 YOLO26 nano(`yolo26n.pt`)를 사용합니다. 모든 모델은 기본적으로 640 픽셀 입력과 FP32 프레임워크 추론을 사용합니다. YOLO26은 NMS가 없는 엔드투엔드 추론을 선택적으로 지원하므로, 평가 시 선택한 체크포인트의 추론 모드를 기록해야 합니다. [공식 YOLO26 문서](https://docs.ultralytics.com/models/yolo26/)

## 설정

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[train,export]"

```

`configs/coco.yaml` 파일에 설정된 경로에 맞추어 COCO 2017 이미지 및 주석(annotation) 파일을 `data/coco/` 하위에 배치하십시오. 패키지를 설치하기 전에 시스템의 CUDA 런타임 버전과 일치하는 PyTorch/torchvision 빌드를 설치해야 합니다.

## 현재 진입점 (Entry points)

```powershell
python scripts/train.py --config configs/coco.yaml
python scripts/evaluate_coco.py --model faster_rcnn
python scripts/evaluate_coco.py --model light_rcnn --weights outputs/coco/last.pth
python scripts/benchmark.py --model faster_rcnn --device cuda
python scripts/benchmark.py --model light_rcnn --weights outputs/coco/last.pth --device cuda
python scripts/export_onnx.py --weights outputs/coco/last.pth
python scripts/benchmark_yolo26.py --source data/coco/val2017 --device 0

```

학습 진입점은 torchvision의 Faster R-CNN RPN/RoI 손실 함수를 사용하며 `last.pth`를 기록합니다. 학습 재개(resume) 기능이나 검증 AP 보고 기능은 아직 지원하지 않습니다. ONNX 내보내기는 고정 크기(fixed-shape)의 전체 탐지 경로를 지원합니다. 타겟 TensorRT 버전에 대한 RoIAlign 및 NMS 지원 검증은 [deploy/tensorrt/README.md](https://www.google.com/search?q=deploy/tensorrt/README.md)에 설명된 절차를 따르십시오.

YOLO26 스크립트는 JSON 형태의 타이밍 보고서를 출력합니다. Faster R-CNN 벤치마크 스크립트는 두 PyTorch 모델에 대해 배치-1 p50/p95 타이밍을 보고합니다. 두 스크립트 모두 기본적으로 640 픽셀 입력을 사용합니다. PyTorch 베이스라인 수치는 프레임워크 지연 시간(framework latency)을 의미하며, 최종 T4 하드웨어 비교 시에는 지원되는 범위 내에서 세 모델 모두 동일한 TensorRT 정밀도와 측정 범위를 적용하여 테스트를 반복해야 합니다.

## 설계 참고 사항

* 표준 앵커 기반 RPN 및 2단(two-stage) RoI 탐지 방식을 유지합니다.
* 추론 시 Light R-CNN의 RPN 제안(proposal) 개수는 300개로 제한됩니다(기본 Faster R-CNN은 표준 설정을 유지).
* BiFPN 및 깊이별 분리 합성곱(separable convolutions)을 적용하여 다중 스케일 특징 병합(feature-fusion) 연산 비용을 감소시킵니다.
* FPN 특징은 RoIAlign 단계 이전에 10채널로 투영되며, RoI 박스 헤드에는 전역 평균 풀링(global average pooling)이 사용됩니다.
* 지식 증류 기본 요소(Distillation primitives)는 `src/light_rcnn/distillation.py`에 작성되어 있으나, 기본 Faster R-CNN 교사 모델과 학생 모델의 제안(proposal) 정렬 로직은 아직 학습 파이프라인에 연결되어야 하는 상태입니다.