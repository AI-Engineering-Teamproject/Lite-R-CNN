# 학습·재개·지식 증류

저장소 루트에서 가상환경을 활성화한 후 실행합니다. CUDA에 맞는 torch/torchvision과 `pip install -e ".[train,dev]"`가 필요합니다. 기본 설정은 COCO 원래 category ID(빈 번호 포함)와 background를 합쳐 `num_classes: 91`을 사용합니다.

## 기본 학습

```bash
python scripts/train.py --config configs/coco.yaml
```

기본 장치는 CUDA입니다. CPU 점검은 `--device cpu`로 명시합니다. 데이터 경로와 모델 폭, batch size, worker 수는 `configs/coco.yaml`에서 설정합니다. T4 메모리가 부족하면 새 실행의 batch size를 낮추십시오. 증류는 교사 특징까지 보관하므로 메모리를 더 사용합니다.

기본적으로 매 epoch 전체 validation split에서 공식 COCOeval을 실행합니다. `train.validate_every` 또는 `--validate-every`로 간격을 바꿀 수 있으며 마지막 epoch에는 항상 검증합니다. 검증은 FP32, 추론은 batch=1입니다. AP는 0~1 단위로 저장합니다.

`outputs/coco/`에는 다음 파일이 생깁니다.

| 파일 | 용도 |
| --- | --- |
| `last.pth` | 마지막 완료 epoch의 학습 재개 |
| `best.pth` | 검증 AP@[.50:.95]가 가장 높았던 모델; 동점이면 먼저 나온 모델 유지 |
| `history.jsonl` | epoch별 개별 손실·총손실·학습률·검증 지표·best epoch |
| `config.yaml` | 실행에 사용한 설정 |

체크포인트에는 학생 모델, optimizer, scheduler, AMP scaler, 완료 epoch, update 수, 최적 AP, 설정 및 Python/NumPy/PyTorch/CUDA 난수 상태가 들어갑니다. 증류 학습은 학습용 adapter 상태도 보관합니다. 저장은 임시 파일을 쓴 뒤 교체합니다. epoch 도중 중단되면 마지막 완료 epoch부터 다시 시작합니다.

학습률은 `warmup_steps` 동안 선형 warmup하고 `lr_milestones` epoch 경계에서 `lr_gamma`를 곱합니다. gradient clipping과 비정상 손실 검사도 적용됩니다.

## 학습 재개

```bash
python scripts/train.py --resume outputs/coco/last.pth
# 총 60 epoch까지 연장: 추가 60 epoch라는 뜻이 아닙니다.
python scripts/train.py --resume outputs/coco/last.pth --epochs 60
```

기본적으로 체크포인트의 설정을 그대로 사용합니다. 모델·증류·데이터·batch size·검증 subset·학습률 설정을 바꾸는 재개는 거부합니다. 다른 실험은 새 output directory와 `--init-weights`로 시작합니다. 기존 초기 버전의 `last.pth`에는 완전한 재개 상태가 없으므로 `--init-weights`로 가져옵니다.

재개용 체크포인트는 일반적으로 `last.pth`를 사용합니다. 학습 이력은 완료 epoch까지 유지합니다. `--output-dir`로 재개 위치를 바꾸면 기존 `best.pth`도 복사합니다. 로컬 교사 체크포인트를 사용하는 증류 재개에서는 같은 교사 파일을 유지해야 합니다. CPU 통합 테스트에서는 연속 2 epoch와 1 epoch 후 재개한 모델 가중치가 정확히 일치하는지 확인합니다. GPU에서 비결정적 연산까지 동일함을 보장하는 것은 아닙니다.

## 지식 증류

처음부터 증류하거나, 지도학습 모델을 초기값으로 증류할 수 있습니다.

```bash
python scripts/train.py --config configs/coco.yaml --distill --output-dir outputs/coco_kd
# 권장 비교 실험: 기본 학습 결과에서 새 증류 실행 시작
python scripts/train.py --config configs/coco.yaml --init-weights outputs/coco/best.pth --distill --output-dir outputs/coco_kd
python scripts/train.py --resume outputs/coco_kd/last.pth --epochs 60
```

`distillation.enabled: true`로 설정해도 동일합니다. 기본 교사는 torchvision Faster R-CNN ResNet50-FPN COCO_V1입니다. 최초 실행 시 가중치를 다운로드하며, `distillation.teacher.checkpoint`에 로컬 ResNet50-FPN checkpoint 경로도 지정할 수 있습니다. 기본 COCO 교사는 `num_classes=91`과 원래 COCO ID를 요구합니다. 사용자 교사는 학생과 클래스 개수뿐 아니라 클래스 의미와 순서도 같아야 합니다.

구현 흐름은 다음과 같습니다.

1. 학생 transform으로 이미지와 정답을 한 번 resize·normalize·pad합니다.
2. 학생은 표준 RPN 및 sampled RoI 탐지 손실을 계산합니다.
3. 교사는 같은 변환 tensor를 받아 eval/no-grad 상태로 특징을 계산합니다.
4. 학생 특징 `0/1/2`와 교사 FPN 특징 `1/2/3`을 stride 8/16/32로 대응시킵니다. 학습 가능한 1×1 adapter로 학생 채널을 교사 채널에 맞춘 후 정규화 특징의 Smooth L1 손실을 계산합니다.
5. 학생이 선택한 동일한 sampled RoI와 이미지 크기를 교사 RoI head에 전달해 클래스 logits의 temperature-scaled KL 손실을 계산합니다.

전체 손실은 `탐지 지도학습 손실 + feature_weight × 특징 손실 + roi_kl_weight × RoI KL 손실`입니다. 두 증류 가중치의 기본값은 1.0, temperature는 2.0이며 실험으로 조정해야 합니다. `history.jsonl`의 `loss_kd_feature`, `loss_kd_roi`는 가중치를 곱한 값입니다. 교사와 adapter는 학생 추론 경로에 포함되지 않습니다. 박스 회귀 증류는 구현 범위에 포함하지 않습니다.

## 작은 데이터로 점검

실제 COCO 데이터가 준비된 환경에서 먼저 실행합니다.

```bash
python scripts/train.py --config configs/coco.yaml --epochs 2 --train-limit 16 --val-limit 16 --output-dir outputs/smoke
```

이 실행의 `best.pth`는 해당 16개 검증 이미지 기준입니다. 전체 COCO 성능으로 보고하면 안 됩니다. subset 변경은 새 실행으로 시작합니다. 데이터 없이 실행하는 통합 테스트는 합성 COCO 이미지를 임시 생성하며, 교사도 다운로드 없는 무작위 가중치 fixture를 사용합니다.

```bash
python -m pytest -q
```

## 최적 모델 평가·인계

```bash
python scripts/evaluate_coco.py --model light_rcnn --weights outputs/coco/best.pth --device cuda
python scripts/export_onnx.py --weights outputs/coco/best.pth
```

평가·벤치마크·export는 학생 구조를 체크포인트 config에서 복원합니다. 평가의 `--config`는 다른 데이터 경로를 지정할 때 사용할 수 있습니다. 벤치마크의 `--size`는 기본 640이며, export의 크기 기본값은 저장된 설정입니다. 최종 인계 시 `best.pth`, config, 코드 커밋과 학습 이력을 함께 전달하십시오. 실제 전체 COCO 학습, 증류의 정확도 개선, T4 지연 시간 및 TensorRT 변환은 별도 확인이 필요합니다.
