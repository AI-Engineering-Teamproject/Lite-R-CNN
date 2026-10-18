# 학습·재개·지식 증류

저장소 루트에서 가상환경을 활성화한 후 실행합니다. CUDA에 맞는 torch/torchvision과 `pip install -e ".[train,dev,huggingface]"`가 필요합니다. 기본 설정은 COCO 원래 category ID(빈 번호 포함)와 background를 합쳐 `num_classes: 91`을 사용합니다.

## Hugging Face에서 데이터 준비

압축 해제가 중단되어도 같은 준비 명령을 다시 실행하면 정상 추출된 파일은 건너뛰고, ZIP 기준 누락되거나 잘린 이미지만 복구합니다.

프로젝트에서 사용하는 공개 데이터 저장소는 [Hyeon1222/COCO](https://huggingface.co/datasets/Hyeon1222/COCO)이며, COCO 2017 train/val 이미지 ZIP과 annotation ZIP을 제공합니다. 아래 명령은 이를 `configs/coco.yaml`과 `configs/coco_cpu.yaml`이 기대하는 `data/coco/images/` 및 `data/coco/annotations/` 경로에 풀어 둡니다.

```powershell
python scripts/prepare_hf_coco.py
```

전체 ZIP은 약 20 GB이고 압축 해제 공간이 별도로 필요합니다. 검증만 먼저 준비하려면 `python scripts/prepare_hf_coco.py --splits annotations val`을 사용합니다. ZIP 캐시는 기본적으로 `data/hf_cache/`에 저장되며, 추출 후 삭제해도 됩니다. 인증이 필요한 저장소라면 `huggingface-cli login`으로 먼저 로그인하십시오.

## RTX 4090에서 전체 학습 시작

현재 개발 환경에서 전체 COCO CPU 학습은 약 70일 이상으로 추정되어, 전체 학습은 RTX 4090 같은 CUDA GPU 환경에서 실행합니다. 새 GPU 환경에서 저장소를 받은 뒤 저장소 루트에서 CUDA에 맞는 PyTorch/torchvision을 설치하고 프로젝트 의존성을 설치하십시오. PyTorch CUDA wheel 설치 명령은 해당 환경의 공식 PyTorch 설치 선택기에서 OS와 CUDA 버전에 맞춰 확인합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA 지원 PyTorch/torchvision을 먼저 설치한 다음 프로젝트 의존성을 설치
python -m pip install -e ".[train,huggingface]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python scripts/prepare_hf_coco.py
python scripts/train.py --config configs/coco.yaml --device cuda --epochs 36 --output-dir outputs/coco_4090
```

학습 전 데이터 경로는 `data/coco/images/{train2017,val2017}` 및 `data/coco/annotations/instances_{train2017,val2017}.json`이어야 합니다. `configs/coco.yaml`은 640 입력, batch size 8, worker 8, AMP 사용, 매 epoch 전체 validation으로 설정되어 있습니다. RTX 4090의 VRAM이 부족하면 batch size를 낮추고 새 output directory로 시작하십시오. 중단 후 같은 설정을 이어가려면 다음 명령을 사용합니다.

```powershell
python scripts/train.py --resume outputs/coco_4090/last.pth --epochs 36
```

GPU가 있는 다른 컴퓨터로 이동할 때는 코드와 설정을 Git으로 전달하고, COCO 데이터와 체크포인트는 별도 다운로드/복사합니다. `data/`, `outputs/`, `.venv/`, 가중치 파일은 `.gitignore` 대상입니다. 현재 커밋에는 전체 COCO 학습 결과가 없으며, 기록된 AP 0 결과는 소규모 초기 GAP head 실험에 한정됩니다.

## 기본 학습

```bash
python scripts/train.py --config configs/coco.yaml
```

기본 장치는 CUDA입니다. CPU 점검은 `--device cpu`로 명시합니다. 데이터 경로와 모델 폭, batch size, worker 수는 `configs/coco.yaml`에서 설정합니다. T4 메모리가 부족하면 새 실행의 batch size를 낮추십시오. 증류는 교사 특징까지 보관하므로 메모리를 더 사용합니다.

전체 데이터 CPU 학습용 프리셋은 `configs/coco_cpu.yaml`입니다. batch size 1, worker 0, AMP off를 사용합니다. 16장 실험 처리량을 단순 환산하면 COCO train 전체 36 epoch는 CPU에서 약 70일 규모이며 실제 시간은 다를 수 있습니다. CPU에서는 증류 교사의 추가 추론 때문에 더 느리므로, 파이프라인 확인은 작은 subset으로 하고 전체 실험은 가능한 경우 GPU에서 진행하십시오.

Windows에서는 루트의 `start_lite_rcnn_research.ps1` 또는 `start_lite_rcnn_research.bat` 한 번 실행으로 데이터 누락 확인/준비, 학습, epoch별 COCO 검증, cycle별 보고서 생성을 반복합니다. 새 cycle은 최고 체크포인트에서 초기화하고 검증 AP 개선 여부에 따라 learning rate를 조절합니다. 중단하면 `outputs/research/state.json`과 현재 cycle 체크포인트를 사용해 다음 실행에서 이어갑니다. 누적 결과는 `outputs/research/RESEARCH_REPORT.md`와 `experiments.jsonl`에 기록합니다. 이 실행기는 Lite R-CNN만 학습하며 다른 모델 비교는 실행하지 않습니다. 기본 실행은 Ctrl+C로 끝낼 때까지 계속되며, 제한 실행에는 `--cycles 1 --epochs 1 --train-limit 16 --val-limit 16 --output-root outputs/research_smoke`를 사용합니다. 모델이나 데이터 설정을 바꾸는 새 연구는 새 `--output-root`에서 시작합니다.

```powershell
python scripts/train.py --config configs/coco_cpu.yaml --device cpu
python scripts/train.py --config configs/coco_cpu.yaml --device cpu --epochs 2 --train-limit 16 --val-limit 16 --output-dir outputs/coco_cpu_smoke
```

Windows CMD에서는 저장소 루트에서 다음처럼 전체 COCO CPU 학습을 시작할 수 있습니다.

```bat
.venv312\Scripts\python.exe scripts\train.py --config configs\coco_cpu.yaml --device cpu --epochs 36
```

현재 COCO 설정은 3×3 RoI pooling과 FC 128을 사용하는 공간형 head로 시작합니다. 이전 GAP checkpoint와 head 차원이 다르므로 그 checkpoint를 resume하지 말고 기본 `outputs/coco_cpu/`의 새 run으로 학습하십시오.

epoch마다 `outputs/coco_cpu/last.pth`와 AP 최고 모델 `best.pth`가 저장됩니다. 중단 뒤에는 아래 명령으로 마지막 완료 epoch부터 재개합니다.

```bat
.venv312\Scripts\python.exe scripts\train.py --resume outputs\coco_cpu\last.pth --epochs 36
```

모든 이미지는 dataset에서 RGB 3채널로 정규화합니다. `--train-limit`과 `--val-limit`을 지정한 subset 실험은 config seed로 재현 가능한 무작위 image subset을 선택합니다(0은 전체 split). 같은 seed를 쓰는 GAP/spatial 실험은 같은 train·validation image를 사용합니다. 학습 시작 전 네 COCO 경로를 검사하고, 누락 경로가 있으면 데이터 준비 스크립트를 안내합니다. 학습 데이터에는 확률 0.5 좌우 반전을 적용하고 box의 x 좌표도 함께 변환합니다. validation에는 augmentation을 적용하지 않습니다. COCO crowd annotation은 학습 positive box로 사용하지 않습니다.

### AP가 0으로 보일 때

작은 smoke run의 AP 0은 전체 데이터에서의 모델 성능을 뜻하지 않습니다. 특히 Lite R-CNN은 ImageNet 사전학습 backbone만 가져오며 RPN과 RoI 분류·회귀 head는 새로 학습합니다. train 200장, validation 500장, 2 epoch로는 예측이 정답과 IoU 0.5 이상으로 맞지 않아 AP가 계속 0일 수 있습니다. 이 실험에서 train loss는 5.415에서 4.183으로 내려갔으므로 optimizer는 업데이트됐지만, AP 향상은 아직 확인되지 않았습니다.

validation 결과에는 `detections`, `images_with_detections`, `max_detection_score`와 이미지별 최대 IoU 진단도 기록됩니다. `mean_best_iou_any_class`와 `mean_best_iou_same_class`를 비교하면 박스 위치와 클래스 일치 정도를 분리해서 살펴볼 수 있습니다. 이 최대 IoU 집계는 recall이나 AP가 아니며 COCO AP를 대체하지 않습니다. 작은 subset 결과는 코드 동작 점검에만 쓰고, 모델 성능은 충분한 학습 후 전체 `val2017`에서 판단하십시오.

기본적으로 매 epoch 전체 validation split에서 공식 COCOeval을 실행합니다. `train.validate_every` 또는 `--validate-every`로 간격을 바꿀 수 있으며 마지막 epoch에는 항상 검증합니다. CUDA 설정은 검증 batch 4, CPU 설정은 batch 1을 사용합니다. 평가 명령은 `--batch-size`로 덮어쓸 수 있습니다. 검증은 FP32이며 AP는 0~1 단위로 저장합니다.

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

분류 진단을 바탕으로 RoI logits만 증류하는 선택지도 있습니다. `--distill-roi-only`는 feature loss를 끄고 teacher RoI KL loss만 켭니다. student가 생성한 proposals에 teacher의 COCO class 분포를 학습하므로 localization 변경 없이 분류 증류를 비교할 수 있습니다.

```powershell
.venv312\Scripts\python.exe scripts\train.py --config configs\coco.yaml --device cuda --distill-roi-only --epochs 36 --output-dir outputs\coco_roi_kd
```

Teacher는 매 batch에서 추가 추론을 하므로 CPU에서는 supervised-only보다 훨씬 오래 걸립니다. 먼저 supervised baseline을 학습하고, GPU를 사용할 수 있을 때 별도 output directory에서 이 실험을 권장합니다. 증류 효과는 동일한 전체 `val2017` AP로 판단하십시오.

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
