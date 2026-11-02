# 내 버전: 네 모델 공통 비교 도구

기존 팀원 코드와 별도로 실행하는 비교 도구입니다. 기존 학습·평가·벤치마크·대시보드 파일은 수정하지 않습니다. Lite R-CNN의 모델 생성 함수와 체크포인트 형식만 재사용합니다.

실행 파일은 `scripts/run_my_comparison.py`, 전용 설정은 `configs/my_comparison.yaml`, 구현은 `src/light_rcnn/comparison/`입니다. 아래 명령은 저장소 루트에서 실행합니다. 경로 옵션은 다른 위치에서 실행해도 저장소 루트를 기준으로 해석합니다.

## 비교 대상

| 설정 키 | 모델 | 가중치 |
|---|---|---|
| `lite_rcnn` | 팀의 Lite R-CNN | v2의 `best.pth`와 저장된 모델 설정 |
| `faster_rcnn` | torchvision Faster R-CNN ResNet50-FPN | `COCO_V1` |
| `mobilenet_faster_rcnn` | torchvision Faster R-CNN MobileNetV3-Large-FPN | `COCO_V1`; 320 전용 모델이 아님 |
| `yolo26` | Ultralytics YOLO26 nano | 로컬 `yolo26n.pt` |

MobileNet 비교군은 “기존 경량 Faster R-CNN 대비 우리 모델이 어떤 위치인지” 보여줍니다. 각 모델의 학습 과정·proposal 수·세부 구성이 다르므로 BiFPN 하나의 인과 효과를 입증하는 ablation은 아닙니다.

## 먼저 준비할 것

기존 CUDA 지원 PyTorch 환경을 활성화하고 추가 의존성을 설치합니다. CUDA용 torch/torchvision 설치는 해당 GPU 환경에 맞춰 기존 설정을 사용하십시오.

```powershell
python -m pip install -e ".[train,compare-yolo,dev]"
```

필요한 파일:

```text
data/coco/images/val2017/                         # 전체 5,000장
data/coco/annotations/instances_val2017.json      # 원본 COCO annotation
outputs/coco_4090_v2/best.pth                     # 팀원에게 받을 학습된 체크포인트
yolo26n.pt                                      # 공식 COCO 사전학습 YOLO26 nano
```

COCO 다운로드 방법은 기존 README의 데이터 준비 안내를 참고합니다. `best.pth`는 `model`, `config`를 포함하는 기존 학습 체크포인트여야 합니다. raw state_dict만 있는 파일은 받지 않습니다. 다른 위치에 있으면 `--light-weights`와 `--yolo-weights`로 지정하십시오.

YOLO 가중치가 없다면 인터넷 연결된 환경에서 먼저 한 번 준비할 수 있습니다.

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

ResNet/MobileNet COCO 가중치는 torchvision이 최초 실행 시 다운로드하고 캐시합니다. 오프라인 실행에는 해당 캐시도 먼저 준비해야 합니다. Lite/YOLO 파일이 없으면 측정 전 중단하며 초기화된 모델로 성능표를 만들지 않습니다.

## 바로 실행

전체 COCO 정확도와 64개 공통 이미지의 지연 시간을 네 모델에서 측정합니다.

```powershell
python scripts/run_my_comparison.py --device cuda:0
```

기본값은 입력 640, batch 1, FP32, 공통 정사각형 패딩, 예열 50회, 측정 500회입니다. TF32는 끕니다. 네 모델은 메모리에 하나씩 올리고 순서대로 평가합니다. 전체 학습은 실행하지 않습니다.

먼저 16장만 확인하려면:

```powershell
python scripts/run_my_comparison.py --device cuda:0 --limit 16 --image-count 8 --warmup 5 --iterations 16
```

GPU 없는 환경에서는 같은 명령의 `--device cpu`로 파이프라인을 확인할 수 있습니다. 부분 데이터나 CPU 결과를 전체 COCO/T4 성능으로 보고하지 마십시오.

일부 모델만 비교할 수도 있습니다.

```powershell
python scripts/run_my_comparison.py --device cuda:0 --models lite_rcnn mobilenet_faster_rcnn
```

정확도만 또는 지연 시간만 측정하려면 `--mode accuracy`, `--mode latency`를 추가합니다. 두 모드 모두 같은 COCO 이미지 선택을 위해 annotation이 필요합니다. seed는 기본 42이며 `--seed`로 바꿀 수 있습니다.

## 어떤 조건을 같게 맞추는가

**입력과 출력 경계:** 모든 모델은 디코딩된 CPU RGB uint8 배열에서 시작합니다. 정사각형 패딩·RGB/텐서 변환·정규화·CPU→GPU 전송·검출·모델 고유 후처리·GPU→CPU 전송·원본 좌표 복원·COCO label 변환을 측정에 포함합니다. 이미지 파일 디코딩, 모델 로딩, JSON 저장은 제외합니다. CPU 텐서를 GPU에 미리 옮겨놓는 모델은 없습니다. CUDA는 각 측정 직전과 직후에 선택한 장치로 synchronize합니다.

**공통 이미지:** 평가 이미지 ID는 annotation에서 한 번 선택해 모든 모델이 공유합니다. `--limit 0`은 전체, 양수는 같은 seed로 고정된 무작위 표본입니다. 속도용 이미지도 평가 목록에서 같은 seed로 선택하며, 모든 모델이 같은 순서로 반복 측정합니다. 반복 수는 최소한 선택된 속도용 이미지 수 이상이어야 합니다. 전체 evaluation 이미지는 하나씩 읽고 속도용 표본만 CPU에 미리 디코딩합니다.

**공통 평가기:** 네 모델의 예측을 원래 COCO category ID와 원본 좌표의 xywh로 저장합니다. YOLO의 0–79 class index는 이름을 검증해 원래 sparse ID 1–90으로 바꿉니다. 원본 annotation의 crowd 영역은 그대로 보존하고, 모든 모델에 공식 `pycocotools.COCOeval`과 `maxDets=[1,10,100]`을 적용합니다. YOLO 라벨 파일로 annotation을 재작성하거나 Ultralytics의 별도 mAP를 사용하지 않습니다. AP/AR은 JSON에서 0–1, 비교표에서는 0–100입니다. GT가 없는 크기 구간은 0으로 꾸미지 않고 `null`/`—`로 표시합니다.

**threshold와 후처리:** score threshold 0.001, NMS IoU 0.5, 최대 검출 100개입니다. `predict.py`의 별도 부위 포함 억제는 사용하지 않습니다. YOLO26의 native NMS-free head는 유지하고 실제 모드를 모델 JSON에 기록합니다. RPN 후보 수는 각 모델의 고유 설정을 유지하고 기록합니다. FP16/INT8·TensorRT는 이번 도구의 범위에 포함되지 않습니다.

## 입력 크기와 기존 AP의 관계

기본 `square` 모드에서는 네 모델 모두 **같은 PIL bilinear resize와 114색 중앙 패딩을 수행해 같은 640×640 RGB 이미지**를 받습니다. 이 처리 비용도 각 추론 시간에 포함됩니다. 검출 박스는 실제 가로·세로 resize 비율과 padding을 반영해 원본 좌표로 되돌립니다. 프레임워크의 모델 입력 텐서 크기를 별도로 관찰해 각 속도용 이미지에 기록합니다. 관찰용 hook은 측정 전에 제거합니다.

v2 학습 검증은 정사각형 입력을 미리 만들지 않았고 기본 score threshold도 다릅니다. 따라서 이 비교의 AP가 기존 28.82와 같다고 가정하면 안 됩니다. 기존 입력 방식과의 차이를 확인하려면 같은 이미지에서 `native`도 별도로 실행하십시오.

```powershell
python scripts/run_my_comparison.py --device cuda:0 --geometry native
```

`native`는 torchvision의 max-side resize/stride padding과 YOLO의 `rect=False` letterbox를 각각 사용합니다. 이때 명목 크기는 640이어도 실제 텐서 면적은 다를 수 있습니다. `square`와 `native`의 latency를 같은 조건인 것처럼 섞지 마십시오.

## 결과와 재현

출력은 `outputs/my_comparison/<UTC 실행시각>/` 아래에 저장됩니다. `--output-dir outputs/my_comparison/run_name`으로 지정할 수 있으며 기존 디렉터리는 덮어쓰지 않습니다.

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 네 모델을 한 표로 정리한 한국어 결과 및 측정 조건 |
| `comparison.csv` | AP·AP50·AP75·크기별 AP·AR@100·파라미터·p50/p95; Excel에서 열 수 있음 |
| `comparison.json` | 전체 결과, 조건과 비교표 |
| `manifest.json` | 공통 평가/속도 이미지 ID, annotation SHA-256, 설정·라이브러리·장치·코드 해시 |
| `<model>.json` | 가중치 SHA-256, 파라미터 수, proposal/후처리 조건, 실제 입력 크기와 개별 지연 시간 |
| `<model>_predictions.json` | 공식 COCO 형식의 예측; 정확도 측정 시 생성 |

파라미터 수는 최초 모델의 전체 파라미터 수이며, YOLO의 추론 시 Conv/BN fusion 이전 기준입니다. FLOPs·tensor-only latency·단계별 NMS 시간은 산출하지 않습니다. p95는 NumPy linear interpolation으로 계산하고 원시 표본을 함께 보존합니다. shape 관찰·모델 setup은 예열 이전에 별도로 실행하므로 cold-start 시간은 아닙니다.

이전 실행과 **동일한 평가·속도 이미지 ID**를 재사용하려면:

```powershell
python scripts/run_my_comparison.py --device cuda:0 --image-ids outputs/my_comparison/이전실행/manifest.json
```

원래 annotation 해시가 다르면 중단합니다. `--image-ids`에는 JSON 정수 ID 목록도 전달할 수 있습니다. manifest 재사용은 이미지 목록을 고정하는 기능이며, 크기·geometry·threshold 등 나머지 조건은 현재 설정이므로 `manifest.json`을 함께 확인하십시오. 이전 manifest를 사용하면 그 manifest의 속도용 이미지 목록을 우선합니다.

모델 실행이 실패하면 중단하고 실패 이유와 완료된 모델의 결과를 보존합니다. 실패·미측정 값을 0 AP 또는 0ms로 표시하지 않습니다. 실제 GPU 수치가 필요한 경우 다른 GPU 작업을 종료한 뒤 같은 장치에서 반복 실행하고 변동도 확인하십시오.

## 개발 검증

```powershell
python -m pytest tests/test_my_comparison.py -q
```

테스트는 sparse category 매핑, 정사각형 패딩/좌표 복원, crowd 무시, 빈 예측, 공통 타이밍 경계, 실패 기록, 이미지 목록 재사용을 검증합니다. 네 모델의 실제 CPU 추론·입력 크기 경로도 **다운로드 없는 초기화 가중치**로 검사합니다. 이 테스트 결과는 학습된 모델의 성능 측정이 아닙니다.

## GitHub에서 실행

`.github/workflows/my-comparison.yml`이 GitHub Actions의 Ubuntu CPU 러너에서 실행합니다. 이 파일과 비교 도구가 포함된 PR을 열거나 갱신하면 자동 실행하고, main 반영 후에는 Actions의 **Personal comparison CPU checks → Run workflow**로도 실행할 수 있습니다. 기존 학습 코드를 바꾸거나 학습을 새로 시작할 필요는 없습니다.

Python 3.12, CPU PyTorch 2.8.0/torchvision 0.23.0, Ultralytics 8.4.177 환경에서 새 비교 테스트와 기존 학습·재개·증류 테스트를 실행합니다. 합성 COCO 데이터로 실제 Lite 체크포인트 로딩 → 예측 → 공식 COCOeval → latency → 보고서 생성까지 확인합니다. 네 모델 각각의 실제 CPU 추론 경로도 검사합니다.

실행 후 해당 Actions run의 **Summary**에서 테스트 수와 실패 여부를 확인하고, **Artifacts**에서 `comparison-cpu-checks-synthetic-<run id>`를 내려받을 수 있습니다. JUnit 결과, 설치 버전 목록, 초기화 가중치로 실행한 Lite smoke-test의 Markdown/CSV/JSON이 들어 있습니다. 이 artifact의 AP와 latency는 실제 학습 모델의 성능을 나타내지 않습니다.

일반 `ubuntu-24.04` 러너에는 이번 워크플로가 사용할 CUDA GPU가 없습니다. 실제 GPU 비교에는 GPU 러너를 별도로 구성하고 원본 COCO val2017, v2 `best.pth`, `yolo26n.pt`를 준비한 뒤 본문의 전체 비교 명령을 실행해야 합니다. 프로젝트의 학습된 체크포인트가 현재 GitHub Release로 배포된 것은 아니므로, CPU 워크플로가 자동으로 그 가중치를 받아 전체 COCO를 비교하지는 않습니다.
