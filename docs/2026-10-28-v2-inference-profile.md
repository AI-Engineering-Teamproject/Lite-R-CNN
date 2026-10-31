# Lite R-CNN v2 추론 병목 프로파일

## 측정 조건

- 체크포인트: `outputs/coco_4090_v2/best.pth` (COCO val AP 최고 epoch 25)
- 장치: NVIDIA GeForce RTX 4090, CUDA, FP32, batch 1
- 입력: `data/coco/images/val2017/000000129756.jpg`, 480×640
- 단계 계측: 15회 warmup 후 40회 추론, 각 모듈 경계에서 CUDA event 기록
- 연산자 프로파일: 별도 5회 추론을 `torch.profiler`로 기록
- 원자료: [`profile_20261028.json`](../outputs/coco_4090_v2/profile_20261028.json)

## 단계별 시간

| 단계 | CUDA 이벤트 중앙값 | 해석 |
|---|---:|---|
| MobileNetV3 body | 1.469 ms | 가장 큰 개별 모듈 구간; convolution 및 batch normalization 중심 |
| BiFPN neck | 0.652 ms | 특징 융합과 반복 separable convolution |
| RPN 전체 | 1.283 ms | 앵커별 점수/박스, proposal 정렬·선택과 NMS 포함 |
| RoI projection + RoIAlign | 0.405 ms | 세 FPN 레벨 projection 후 proposal별 특징 추출 |
| RoI FC head | 0.091 ms | 3136→256 FC 및 pooling; 전체 지연의 작은 부분 |
| 분류·박스 predictor | 0.016 ms | 분류와 박스 회귀 선형층 |
| RoI heads 전체 | 1.011 ms | RoI 단계를 포함한 합계; 위 세부 항목과 중복되므로 다시 더하지 않음 |
| 입력 transform | 0.083 ms | CUDA event 기준; 입력은 이미 480×640 |

### 실제 Conv/Linear 텐서 연산량

같은 입력의 feature map은 C3 `[1,40,60,80]`, C4 `[1,112,30,40]`, C5 `[1,960,15,20]`이며, BiFPN 이후 P3/P4/P5는 각각 `[1,96,60,80]`, `[1,96,30,40]`, `[1,96,15,20]`입니다. 아래 MAC은 `N × Hout × Wout × Cout × (Cin/groups) × Kh × Kw`로 각 Conv2d 호출을 합산했습니다.

| 구성 | Conv MAC | 설명 |
|---|---:|---|
| MobileNetV3 body | **1.303 GMAC** | 62개 Conv 호출; 앞쪽 고해상도 inverted bottleneck 확장 연산이 누적 |
| RPN head | **0.550 GMAC** | 9개 호출; 같은 3×3 96→96 Conv를 P3/P4/P5 각각에 적용하고 objectness/box 출력을 계산 |
| BiFPN neck | 0.210 GMAC | 입력 1×1 투영 및 반복 양방향 융합 |
| RoI 96→64 projections | 0.039 GMAC | P3/P4/P5를 각각 64채널로 투영 |

v2는 NMS 후 최대 300 proposals를 RoI FC에 보냅니다. 실제 Linear 입력 `[300,3136]`의 3136→256 FC와 두 예측층을 합치면 추가로 **0.276 GMAC**입니다. 따라서 FC는 Conv-only 표에 포함하지 않았지만, 측정된 직접 시간은 작았습니다. MAC은 곱셈-누산 쌍 수이며 elementwise 연산, 보간/풀링, RoIAlign, box decode, sorting 및 NMS는 이 MAC 표에 포함하지 않습니다. 1 MAC을 2 FLOPs로 세는 관례를 쓰면 MAC 합의 대략 두 배가 되지만, 여기 값은 전체 모델 FLOPs가 아니라 명시된 Conv/Linear 계층 집계입니다.

상위 단계의 CUDA 이벤트 중앙값(transform + backbone + RPN + RoI heads)을 단순 합산하면 약 4.54ms입니다. 기존 64장 순환 지연 측정의 Lite R-CNN p50 4.36ms와 대략 같은 규모입니다. 다만 단계 계측 hook 자체의 오버헤드와 단일 입력 이미지 영향이 있어 이 합은 정밀한 end-to-end 기준값이 아닙니다. `cpu_call_ms_median`은 CUDA 계산이 끝날 때까지의 동기화 시간이 아니라 Python/CPU 호출 구간이므로 CUDA 실행 시간과 혼동하지 않습니다.

## 병목 진단

1. **MobileNetV3 body가 가장 큰 단일 모듈 병목**입니다. 여기서 말하는 mobile 경량화는 ResNet 같은 큰 backbone 대비 효율적인 설계라는 뜻이지, 모든 해상도에서 총 연산량이 작다는 뜻은 아닙니다. 입력은 480×640이고, MobileNet 초반부는 240×320 또는 120×160 feature map을 처리합니다. Inverted bottleneck은 depthwise convolution 전후에 1×1 convolution으로 채널 폭을 확장·축소합니다. 예를 들어 `[1,16,240,320]→[1,64,240,320]`의 1×1 확장은 한 번에 약 78.6M MAC이며, 이런 층이 여러 stage에 걸쳐 쌓여 body 총 1.303 GMAC가 됩니다. Depthwise 연산 자체는 MAC을 줄이지만, 채널 확장 1×1 연산, 메모리 이동, 다수의 작은 커널 실행 비용은 남습니다.
2. **RPN은 MAC보다 후보 후처리까지 포함한 지연으로 해석해야 합니다.** RPN convolution 계산은 0.550 GMAC로 MobileNet body의 약 42%지만, RPN 시간은 점수화 외에도 수만 anchor 후보 중 top-k 선택, box decode/clip, 정렬과 NMS를 포함해 1.283ms입니다. PyTorch profiler에서 NMS 관련 CUDA 연산은 5회 프로파일 동안 `torchvision::nms` 1.502ms(10회), `gather_keep_from_mask` 1.372ms(10회), `topk` 0.623ms(15회)로 관찰됐습니다. 프로파일러 오버헤드가 있으므로 이 누적값은 실제 p50에 그대로 합산하지 않습니다.
3. **BiFPN은 의미 있는 보조 병목**입니다. 작은 depthwise convolution을 여러 번 실행하고 피라미드 간 보간·풀링·가중 합산을 수행합니다. 소형 텐서 다수의 launch 비용도 고려할 지점입니다.
4. **RoI FC와 predictor는 현재 우선 최적화 대상이 아닙니다.** RoIAlign/projection은 일정 비용이 있지만 FC와 predictor의 직접 시간은 각각 약 0.09ms, 0.02ms 수준이므로 이 부분만 줄여 얻을 수 있는 전체 속도 이득은 제한적입니다.

### 개선 실험 우선순위

- MobileNet 입력 해상도나 backbone 폭 변경: 속도 이득과 작은 객체 AP 하락을 함께 비교합니다.
- BiFPN 반복 2회 대 1회 ablation: 전체 val AP와 p50/p95를 함께 측정합니다.
- RPN pre/post-NMS proposal 한도 축소: RPN/RoI 속도와 COCO AP, 특히 작은 객체 AP의 변화를 측정합니다.
- 최적화마다 전체 COCO AP와 동일 장치의 latency를 재측정합니다. 단일 이미지 latency만으로 구조 변경을 채택하지 않습니다.
