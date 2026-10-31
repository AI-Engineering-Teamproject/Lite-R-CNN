"""Draw Korean, code-based flow diagrams for Lite R-CNN v2 and v3."""

from __future__ import annotations

import argparse
from pathlib import Path


BG = "#111316"
HEADER = "#D8D8D8"
BLOCK = "#B9E5A9"
TEXT = "#15191B"
WHITE = "#F1F3F4"
MUTED = "#C2C8CC"


def draw(version: int, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    is_v3 = version == 3
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 25), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 27)
    ax.axis("off")
    ax.text(0.45, 26.5, f"Lite R-CNN v{version}  |  계층별 추론 순서",
            fontsize=20, color=WHITE, fontweight="bold", va="center")
    ax.text(0.48, 26.08,
            "위에서 아래로 따라가세요. RPN이 만든 후보 상자와 특징 지도가 RoI 단계에서 합쳐집니다.",
            fontsize=10.5, color=MUTED, va="center")
    ax.text(0.48, 25.82,
            "학생 계층 구성은 같습니다. 학습 중 가중치를 갱신하고, 추론은 학습된 가중치를 씁니다. 학습 손실·v3 교사·RPN 후보 수는 모드별 차이입니다.",
            fontsize=8.4, color="#B9E5A9", va="center")

    def rounded(x, y, w, h, color, radius=0.1, edge=None, zorder=2):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.035,rounding_size={radius}",
            facecolor=color, edgecolor=edge or color, linewidth=1.0, zorder=zorder,
        )
        ax.add_patch(patch)

    def block(x, y, w, h, label, *, header=False, fs=9.2, color=None):
        face = color or (HEADER if header else BLOCK)
        rounded(x, y, w, h, face, radius=0.11 if header else 0.09)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=TEXT, fontweight="bold" if header else "normal",
                linespacing=1.25, zorder=3)

    def arrow(start, end, *, color=WHITE, width=1.5, scale=12):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=1,
        ))

    def vertical(label, y, h=0.62, *, fs=8.8, color=None, next_y=None):
        nonlocal flow_x
        block(flow_x - flow_w / 2, y, flow_w, h, label, fs=fs, color=color)
        if next_y is not None:
            arrow((flow_x, y - 0.015), (flow_x, next_y + h + 0.015))

    # Image tensor and every MobileNetV3 feature module are listed in source order.
    flow_x, flow_w = 7.0, 10.2
    block(flow_x - 2.5, 25.05, 5.0, 0.62, "입력 이미지  |  3 × 640 × 640", fs=10)
    arrow((flow_x, 25.02), (flow_x, 24.78))
    block(1.0, 24.25, 12.0, 0.46,
          "BACKBONE  |  MobileNetV3-Large  (backbone.body.features)", header=True, fs=10)
    arrow((flow_x, 24.23), (flow_x, 24.02))

    tap_labels = (
        {6: "출력 C3 · 40채널 · stride 8", 12: "출력 C4 · 112채널 · stride 16", 16: "출력 C5 · 960채널 · stride 32"}
        if not is_v3 else
        {3: "출력 C2 · 24채널 · stride 4", 6: "출력 C3 · 40채널 · stride 8",
         12: "출력 C4 · 112채널 · stride 16", 16: "출력 C5 · 960채널 · stride 32"}
    )
    types = {i: "InvertedResidual" for i in range(1, 16)}
    types[0] = "Conv2dNormActivation"
    types[16] = "Conv2dNormActivation"
    layer_y, layer_h, layer_gap = 23.55, 0.35, 0.075
    for i in range(17):
        label = f"features[{i}]  ·  {types[i]}"
        if i in tap_labels:
            label += f"     →     {tap_labels[i]}"
        block(2.0, layer_y, 10.0, layer_h, label,
              fs=8.0 if i in tap_labels else 8.8,
              color="#C6EAB9" if i in tap_labels else BLOCK)
        if i < 16:
            arrow((flow_x, layer_y - 0.012), (flow_x, layer_y - layer_gap + 0.008), scale=9)
        layer_y -= layer_h + layer_gap

    last_layer_bottom = layer_y + layer_h + layer_gap
    tap_box_y = 15.5
    arrow((flow_x, last_layer_bottom - 0.02), (flow_x, tap_box_y + 0.68 + 0.03))
    block(2.0, tap_box_y, 10.0, 0.68,
          "추출한 다중 스케일 특징 지도  |  v2: C3–C5  ·  v3: C2–C5", fs=9.0,
          color="#C6EAB9")

    # Pyramid neck: exact operation order differs in v2 and v3.
    neck_header_y = 14.65
    arrow((flow_x, tap_box_y - 0.015), (flow_x, neck_header_y + 0.48 + 0.02))
    neck_title = "NECK  |  " + ("MobileBiFPNBackbone" if not is_v3 else "MobileFPNBackboneV3")
    block(1.0, neck_header_y, 12.0, 0.48, neck_title, header=True, fs=10)

    neck_y = 13.88
    if not is_v3:
        neck_ops = [
            ("1×1 Conv + BN + SiLU  |  C3/C4/C5 채널을 각각 96으로 맞춤\n40 / 112 / 960 → 96채널", 0.62, 8.4),
            ("BiFPNLayer ×2  |  각 셀에서 top-down 가중 합산 후\nnearest upsample + separable 3×3 convolution", 0.72, 8.4),
            ("같은 BiFPN 셀에서 bottom-up 가중 합산\nmax-pool + separable 3×3 convolution", 0.65, 8.5),
            ("P3 / P4 / P5 특징 지도  |  각 96채널\n80×80 / 40×40 / 20×20  (입력 640×640 기준)", 0.68, 8.5),
        ]
    else:
        neck_ops = [
            ("1×1 Conv + BN + ReLU  |  C2/C3/C4/C5 채널을 각각 64로 맞춤\n24 / 40 / 112 / 960 → 64채널", 0.62, 8.2),
            ("LiteFeaturePyramid  |  top-down 경로\nnearest upsample 후 같은 크기 특징 지도를 element-wise add", 0.68, 8.4),
            ("P2 / P3 / P4 / P5 출력 convolution\ndepthwise 3×3 → pointwise 1×1 → BN + SiLU", 0.65, 8.4),
            ("P6 생성 후 P7 생성  |  각 단계 kernel 1, stride 2 max-pool\n지도 크기: 160 / 80 / 40 / 20 / 10 / 5", 0.68, 8.4),
        ]
    for idx, (label, h, fs) in enumerate(neck_ops):
        block(1.2, neck_y, 11.6, h, label, fs=fs,
              color="#C6EAB9" if idx == len(neck_ops) - 1 else BLOCK)
        if idx < len(neck_ops) - 1:
            next_h = neck_ops[idx + 1][1]
            next_y = neck_y - 0.12 - next_h
            arrow((flow_x, neck_y - 0.015), (flow_x, next_y + next_h + 0.015), scale=10)
            neck_y = next_y
    neck_output_bottom = neck_y

    # The neck output splits: RPN generates boxes; feature maps bypass it to RoIAlign.
    branch_header_y = neck_output_bottom - 0.85
    arrow((flow_x, neck_output_bottom - 0.02), (3.55, branch_header_y + 0.48 + 0.04))
    arrow((flow_x, neck_output_bottom - 0.02), (10.45, branch_header_y + 0.48 + 0.04))
    block(0.55, branch_header_y, 6.0, 0.48, "RPN  |  후보 상자 생성 경로", header=True, fs=9.5)
    block(7.45, branch_header_y, 6.0, 0.48, "RoI 입력  |  특징 지도 전달 경로", header=True, fs=9.5)

    if not is_v3:
        anchor_text = "앵커 크기 P3/P4/P5: [16,32,64] / [64,128,256] / [128,256,512]\n종횡비 0.5/1/2  →  위치별 9개 앵커"
        roi_map_text = "P3–P5 특징 지도 입력\n각 지도의 1×1 Conv + BN + ReLU: 96 → 64채널"
    else:
        anchor_text = "앵커 크기 P2–P7: 8 / 16 / 32 / 64 / 128 / 256\n종횡비 0.5/1/2  →  위치별 3개 앵커"
        roi_map_text = "P2–P5 특징 지도 입력 (각 64채널)\nP6/P7은 RPN 전용이며 RoIAlign에는 전달하지 않음"

    left_x, left_w = 0.55, 6.0
    rpn_blocks = [
        (anchor_text, 0.76, 7.7),
        ("RPNHead  |  공유 3×3 convolution\n각 앵커의 객체성 점수와 박스 보정값 예측", 0.70, 8.3),
        ("후보 박스 생성  |  anchor decode → 이미지 경계로 자르기\n작은 박스 제거 → proposal NMS → 상위 후보 선택", 0.80, 8.0),
        ("후보 수 제한\n학습: 사전 2,000 → 사후 1,000  |  추론: 1,000 → 300", 0.68, 8.1),
    ]
    rpn_y = branch_header_y - 0.18 - rpn_blocks[0][1]
    arrow((left_x + left_w / 2, branch_header_y - 0.015),
          (left_x + left_w / 2, rpn_y + rpn_blocks[0][1] + 0.015))
    current_y = rpn_y
    for idx, (label, h, fs) in enumerate(rpn_blocks):
        block(left_x, current_y, left_w, h, label, fs=fs)
        if idx < len(rpn_blocks) - 1:
            next_h = rpn_blocks[idx + 1][1]
            next_y = current_y - 0.17 - next_h
            arrow((left_x + left_w / 2, current_y - 0.015),
                  (left_x + left_w / 2, next_y + next_h + 0.015), scale=10)
            current_y = next_y
    proposals_bottom = current_y

    right_x, right_w = 7.45, 6.0
    map_h = 1.28
    map_y = branch_header_y - 0.15 - map_h
    block(right_x, map_y, right_w, 1.28, roi_map_text, fs=8.2)
    arrow((right_x + right_w / 2, branch_header_y - 0.015),
          (right_x + right_w / 2, map_y + map_h + 0.015))

    # Merge both inputs at RoIAlign, then follow the compact box head in execution order.
    roi_y = branch_header_y - 4.55
    roi_w, roi_h, roi_x = 7.0, 0.72, 3.5
    arrow((left_x + left_w / 2, proposals_bottom - 0.015),
          (roi_x + 1.15, roi_y + roi_h + 0.04), color="#FFFFFF", width=1.7, scale=13)
    arrow((right_x + right_w / 2, map_y - 0.02),
          (roi_x + roi_w - 1.15, roi_y + roi_h + 0.04), color="#8CCB76", width=1.7, scale=13)
    block(roi_x, roi_y, roi_w, roi_h,
          "두 입력을 결합  |  MultiScaleRoIAlign · RoIAlign 7×7 · sampling ratio 2",
          fs=8.6, color="#C6EAB9")

    head1_y = roi_y - 0.92
    arrow((flow_x, roi_y - 0.02), (flow_x, head1_y + 0.62 + 0.04), scale=11)
    if not is_v3:
        block(2.0, head1_y, 10.0, 0.62,
              "LightRoIHead  |  AdaptiveAvgPool 7×7 → flatten 3,136", fs=9.0)
        head2_y = head1_y - 0.78
        arrow((flow_x, head1_y - 0.02), (flow_x, head2_y + 0.58 + 0.04), scale=11)
        block(2.0, head2_y, 10.0, 0.58, "Linear 3,136 → 256  +  ReLU", fs=9.0)
        predictor_y = head2_y - 0.82
    else:
        block(2.0, head1_y, 10.0, 0.62,
              "LightRoIHead  |  1×1 Conv로 채널 축소  64 → 32", fs=9.0)
        head2_y = head1_y - 0.78
        arrow((flow_x, head1_y - 0.02), (flow_x, head2_y + 0.68 + 0.04), scale=11)
        block(2.0, head2_y, 10.0, 0.68,
              "AdaptiveAvgPool 7×7 → flatten 1,568 → Linear 1,568 → 256 + ReLU", fs=8.8)
        predictor_y = head2_y - 0.9

    arrow((flow_x, head2_y - 0.02), (flow_x, predictor_y + 0.68 + 0.04), scale=11)
    block(2.0, predictor_y, 10.0, 0.68,
          "FastRCNNPredictor  |  분류 91개 점수 + 박스 보정값 364개 (91×4)", fs=9.0)
    post_y = predictor_y - 0.9
    arrow((flow_x, predictor_y - 0.02), (flow_x, post_y + 0.68 + 0.04), scale=11)
    block(2.0, post_y, 10.0, 0.68,
          "후처리  |  score ≥ 0.05 → 클래스별 NMS (IoU 0.50) → 이미지당 최대 100개", fs=8.8)
    output_y = post_y - 0.88
    arrow((flow_x, post_y - 0.02), (flow_x, output_y + 0.56 + 0.04), scale=11)
    block(4.2, output_y, 5.6, 0.56, "최종 탐지 결과  |  클래스 + 점수 + 박스", fs=9.2,
          color="#C6EAB9")

    if is_v3:
        footer = "학습 중에만 사용: 고정 Faster R-CNN ResNet50-FPN 교사 → 정규화 feature Smooth L1 × 0.5 + RoI-logit KL × 1.0 (T=2). 추론에는 교사 모델이 없음."
    else:
        footer = "v2 학습 손실: RPN 객체성 + RPN 박스 회귀 + RoI 분류 + RoI 박스 회귀. 저장된 v2 설정에서 증류는 비활성화."
    footer_y = max(0.18, output_y - 0.92)
    block(0.55, footer_y, 12.9, 0.58, footer, fs=8.1, header=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / f"lite_rcnn_v{version}_architecture.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=BG)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/figures", help="Directory for PNG/SVG diagrams")
    args = parser.parse_args()
    for version in (2, 3):
        for path in draw(version, Path(args.output_dir)):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
