"""Draw compact, code-based overview diagrams for Lite R-CNN v2 and v3."""

from __future__ import annotations

import argparse
from pathlib import Path


BG = "#111316"
GRAY = "#D8D8D8"
GREEN = "#B9E5A9"
GREEN_HI = "#C8EBB9"
INK = "#15191B"
WHITE = "#F2F4F5"


def draw(version: int, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    v3 = version == 3
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 13), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 13)
    ax.axis("off")

    ax.text(0.5, 12.55, f"Lite R-CNN v{version}  |  핵심 구조 요약",
            color=WHITE, fontsize=21, fontweight="bold", va="center")
    ax.text(0.53, 12.12,
            "주요 처리 순서와 RPN·RoI 경로의 합류만 표시했습니다.",
            color="#C2C8CC", fontsize=10, va="center")

    def box(cx, y, w, h, text, *, header=False, fs=10, color=None, weight=None):
        fill = color or (GRAY if header else GREEN)
        patch = FancyBboxPatch(
            (cx - w / 2, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.13",
            facecolor=fill, edgecolor=fill, linewidth=1.0, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(cx, y + h / 2, text, ha="center", va="center", color=INK,
                fontsize=fs, fontweight=weight or ("bold" if header else "normal"),
                linespacing=1.25, zorder=3)

    def arrow(start, end, *, color=WHITE, width=1.8, scale=14):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=scale,
            linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=1,
        ))

    center = 7.0
    box(center, 11.28, 4.7, 0.62, "입력 이미지  |  640 × 640", fs=10)
    box(center, 10.04, 8.9, 0.76,
        "공유 백본  |  MobileNetV3-Large\n" +
        ("C3·C4·C5 특징 추출" if not v3 else "C2·C3·C4·C5 특징 추출"), fs=10.2)
    arrow((center, 11.25), (center, 10.84))

    if not v3:
        neck_text = "BiFPN ×2  |  top-down + bottom-up 가중 특징 융합\nP3·P4·P5  |  각 96채널"
    else:
        neck_text = "Lite FPN  |  top-down 특징 융합 + depthwise separable conv\nP2·P3·P4·P5: 64채널  |  P6·P7: RPN용 max-pool"
    box(center, 8.72, 9.8, 0.85, neck_text, fs=9.4)
    arrow((center, 10.01), (center, 9.61))

    branch_y, branch_h = 7.48, 0.53
    left, right = 3.45, 10.55
    box(left, branch_y, 5.6, branch_h, "RPN  |  후보 상자 생성", header=True, fs=10)
    box(right, branch_y, 5.6, branch_h, "RoI 경로  |  특징 지도 전달", header=True, fs=10)
    arrow((center, 8.7), (left, branch_y + branch_h + 0.04))
    arrow((center, 8.7), (right, branch_y + branch_h + 0.04))

    if not v3:
        rpn_text = "앵커 기반 RPN\n3개 크기 × 3 종횡비 → 위치마다 9개 앵커\n추론 후보: 1,000개 → NMS 후 최대 300개"
        roi_text = "P3·P4·P5 특징 지도\n1×1 Conv로 96 → 64채널 압축"
        head_text = "경량 RoI Head\nRoI 7×7 → 평균 풀링 → FC 256"
        key_text = "핵심 특징  |  BiFPN 양방향 융합 · RoIAlign 전 채널 압축 · Faster R-CNN의 2단계 검출 유지"
    else:
        rpn_text = "앵커 기반 RPN\nP2–P7: 1개 크기 × 3 종횡비 → 위치마다 3개 앵커\n추론 후보: 1,000개 → NMS 후 최대 300개"
        roi_text = "P2·P3·P4·P5 특징 지도 (각 64채널)\nP6·P7은 RPN 전용"
        head_text = "경량 RoI Head\n1×1 Conv 64 → 32 → 평균 풀링 → FC 256"
        key_text = "핵심 특징  |  P2로 작은 객체 특징 추가 · separable FPN · 학습 중에만 Faster R-CNN 교사 증류"

    branch_box_y, branch_box_h = 6.05, 1.02
    box(left, branch_box_y, 5.9, branch_box_h, rpn_text, fs=8.9)
    box(right, branch_box_y, 5.9, branch_box_h, roi_text, fs=9.0)
    arrow((left, branch_y - 0.02), (left, branch_box_y + branch_box_h + 0.04))
    arrow((right, branch_y - 0.02), (right, branch_box_y + branch_box_h + 0.04))

    roi_y, roi_h = 4.70, 0.66
    roi_cx, roi_w = center, 7.2
    arrow((left, branch_box_y - 0.02), (roi_cx - 2.15, roi_y + roi_h + 0.04), color=WHITE)
    arrow((right, branch_box_y - 0.02), (roi_cx + 2.15, roi_y + roi_h + 0.04), color="#8CCB76")
    box(roi_cx, roi_y, roi_w, roi_h,
        "RoIAlign  |  후보 상자 + 대응 특징 지도 → 고정 크기 RoI 특징", fs=9.4, color=GREEN_HI)

    head_y, head_h = 3.35, 0.83
    arrow((center, roi_y - 0.02), (center, head_y + head_h + 0.04))
    box(center, head_y, 8.7, head_h, head_text, fs=9.7)

    output_y, output_h = 1.96, 0.78
    arrow((center, head_y - 0.02), (center, output_y + output_h + 0.04))
    box(center, output_y, 8.7, output_h,
        "Fast R-CNN 예측  |  COCO 91개 클래스 점수 + 박스 보정\nscore 기준 + 클래스별 NMS → 최종 객체 검출",
        fs=9.4, color=GREEN_HI)

    box(center, 0.45, 12.7, 0.88, key_text, fs=8.7, header=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = output_dir / f"lite_rcnn_v{version}_summary.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=BG)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/figures", help="Directory for summary PNG/SVG files")
    args = parser.parse_args()
    for version in (2, 3):
        for path in draw(version, Path(args.output_dir)):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
