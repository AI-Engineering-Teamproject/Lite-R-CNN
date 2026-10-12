#include "deterministic_nms.hpp"

#include <algorithm>
#include <cmath>

namespace light_rcnn {
namespace {

float intersection_over_union(const Detection& a, const Detection& b) {
  const float left = std::max(a.x1, b.x1);
  const float top = std::max(a.y1, b.y1);
  const float right = std::min(a.x2, b.x2);
  const float bottom = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  const float denominator = area_a + area_b - intersection;
  return denominator > 0.0F ? intersection / denominator : 0.0F;
}

}  // namespace

std::vector<Detection> deterministic_nms(const std::vector<Detection>& detections,
                                         float iou_threshold,
                                         std::size_t max_detections) {
  if (max_detections == 0) return {};
  std::vector<Detection> ordered;
  ordered.reserve(detections.size());
  for (const auto& detection : detections) {
    if (std::isfinite(detection.score) && detection.x2 > detection.x1 && detection.y2 > detection.y1) {
      ordered.push_back(detection);
    }
  }
  std::stable_sort(ordered.begin(), ordered.end(), [](const Detection& a, const Detection& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.source_index < b.source_index;
  });

  std::vector<Detection> kept;
  kept.reserve(std::min(max_detections, ordered.size()));
  for (const auto& candidate : ordered) {
    const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const Detection& prior) {
      return prior.label == candidate.label && intersection_over_union(prior, candidate) > iou_threshold;
    });
    if (!suppressed) {
      kept.push_back(candidate);
      if (kept.size() >= max_detections) break;
    }
  }
  return kept;
}

}  // namespace light_rcnn
