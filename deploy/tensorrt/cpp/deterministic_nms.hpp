#pragma once

#include <cstddef>
#include <vector>

namespace light_rcnn {

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
  std::size_t source_index;
};

// Stable class-aware greedy NMS. Equal scores keep the lower source_index first.
std::vector<Detection> deterministic_nms(const std::vector<Detection>& detections,
                                         float iou_threshold,
                                         std::size_t max_detections);

}  // namespace light_rcnn
