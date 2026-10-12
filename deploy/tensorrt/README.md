# TensorRT deployment

The ONNX export targets the fixed-shape Faster R-CNN student, including its RPN and RoI detection path. RoIAlign and NMS support must be checked against the installed TensorRT/CUDA version and may require plugins.

## Build outline

1. Export weights with `python scripts/export_onnx.py --weights outputs/coco/best.pth --output outputs/light_rcnn.onnx`.
2. Build an FP16 engine with `trtexec --onnx=outputs/light_rcnn.onnx --saveEngine=outputs/light_rcnn_fp16.engine --fp16 --shapes=images:1x3x640x640`.
3. Implement and register TensorRT plugins for bounded top-k proposal decode, RoIAlign and class-aware NMS, then benchmark with identical preprocessing and confidence settings.

`cpp/deterministic_nms.cpp` provides a standalone stable class-aware greedy NMS implementation and can be built with CMake. It is CPU postprocessing code; it is not yet a TensorRT plugin or complete inference runner. The TensorRT plugin and runner need to target the installed TensorRT/CUDA versions. Do not compare the raw ONNX backbone latency with end-to-end detector latency.
