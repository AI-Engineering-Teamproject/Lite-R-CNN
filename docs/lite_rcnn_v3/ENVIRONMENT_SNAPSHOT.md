# Local Python Environment Snapshot

Captured from this Windows workspace on 2026-11-08. This describes the local CPU development environment only; it is not evidence of the RTX 4090 training environment.

## `.venv312`

- Executable: `.venv312/Scripts/python.exe`
- Python: `3.12.14` (64-bit)
- PyTorch: `2.14.1+cpu`
- TorchVision: `0.29.1+cpu`
- CUDA available: `False`
- NumPy: `2.3.5`
- PyYAML: `6.0.3`
- Pillow: `12.3.0`
- pycocotools: `2.0.11`
- Matplotlib: `3.11.2`
- Ultralytics: `8.4.177` (installed for the separate YOLO comparison; Lite R-CNN v3 does not use it)
- Streamlit: `1.65.0`
- ONNX: `1.23.2`

## `.venv`

- Python: `3.6.8`
- PyTorch import: unavailable (`ModuleNotFoundError`)
- This interpreter is below the project declaration `requires-python >=3.10`.

## RTX 4090 training environment

The saved `outputs/coco_4090_v3/RESULTS.md` records PyTorch `2.7.1+cu128` and CUDA runtime `12.8`. The Python and TorchVision versions were not recorded in the saved run report/config/history; do not infer them from `.venv312`.

## Probe method

Package versions were read from the active interpreter with Python package metadata; PyTorch/TorchVision build suffixes and CUDA availability were read from `torch.__version__`, `torchvision.__version__`, and `torch.cuda.is_available()`. The old `.venv` was checked with `python --version` and an import attempt for `torch`.
