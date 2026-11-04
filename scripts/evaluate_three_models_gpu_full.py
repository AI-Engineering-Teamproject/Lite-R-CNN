"""Evaluate Faster R-CNN, Lite R-CNN v2 and v3 on all COCO val2017 images using CUDA by default
(--with-mobilenet adds MobileNetV3 Faster R-CNN as a fourth model)."""

import importlib.util
from pathlib import Path

# Same evaluation as evaluate_three_models.py; only the defaults below differ. Loaded by file path
# so the script also runs from another folder or through runpy.
_spec = importlib.util.spec_from_file_location(
    "evaluate_three_models", Path(__file__).resolve().with_name("evaluate_three_models.py"))
evaluate_three_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_three_models)

DEFAULTS = {"device": "cuda", "images": 5000}


def build_parser():
    return evaluate_three_models.build_parser(__doc__, **DEFAULTS)


if __name__ == "__main__":
    evaluate_three_models.main(parser=build_parser())
