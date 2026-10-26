"""Lightweight two-stage object detector."""

__all__ = ["LightRCNN", "build_detector", "build_vanilla_faster_rcnn"]


def __getattr__(name: str):
    # Import the detector lazily so torch-free helpers (reporting, viz) load without PyTorch.
    if name in __all__:
        from light_rcnn.model import detector

        return getattr(detector, name)
    raise AttributeError(f"module 'light_rcnn' has no attribute {name!r}")
