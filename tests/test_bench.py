import numpy as np
import pytest
import torch
from PIL import Image

from light_rcnn.bench import load_images, resolve_device


def test_load_images_returns_sorted_uint8_chw_rgb(tmp_path):
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[..., 0] = 200  # red channel
    Image.fromarray(pixels).save(tmp_path / "b.png")
    Image.fromarray(pixels[..., 0]).save(tmp_path / "a.png")  # grayscale input converts to RGB
    (tmp_path / "notes.txt").write_text("skip me")
    paths, images = load_images(tmp_path, count=5)
    assert [path.name for path in paths] == ["a.png", "b.png"]
    assert all(image.dtype == torch.uint8 and image.shape == (3, 4, 6) for image in images)
    assert images[1][0].eq(200).all() and images[1][1:].eq(0).all()


def test_load_images_respects_count_and_rejects_empty(tmp_path):
    for name in ("1.png", "2.png", "3.png"):
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    assert len(load_images(tmp_path, count=2)[1]) == 2
    with pytest.raises(SystemExit):
        load_images(tmp_path / "missing", count=1)


def test_resolve_device_accepts_index_and_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("0").type == "cpu"
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unknown_names():
    with pytest.raises(ValueError):
        resolve_device("gpu-please")
