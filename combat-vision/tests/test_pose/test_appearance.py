"""Appearance descriptor tests: synthetic solid-color crops, no real images."""

from __future__ import annotations

import numpy as np
import pytest

from combat_vision.events.types import BBox
from combat_vision.pose import appearance


def _solid_image(color_bgr: tuple[int, int, int], size: int = 200) -> np.ndarray:
    """A uniform-color image, with a differently-colored border so the crop
    region is what actually matters, not the whole frame."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)  # background: white
    image[40:160, 40:160] = color_bgr  # the "person" region
    return image


_BBOX = BBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)  # matches the colored region above


def test_histogram_is_none_for_a_tiny_crop() -> None:
    image = _solid_image((0, 0, 255))
    tiny_bbox = BBox(x_min=0.5, y_min=0.5, x_max=0.501, y_max=0.501)
    assert appearance.histogram(image, tiny_bbox) is None


def test_histogram_is_normalized() -> None:
    image = _solid_image((0, 0, 255))  # red
    hist = appearance.histogram(image, _BBOX)
    assert hist is not None
    assert sum(hist) == pytest.approx(1.0)


def test_same_color_crops_have_zero_distance() -> None:
    red_a = appearance.histogram(_solid_image((0, 0, 255)), _BBOX)
    red_b = appearance.histogram(_solid_image((0, 0, 255)), _BBOX)
    assert appearance.distance(red_a, red_b) == pytest.approx(0.0, abs=1e-4)


def test_distinct_colors_have_nonzero_distance() -> None:
    red = appearance.histogram(_solid_image((0, 0, 255)), _BBOX)
    blue = appearance.histogram(_solid_image((255, 0, 0)), _BBOX)
    dist = appearance.distance(red, blue)
    assert dist is not None
    assert dist > 0.3


def test_distance_is_none_when_either_descriptor_is_missing() -> None:
    red = appearance.histogram(_solid_image((0, 0, 255)), _BBOX)
    assert appearance.distance(red, None) is None
    assert appearance.distance(None, red) is None
    assert appearance.distance(None, None) is None
