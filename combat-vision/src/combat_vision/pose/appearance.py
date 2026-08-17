"""Lightweight visual appearance descriptor for cross-occlusion re-identification.

A full re-ID embedding network is out of scope for this v1 — no labelled
training data exists to build or validate one, and it would pull in a new
heavy dependency. Instead: a normalized HSV hue histogram of the detection's
bounding-box crop, cheap enough to compute every frame.

This is good enough to fix the actual failure mode it targets — after a long
occlusion, ByteTrack hands a reappearing fighter a fresh internal track id,
and the label pool previously just recycled the first available label onto
it regardless of who it actually was (see
:class:`~combat_vision.tracking.supervision_tracker.SupervisionTracker`).
Two people of visibly different skin tone/clothing color are now
disambiguated correctly. It will **not** reliably distinguish two fighters
in near-identical kit (e.g. both in the same-color rash guard) — that's a
real, known limitation of a color-histogram descriptor, not a bug.
"""

from __future__ import annotations

import cv2
import numpy as np

from combat_vision.events.types import BBox

_HIST_BINS = 32
_MIN_CROP_PX = 4


def histogram(image: np.ndarray, bbox: BBox) -> tuple[float, ...] | None:
    """A normalized hue histogram of the bbox region, or None if too small to sample."""
    height, width = image.shape[:2]
    x1, y1 = max(int(bbox.x_min * width), 0), max(int(bbox.y_min * height), 0)
    x2, y2 = min(int(bbox.x_max * width), width), min(int(bbox.y_max * height), height)
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None

    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [_HIST_BINS], [0, 180])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return tuple(float(v) for v in hist.flatten())


def distance(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    """Bhattacharyya distance between two histograms (0 = identical), or None
    if either descriptor is missing."""
    if a is None or b is None:
        return None
    hist_a = np.asarray(a, dtype=np.float32)
    hist_b = np.asarray(b, dtype=np.float32)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))
