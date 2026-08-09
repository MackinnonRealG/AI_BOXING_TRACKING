"""Deterministic fixture generator for engine tests.

Run ``python tests/fixtures/generate_fixtures.py`` to (re)create the JSON
pose sequences. Fixtures are synthetic so tests have *known ground truth*:
the jab's wrist speed follows v(t) = V_PEAK * sin(pi * t / T), so the true
peak speed is exactly V_PEAK m/s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FPS = 60
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
METRES_PER_PIXEL = 0.002

V_PEAK_MPS = 6.0        # true peak hand speed of the synthetic jab
EXTENSION_S = 0.25      # duration of the extension stroke
HOLD_S = 0.10           # pause at full extension (separates out/back strokes)
IDLE_S = 0.50           # stillness before and after

# Orthodox fighter facing +x (nose ahead of the hip midline), left side lead.
_BASE = {
    "nose": (0.52, 0.20),
    "left_shoulder": (0.55, 0.35),
    "right_shoulder": (0.45, 0.35),
    "left_elbow": (0.575, 0.40),   # midpoint shoulder->wrist: straight arm
    "right_elbow": (0.425, 0.40),
    "left_wrist": (0.60, 0.45),
    "right_wrist": (0.40, 0.45),
    "left_hip": (0.53, 0.60),
    "right_hip": (0.47, 0.60),
    "left_knee": (0.545, 0.75),
    "right_knee": (0.455, 0.75),
    "left_ankle": (0.55, 0.90),
    "right_ankle": (0.44, 0.90),
}


def _displacement_m(t: float) -> float:
    """Integral of the sinusoidal speed profile up to time t (extension)."""
    # ∫ V sin(pi t / T) dt = V T / pi * (1 - cos(pi t / T))
    return V_PEAK_MPS * EXTENSION_S / math.pi * (1 - math.cos(math.pi * t / EXTENSION_S))


def _jab_frames() -> list[dict]:
    """Idle → extend (sin profile) → hold → retract → idle, left wrist only."""
    frames = []
    total = IDLE_S + EXTENSION_S + HOLD_S + EXTENSION_S + IDLE_S
    n = int(total * FPS) + 1
    full_extension_m = _displacement_m(EXTENSION_S)
    for i in range(n):
        t = i / FPS
        if t < IDLE_S:
            offset_m = 0.0
        elif t < IDLE_S + EXTENSION_S:
            offset_m = _displacement_m(t - IDLE_S)
        elif t < IDLE_S + EXTENSION_S + HOLD_S:
            offset_m = full_extension_m
        elif t < IDLE_S + 2 * EXTENSION_S + HOLD_S:
            back_t = t - (IDLE_S + EXTENSION_S + HOLD_S)
            offset_m = full_extension_m - _displacement_m(back_t)
        else:
            offset_m = 0.0
        offset_norm = (offset_m / METRES_PER_PIXEL) / FRAME_WIDTH

        keypoints = {name: list(xy) for name, xy in _BASE.items()}
        keypoints["left_wrist"] = [_BASE["left_wrist"][0] + offset_norm, _BASE["left_wrist"][1]]
        # The elbow tracks the shoulder->wrist midpoint: arm stays straight,
        # so the classifier reads this stroke as a lead-hand straight (a jab).
        keypoints["left_elbow"] = [_BASE["left_elbow"][0] + offset_norm / 2, _BASE["left_elbow"][1]]
        frames.append({"t": round(t, 6), "keypoints": keypoints})
    return frames


def _idle_frames() -> list[dict]:
    """Two seconds of stillness with sub-centimetre deterministic jitter."""
    frames = []
    jitter_norm = 0.0005  # ~0.6 px — realistic estimator noise
    for i in range(2 * FPS):
        t = i / FPS
        keypoints = {}
        for j, (name, (x, y)) in enumerate(_BASE.items()):
            keypoints[name] = [
                x + jitter_norm * math.sin(7.0 * t + j),
                y + jitter_norm * math.cos(5.0 * t + j),
            ]
        frames.append({"t": round(t, 6), "keypoints": keypoints})
    return frames


def main() -> None:
    """Write both fixture files next to this script."""
    here = Path(__file__).parent
    meta = {
        "fps": FPS,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "metres_per_pixel": METRES_PER_PIXEL,
        "true_peak_speed_mps": V_PEAK_MPS,
    }
    (here / "jab_sequence.json").write_text(
        json.dumps({**meta, "frames": _jab_frames()}, indent=1)
    )
    (here / "idle_sequence.json").write_text(
        json.dumps({**meta, "frames": _idle_frames()}, indent=1)
    )
    print("fixtures written")


if __name__ == "__main__":
    main()
