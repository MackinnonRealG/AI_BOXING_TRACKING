"""Head-posture engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

import math

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.head_posture import HeadPostureEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    HeadPostureSample,
    Keypoint,
    KeypointName,
    Pose,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import HeadPostureConfig

# Square frame so line_angle() degrees aren't distorted by non-uniform x/y scaling.
_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=720, frame_height_px=720)


def _pose(head_tilt_deg: float, missing: KeypointName | None = None) -> Pose:
    """Level shoulders; eyes tilted by ``head_tilt_deg`` off the shoulder line.

    Matches this codebase's mirrored-camera convention (see stance engine
    fixtures): the "left" keypoint of a pair sits at the larger x, "right"
    at the smaller x, for both the eye line and the shoulder line — keeping
    them consistent is what makes a 0-degree tilt actually measure as 0.
    """
    theta = math.radians(head_tilt_deg)
    dx, dy = 0.03 * math.cos(theta), 0.03 * math.sin(theta)
    keypoints = {
        KeypointName.LEFT_EYE: Keypoint(x=0.5 + dx, y=0.20 + dy),
        KeypointName.RIGHT_EYE: Keypoint(x=0.5 - dx, y=0.20 - dy),
        KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35),
        KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35),
    }
    keypoints.pop(missing, None)
    return Pose(keypoints=keypoints)


def _run(sequence: list[tuple[float, float]]) -> list[HeadPostureSample]:
    """Feed (timestamp, head_tilt_deg) frames through a fresh engine."""
    bus = EventBus()
    samples: list[HeadPostureSample] = []
    bus.subscribe(HeadPostureSample, samples.append)
    engine = HeadPostureEngine(bus, get_profile("boxing"), _CALIBRATION, HeadPostureConfig())
    for t, tilt in sequence:
        engine.process(TrackedPose(fighter_id="A", pose=_pose(tilt), timestamp_s=t))
    return samples


def test_level_head_samples_near_zero_tilt() -> None:
    """Shoulders level, eyes level with them -> tilt close to 0 degrees."""
    samples = _run([(0.0, 0.0), (0.3, 0.0)])
    assert len(samples) == 2
    assert samples[-1].tilt_deg == pytest.approx(0.0, abs=1e-6)


def test_tilted_head_samples_the_gap_from_shoulders() -> None:
    """A 20-degree eye-line tilt against level shoulders reports ~20 degrees."""
    samples = _run([(0.0, 20.0), (0.3, 20.0)])
    assert samples[-1].tilt_deg == pytest.approx(20.0)


def test_samples_are_decimated_by_interval() -> None:
    """Frames closer together than sample_interval_s collapse to one sample."""
    frames = [(i * 0.01, 10.0) for i in range(30)]  # 0.3s span, 10ms steps
    samples = _run(frames)
    assert len(samples) == 2  # t=0.0 and the first frame >= 0.2s later


def test_missing_eye_keypoint_produces_no_sample() -> None:
    """Without both eyes, there is no eye line to compare against the shoulders."""
    bus = EventBus()
    samples: list[HeadPostureSample] = []
    bus.subscribe(HeadPostureSample, samples.append)
    engine = HeadPostureEngine(bus, get_profile("boxing"), _CALIBRATION, HeadPostureConfig())
    pose = _pose(10.0, missing=KeypointName.LEFT_EYE)
    engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=0.0))
    assert samples == []
