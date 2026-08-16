"""Depth-posture engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.depth_posture import DepthPostureEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    DepthPostureSample,
    Keypoint,
    KeypointName,
    Pose,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import DepthPostureConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)


def _pose(
    left_elbow_z: float | None = None,
    right_elbow_z: float | None = None,
    shoulder_z: float | None = 0.0,
    hip_z: float | None = 0.0,
    include_elbows: bool = True,
) -> Pose:
    keypoints = {
        KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35, z=shoulder_z),
        KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35, z=shoulder_z),
        KeypointName.LEFT_HIP: Keypoint(x=0.53, y=0.60, z=hip_z),
        KeypointName.RIGHT_HIP: Keypoint(x=0.47, y=0.60, z=hip_z),
    }
    if include_elbows:
        keypoints[KeypointName.LEFT_ELBOW] = Keypoint(x=0.60, y=0.45, z=left_elbow_z)
        keypoints[KeypointName.RIGHT_ELBOW] = Keypoint(x=0.40, y=0.45, z=right_elbow_z)
    return Pose(keypoints=keypoints)


def _sample(pose: Pose) -> DepthPostureSample | None:
    bus = EventBus()
    samples: list[DepthPostureSample] = []
    bus.subscribe(DepthPostureSample, samples.append)
    engine = DepthPostureEngine(bus, get_profile("boxing"), _CALIBRATION, DepthPostureConfig())
    engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=0.0))
    return samples[0] if samples else None


def test_elbow_closer_than_torso_reads_as_flared() -> None:
    """An elbow at z=-0.1 (closer to camera) against a torso at z=0.0 flares positive."""
    sample = _sample(_pose(left_elbow_z=-0.1, right_elbow_z=0.0))
    assert sample is not None
    assert sample.left_elbow_flare == pytest.approx(0.1)
    assert sample.right_elbow_flare == pytest.approx(0.0)


def test_elbow_tucked_behind_torso_reads_as_not_flared() -> None:
    """An elbow farther from the camera than the torso is negative, not flared."""
    sample = _sample(_pose(left_elbow_z=0.1, right_elbow_z=0.0))
    assert sample is not None
    assert sample.left_elbow_flare == pytest.approx(-0.1)


def test_shoulders_closer_than_hips_reads_as_leaning_forward() -> None:
    sample = _sample(_pose(shoulder_z=-0.1, hip_z=0.0))
    assert sample is not None
    assert sample.torso_lean == pytest.approx(0.1)


def test_shoulders_farther_than_hips_reads_as_leaning_back() -> None:
    sample = _sample(_pose(shoulder_z=0.1, hip_z=0.0))
    assert sample is not None
    assert sample.torso_lean == pytest.approx(-0.1)


def test_missing_elbow_z_leaves_that_side_none_but_still_samples_lean() -> None:
    """Partial data still produces a sample — only the unmeasurable field is None."""
    pose = _pose(left_elbow_z=None, right_elbow_z=0.0, shoulder_z=0.0, hip_z=0.0)
    sample = _sample(pose)
    assert sample is not None
    assert sample.left_elbow_flare is None
    assert sample.right_elbow_flare == pytest.approx(0.0)
    assert sample.torso_lean == pytest.approx(0.0)


def test_no_z_data_at_all_produces_no_sample() -> None:
    """A backend that never reports z (or a frame with nothing measurable) publishes nothing."""
    pose = _pose(include_elbows=False, shoulder_z=None, hip_z=None)
    assert _sample(pose) is None
