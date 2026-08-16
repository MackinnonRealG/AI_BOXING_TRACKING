"""Rotation engine tests — synthetic pose sequences, no camera required.

SpeedPeakEvent candidates are published directly onto the bus (rather than
produced by the speed engine) so each test controls exactly the shoulder/hip
geometry under evaluation, per the "engines are bus-isolated, independently
testable" design rule.
"""

from __future__ import annotations

import math

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.rotation import RotationEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    CleanTechniqueEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    RotationFaultEvent,
    SpeedPeakEvent,
    SpeedUnit,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import RotationEngineConfig

# Square frame: to_pixels() scales x/y independently by frame width/height, so a
# non-square frame would distort the exact angles constructed below. Equal
# width/height keeps line_angle()'s output degrees matching the input degrees.
_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=720, frame_height_px=720)
_MPS = SpeedUnit.METERS_PER_SECOND


def _rotated_pair(
    center: tuple[float, float], angle_deg: float, half_len: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two points straddling ``center``, forming a line at ``angle_deg``."""
    theta = math.radians(angle_deg)
    dx, dy = half_len * math.cos(theta), half_len * math.sin(theta)
    return (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy)


def _pose(shoulder_deg: float, hip_deg: float) -> Pose:
    """A torso with the shoulder line and hip line each at a given angle."""
    l_sh, r_sh = _rotated_pair((0.5, 0.35), shoulder_deg)
    l_hip, r_hip = _rotated_pair((0.5, 0.60), hip_deg)
    return Pose(
        keypoints={
            KeypointName.LEFT_SHOULDER: Keypoint(x=l_sh[0], y=l_sh[1]),
            KeypointName.RIGHT_SHOULDER: Keypoint(x=r_sh[0], y=r_sh[1]),
            KeypointName.LEFT_HIP: Keypoint(x=l_hip[0], y=l_hip[1]),
            KeypointName.RIGHT_HIP: Keypoint(x=r_hip[0], y=r_hip[1]),
        }
    )


def _run(
    start: tuple[float, float], end: tuple[float, float], config: RotationEngineConfig
) -> tuple[list[RotationFaultEvent], list[CleanTechniqueEvent]]:
    """Feed a two-frame (shoulder, hip) stroke through a fresh engine."""
    bus = EventBus()
    faults: list[RotationFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(RotationFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = RotationEngine(bus, get_profile("boxing"), _CALIBRATION, config)

    engine.process(TrackedPose(fighter_id="A", pose=_pose(*start), timestamp_s=0.0))
    engine.process(TrackedPose(fighter_id="A", pose=_pose(*end), timestamp_s=0.3))
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.3,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=0.3,
        )
    )
    return faults, clean


def test_shoulders_turn_without_hips_is_flagged() -> None:
    """40° of shoulder turn with only 5° of hip turn (ratio well under 0.5) faults."""
    faults, clean = _run((0.0, 0.0), (40.0, 5.0), RotationEngineConfig())
    assert len(faults) == 1
    assert clean == []
    fault = faults[0]
    assert fault.limb == Limb.RIGHT_HAND
    assert fault.shoulder_rotation_deg == pytest.approx(40.0)
    assert fault.hip_rotation_deg == pytest.approx(5.0)


def test_hips_following_shoulders_publishes_a_clean_technique_event() -> None:
    """40° of shoulder turn with 30° of hip turn (ratio 0.75) is clean technique.

    The good rep is logged, not just silently skipped — the whole point of
    tracking both mistakes and clean technique, not mistakes alone.
    """
    faults, clean = _run((0.0, 0.0), (40.0, 30.0), RotationEngineConfig())
    assert faults == []
    assert len(clean) == 1
    assert clean[0].check == "hip_rotation"
    assert clean[0].limb == Limb.RIGHT_HAND


def test_small_shoulder_turn_is_not_judged_either_way() -> None:
    """A jab-sized 5° shoulder turn is below the threshold — neither faulted nor praised."""
    faults, clean = _run((0.0, 0.0), (5.0, 0.0), RotationEngineConfig())
    assert faults == []
    assert clean == []


def test_kick_candidates_are_ignored() -> None:
    """Rotation faults are a hand-strike concept; feet/knees are not evaluated."""
    bus = EventBus()
    faults: list[RotationFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(RotationFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = RotationEngine(bus, get_profile("kickboxing"), _CALIBRATION, RotationEngineConfig())

    engine.process(TrackedPose(fighter_id="A", pose=_pose(0.0, 0.0), timestamp_s=0.0))
    engine.process(TrackedPose(fighter_id="A", pose=_pose(40.0, 0.0), timestamp_s=0.3))
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.3,
            fighter_id="A",
            limb=Limb.RIGHT_FOOT,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=0.3,
        )
    )
    assert faults == []
    assert clean == []
