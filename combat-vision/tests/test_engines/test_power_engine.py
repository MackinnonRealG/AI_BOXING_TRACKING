"""Power engine tests, driven by the jab fixture via the speed engine."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.power import PowerEngine
from combat_vision.engines.speed import SpeedEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    PowerEstimateEvent,
    SpeedPeakEvent,
    SpeedUnit,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import PowerEngineConfig, SpeedEngineConfig
from tests.conftest import calibration_from_meta


def _without_keypoint(
    poses: list[TrackedPose], name: KeypointName
) -> list[TrackedPose]:
    """A copy of ``poses`` with one keypoint dropped from every frame."""
    result = []
    for tracked in poses:
        keypoints = dict(tracked.pose.keypoints)
        keypoints.pop(name, None)
        result.append(replace(tracked, pose=Pose(keypoints=keypoints)))
    return result


def _run(poses: list[TrackedPose], meta: dict) -> list[PowerEstimateEvent]:
    """Replay poses through power (buffering) then speed (event source)."""
    bus = EventBus()
    estimates: list[PowerEstimateEvent] = []
    bus.subscribe(PowerEstimateEvent, estimates.append)
    calibration = calibration_from_meta(meta)
    profile = get_profile("boxing")
    engines = [
        PowerEngine(bus, profile, calibration, PowerEngineConfig()),
        SpeedEngine(bus, profile, calibration, SpeedEngineConfig()),
    ]
    for pose in poses:
        for engine in engines:
            engine.process(pose)
    for engine in engines:
        engine.finish()
    return estimates


def test_jab_gets_a_power_estimate(jab_sequence: tuple[list[TrackedPose], dict]) -> None:
    """One candidate -> one power estimate, keyed to the same stroke."""
    poses, meta = jab_sequence
    estimates = _run(poses, meta)

    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.limb == Limb.LEFT_HAND
    assert 0.0 < estimate.score <= 100.0


def test_straight_arm_jab_score_is_plausible(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """Fixture jab: ~5.7 of 12 m/s speed, full extension, no rotation.

    Expected ≈ 100 * (0.5*0.48 + 0.25*1.0 + 0.25*0) ≈ 49 — assert a band
    wide enough to tolerate smoothing attenuation.
    """
    poses, meta = jab_sequence
    estimates = _run(poses, meta)
    assert 35.0 <= estimates[0].score <= 60.0


def test_idle_produces_no_estimates(idle_sequence: tuple[list[TrackedPose], dict]) -> None:
    """No candidates -> no power estimates."""
    poses, meta = idle_sequence
    assert _run(poses, meta) == []


def test_missing_elbow_scores_zero_extension_not_full_extension(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """An occluded elbow must not score as if the limb were fully extended.

    With the extension component forced to 0 (instead of the old 180°-default
    inflating it to 1.0), the score can only come from speed + rotation, so
    it must land well below the full-keypoints baseline (~35-60, see
    test_straight_arm_jab_score_is_plausible).
    """
    poses, meta = jab_sequence
    poses = _without_keypoint(poses, KeypointName.LEFT_ELBOW)
    estimates = _run(poses, meta)

    assert len(estimates) == 1
    assert 0.0 < estimates[0].score <= 30.0


def _shoulder_only_pose(shoulder_deg: float) -> Pose:
    """No elbow (extension forced to 0) and zero speed isolates the score to
    the rotation component alone, so a relabel bug in the buffered window is
    directly visible in the final score."""
    theta = math.radians(shoulder_deg)
    dx, dy = 0.05 * math.cos(theta), 0.05 * math.sin(theta)
    return Pose(
        keypoints={
            KeypointName.LEFT_SHOULDER: Keypoint(x=0.5 - dx, y=0.35 - dy),
            KeypointName.RIGHT_SHOULDER: Keypoint(x=0.5 + dx, y=0.35 + dy),
            KeypointName.LEFT_WRIST: Keypoint(x=0.6, y=0.45),
        }
    )


def test_relabel_clears_the_pose_buffer_so_strikes_do_not_mix_fighters() -> None:
    """A strike thrown right after a relabel must not window over the
    departed fighter's poses when computing the rotation component.

    Speed and extension are both zeroed out (peak_speed=0, no elbow
    keypoint) so the final score is determined by rotation alone, making a
    buffer-contamination bug directly visible as a wrong score instead of
    needing to inspect internals.
    """
    calibration = Calibration(metres_per_pixel=0.002, frame_width_px=720, frame_height_px=720)
    bus = EventBus()
    estimates: list[PowerEstimateEvent] = []
    bus.subscribe(PowerEstimateEvent, estimates.append)
    engine = PowerEngine(bus, get_profile("boxing"), calibration, PowerEngineConfig())

    # Departed fighter "A": shoulders square, buffered at t=0.0.
    engine.process(TrackedPose(fighter_id="A", pose=_shoulder_only_pose(0.0), timestamp_s=0.0))

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=0.0, fighter_id="A"))

    # New fighter "A": shoulders turn 40 -> 80 over their own real window.
    engine.process(TrackedPose(fighter_id="A", pose=_shoulder_only_pose(40.0), timestamp_s=0.5))
    engine.process(TrackedPose(fighter_id="A", pose=_shoulder_only_pose(80.0), timestamp_s=0.6))

    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.6,
            fighter_id="A",
            limb=Limb.LEFT_HAND,
            peak_speed=0.0,
            unit=SpeedUnit.METERS_PER_SECOND,
            start_s=0.0,  # spans the relabel boundary if the old pose were still buffered
            end_s=0.6,
        )
    )

    assert len(estimates) == 1
    # Correct: only the new fighter's 40-degree turn over their 0.1s of real
    # motion counts, at duration_s = event duration (0.6s).
    # rotation = min((40/0.6)/600, 1.0) = 0.1111; score = 100*0.25*rotation.
    expected = 100.0 * PowerEngineConfig().rotation_weight * min((40.0 / 0.6) / 600.0, 1.0)
    assert estimates[0].score == pytest.approx(expected, abs=0.05)
