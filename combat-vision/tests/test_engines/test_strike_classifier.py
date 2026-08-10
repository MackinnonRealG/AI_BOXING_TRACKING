"""Strike classifier tests, driven by the jab fixture via the speed engine."""

from __future__ import annotations

from dataclasses import replace

from combat_vision.engines.speed import SpeedEngine
from combat_vision.engines.stance import StanceEngine
from combat_vision.engines.strike_classifier import StrikeClassifierEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    KeypointName,
    Limb,
    Pose,
    StrikeEvent,
    StrikeType,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import SpeedEngineConfig, StanceConfig, StrikeClassifierConfig
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


def _run(poses: list[TrackedPose], meta: dict) -> list[StrikeEvent]:
    """Replay poses through stance -> classifier -> speed, as the pipeline does."""
    bus = EventBus()
    strikes: list[StrikeEvent] = []
    bus.subscribe(StrikeEvent, strikes.append)
    calibration = calibration_from_meta(meta)
    profile = get_profile("boxing")
    engines = [
        StanceEngine(bus, profile, calibration, StanceConfig()),
        StrikeClassifierEngine(bus, profile, calibration, StrikeClassifierConfig()),
        SpeedEngine(bus, profile, calibration, SpeedEngineConfig()),
    ]
    for pose in poses:
        for engine in engines:
            engine.process(pose)
    for engine in engines:
        engine.finish()
    return strikes


def test_lead_hand_straight_is_a_jab(jab_sequence: tuple[list[TrackedPose], dict]) -> None:
    """The fixture's straight lead-left punch must classify as a jab."""
    poses, meta = jab_sequence
    strikes = _run(poses, meta)

    assert len(strikes) == 1
    strike = strikes[0]
    assert strike.strike_type == StrikeType.JAB
    assert strike.limb == Limb.LEFT_HAND
    assert strike.speed > 0


def test_landed_is_unknown_without_opponent(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """With a single fighter in frame, landed must be None, never False."""
    poses, meta = jab_sequence
    strikes = _run(poses, meta)
    assert strikes[0].landed is None


def test_idle_produces_no_strikes(idle_sequence: tuple[list[TrackedPose], dict]) -> None:
    """No candidates -> no classified strikes."""
    poses, meta = idle_sequence
    assert _run(poses, meta) == []


def test_missing_elbow_is_unknown_not_a_confident_jab(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """An occluded elbow must not be read as a straight (extended) arm.

    Missing geometry should demote the strike to UNKNOWN, not manufacture a
    confident jab/cross classification from a fabricated 180° elbow angle.
    """
    poses, meta = jab_sequence
    poses = _without_keypoint(poses, KeypointName.LEFT_ELBOW)
    strikes = _run(poses, meta)

    assert len(strikes) == 1
    assert strikes[0].strike_type == StrikeType.UNKNOWN
