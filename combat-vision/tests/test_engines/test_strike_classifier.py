"""Strike classifier tests, driven by the jab fixture via the speed engine."""

from __future__ import annotations

from dataclasses import replace

from combat_vision.calibration import Calibration
from combat_vision.engines.speed import SpeedEngine
from combat_vision.engines.stance import StanceEngine
from combat_vision.engines.strike_classifier import StrikeClassifierEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    SpeedPeakEvent,
    SpeedUnit,
    Stance,
    StanceSample,
    StrikeEvent,
    StrikeType,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import (
    SpeedEngineConfig,
    StanceConfig,
    StrikeClassifierConfig,
)
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


def _straight_arm_pose(wrist_y: float) -> Pose:
    """Shoulder(0.35)/elbow(0.45)/wrist colinear on the y axis, elbow between
    shoulder and wrist -- a straight-arm punch reads as elbow angle 180 only
    when ``wrist_y`` is past the elbow (> 0.45); a retracted start frame
    should stay close to the shoulder (~0.36) so it is the *smaller*-reach
    frame and the extended one is correctly picked as the reach maximum.
    """
    return Pose(
        keypoints={
            KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35),
            KeypointName.LEFT_ELBOW: Keypoint(x=0.55, y=0.45),
            KeypointName.LEFT_WRIST: Keypoint(x=0.55, y=wrist_y),
        }
    )


def test_relabel_clears_buffer_and_stale_stance_so_lead_hand_reads_correctly() -> None:
    """A left-hand straight punch right after a relabel must classify off the
    new fighter's own (unknown -> orthodox default) stance, not the departed
    fighter's stale southpaw stance.

    A southpaw's lead hand is the right, so under the departed fighter's
    stance a left-hand punch reads as the rear hand (CROSS). Clearing the
    stale stance on relabel falls back to the orthodox default, correctly
    reading the new fighter's left hand as the lead (JAB) -- the exact
    strike type is the observable signal that the stance was actually
    cleared, not just the buffer.
    """
    calibration = Calibration(metres_per_pixel=0.002, frame_width_px=720, frame_height_px=720)
    bus = EventBus()
    strikes: list[StrikeEvent] = []
    bus.subscribe(StrikeEvent, strikes.append)
    engine = StrikeClassifierEngine(
        bus, get_profile("boxing"), calibration, StrikeClassifierConfig()
    )

    # The departed fighter "A" was recorded as southpaw.
    bus.publish(StanceSample(timestamp_s=0.0, fighter_id="A", stance=Stance.SOUTHPAW))

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=0.4, fighter_id="A"))

    # New fighter "A": a straight left-hand punch, buffered fresh after the relabel.
    engine.process(TrackedPose(fighter_id="A", pose=_straight_arm_pose(0.36), timestamp_s=0.5))
    engine.process(TrackedPose(fighter_id="A", pose=_straight_arm_pose(0.55), timestamp_s=0.6))

    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.6,
            fighter_id="A",
            limb=Limb.LEFT_HAND,
            peak_speed=6.0,
            unit=SpeedUnit.METERS_PER_SECOND,
            start_s=0.5,
            end_s=0.6,
        )
    )

    assert len(strikes) == 1
    assert strikes[0].strike_type == StrikeType.JAB
