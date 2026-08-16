"""Knee-bend engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.knee_bend import KneeBendEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    CleanTechniqueEvent,
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    Limb,
    Pose,
    SpeedPeakEvent,
    SpeedUnit,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import KneeBendConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_MPS = SpeedUnit.METERS_PER_SECOND
_FPS = 60


def _pose(locked: bool, missing: KeypointName | None = None) -> Pose:
    """Both knees locked-straight (ankle directly below hip) or bent (ankle offset in)."""
    ankle_dx = 0.0 if locked else 0.05
    keypoints = {
        KeypointName.LEFT_HIP: Keypoint(x=0.55, y=0.55),
        KeypointName.RIGHT_HIP: Keypoint(x=0.45, y=0.55),
        KeypointName.LEFT_KNEE: Keypoint(x=0.55, y=0.75),
        KeypointName.RIGHT_KNEE: Keypoint(x=0.45, y=0.75),
        KeypointName.LEFT_ANKLE: Keypoint(x=0.55 - ankle_dx, y=0.95),
        KeypointName.RIGHT_ANKLE: Keypoint(x=0.45 + ankle_dx, y=0.95),
    }
    keypoints.pop(missing, None)
    return Pose(keypoints=keypoints)


def _run_posture(sequence: list[tuple[float, bool]]) -> list[KneeBendStateEvent]:
    bus = EventBus()
    events: list[KneeBendStateEvent] = []
    bus.subscribe(KneeBendStateEvent, events.append)
    engine = KneeBendEngine(bus, get_profile("boxing"), _CALIBRATION, KneeBendConfig())
    for t, locked in sequence:
        engine.process(TrackedPose(fighter_id="A", pose=_pose(locked), timestamp_s=t))
    return events


def test_sustained_locked_knees_are_flagged_after_debounce() -> None:
    """Both knees held straight past lock_debounce_s flips posture to locked."""
    frames = [(i / _FPS, False) for i in range(_FPS)]  # start bent
    frames += [(1.0 + i / _FPS, True) for i in range(2 * _FPS)]
    events = _run_posture(frames)

    locks = [e for e in events if e.locked]
    assert len(locks) == 1
    assert locks[0].timestamp_s >= 1.0 + KneeBendConfig().lock_debounce_s


def test_brief_lockout_is_debounced() -> None:
    """A 5-frame straightening between reps must not register as locked posture."""
    frames = [(i / _FPS, False) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, True) for i in range(5)]
    frames += [(1.0 + 5 / _FPS + i / _FPS, False) for i in range(_FPS)]
    events = _run_posture(frames)
    assert all(not e.locked for e in events)


def test_occlusion_does_not_let_a_stale_pending_transition_survive() -> None:
    """A pending "about to lock" transition must not survive an occlusion gap
    longer than the debounce interval — otherwise recovery falsely completes
    the debounce using time spent with no data at all.
    """
    bus = EventBus()
    events: list[KneeBendStateEvent] = []
    bus.subscribe(KneeBendStateEvent, events.append)
    engine = KneeBendEngine(bus, get_profile("boxing"), _CALIBRATION, KneeBendConfig())

    t = 0.0
    for _ in range(_FPS):  # 1s of bent -> current_locked settles to False
        engine.process(TrackedPose(fighter_id="A", pose=_pose(False), timestamp_s=t))
        t += 1 / _FPS
    events.clear()

    # One frame suggesting "about to lock" -- starts a pending candidacy.
    engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=t))
    t += 1 / _FPS

    # Occlusion lasting longer than lock_debounce_s (1.0s default).
    missing_pose = _pose(True, missing=KeypointName.RIGHT_KNEE)
    occlusion_end = t + KneeBendConfig().lock_debounce_s + 0.2
    while t < occlusion_end:
        engine.process(TrackedPose(fighter_id="A", pose=missing_pose, timestamp_s=t))
        t += 1 / _FPS

    # Keypoints return, still locked -- this must start a FRESH candidacy,
    # not instantly satisfy the debounce using time spent with no data.
    engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=t))
    assert events == []


def test_relabel_clears_posture_state_so_the_new_person_gets_an_initial_event() -> None:
    """Same reasoning as the guard engine's relabel test: a relabeled fighter
    starting in the same posture the departed one left behind must still get
    classified, not silently inherit stale state.
    """
    bus = EventBus()
    events: list[KneeBendStateEvent] = []
    bus.subscribe(KneeBendStateEvent, events.append)
    engine = KneeBendEngine(bus, get_profile("boxing"), _CALIBRATION, KneeBendConfig())

    t = 0.0
    for _ in range(2 * _FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=t))
        t += 1 / _FPS
    assert len(events) == 1  # the initial "locked" event
    events.clear()

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=t, fighter_id="A"))

    for _ in range(2 * _FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=t))
        t += 1 / _FPS
    assert len(events) == 1


def test_missing_leg_keypoints_produce_no_posture_events() -> None:
    """Without both full leg triples, there is nothing reliable to classify."""
    bus = EventBus()
    events: list[KneeBendStateEvent] = []
    bus.subscribe(KneeBendStateEvent, events.append)
    engine = KneeBendEngine(bus, get_profile("boxing"), _CALIBRATION, KneeBendConfig())
    for i in range(2 * _FPS):
        pose = _pose(True, missing=KeypointName.RIGHT_KNEE)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i / _FPS))
    assert events == []


def _run_strike(start_locked: bool) -> tuple[list[LegDriveFaultEvent], list[CleanTechniqueEvent]]:
    bus = EventBus()
    faults: list[LegDriveFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(LegDriveFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = KneeBendEngine(bus, get_profile("boxing"), _CALIBRATION, KneeBendConfig())

    engine.process(TrackedPose(fighter_id="A", pose=_pose(start_locked), timestamp_s=0.0))
    engine.process(TrackedPose(fighter_id="A", pose=_pose(start_locked), timestamp_s=0.2))
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.2,
            fighter_id="A",
            limb=Limb.LEFT_HAND,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=0.2,
        )
    )
    return faults, clean


def test_punch_thrown_with_locked_knees_is_flagged() -> None:
    faults, clean = _run_strike(start_locked=True)
    assert len(faults) == 1
    assert clean == []
    assert faults[0].limb == Limb.LEFT_HAND
    assert faults[0].knee_angle_deg == pytest.approx(180.0)


def test_punch_thrown_with_bent_knees_publishes_a_clean_technique_event() -> None:
    """The good rep is logged, not just silently skipped."""
    faults, clean = _run_strike(start_locked=False)
    assert faults == []
    assert len(clean) == 1
    assert clean[0].check == "leg_drive"
    assert clean[0].limb == Limb.LEFT_HAND


def test_kick_candidates_are_ignored_for_leg_drive() -> None:
    """Leg-drive faults are a hand-strike concept; a kicking leg is meant to extend."""
    bus = EventBus()
    faults: list[LegDriveFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(LegDriveFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = KneeBendEngine(bus, get_profile("kickboxing"), _CALIBRATION, KneeBendConfig())

    engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=0.0))
    engine.process(TrackedPose(fighter_id="A", pose=_pose(True), timestamp_s=0.2))
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.2,
            fighter_id="A",
            limb=Limb.RIGHT_FOOT,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=0.2,
        )
    )
    assert faults == []
    assert clean == []
