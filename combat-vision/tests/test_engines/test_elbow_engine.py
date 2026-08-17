"""Elbow engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.elbow import ElbowEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    ElbowStateEvent,
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import ElbowConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_FPS = 60
# Shoulders at x=0.45/0.55 -> centerline=0.5, half_width=0.05.
# flare_ratio default 1.7 -> flare line at offset 0.085 from centerline.
_TUCKED_DX = 0.05  # offset ratio 1.0 -- clearly tucked
_FLARED_DX = 0.12  # offset ratio 2.4 -- clearly flared


def _pose(left_tucked: bool, right_tucked: bool, missing: KeypointName | None = None) -> Pose:
    """A fighter with each elbow tucked near the torso or flared wide."""
    left_dx = _TUCKED_DX if left_tucked else _FLARED_DX
    right_dx = _TUCKED_DX if right_tucked else _FLARED_DX
    keypoints = {
        KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35),
        KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35),
        KeypointName.LEFT_ELBOW: Keypoint(x=0.5 + left_dx, y=0.45),
        KeypointName.RIGHT_ELBOW: Keypoint(x=0.5 - right_dx, y=0.45),
    }
    keypoints.pop(missing, None)
    return Pose(keypoints=keypoints)


def _run(sequence: list[tuple[float, bool, bool]]) -> list[ElbowStateEvent]:
    """Feed (timestamp, left_tucked?, right_tucked?) frames through a fresh engine."""
    bus = EventBus()
    events: list[ElbowStateEvent] = []
    bus.subscribe(ElbowStateEvent, events.append)
    engine = ElbowEngine(bus, get_profile("boxing"), _CALIBRATION, ElbowConfig())
    for t, left_tucked, right_tucked in sequence:
        pose = _pose(left_tucked, right_tucked)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=t))
    return events


def test_elbows_held_tucked_are_sampled_once_per_arm() -> None:
    frames = [(i / _FPS, True, True) for i in range(2 * _FPS)]
    events = _run(frames)
    assert {(e.elbow, e.tucked) for e in events} == {
        (Limb.LEFT_HAND, True),
        (Limb.RIGHT_HAND, True),
    }


def test_sustained_flare_is_flagged_after_debounce() -> None:
    frames = [(i / _FPS, True, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False, True) for i in range(_FPS)]
    events = _run(frames)

    flares = [e for e in events if e.elbow == Limb.LEFT_HAND and not e.tucked]
    assert len(flares) == 1
    assert flares[0].timestamp_s >= 1.0 + ElbowConfig().flare_debounce_s
    assert all(e.tucked for e in events if e.elbow == Limb.RIGHT_HAND)


def test_brief_flare_is_debounced() -> None:
    """A 5-frame flare (e.g. mid-punch extension) must not register as a fault."""
    frames = [(i / _FPS, True, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False, True) for i in range(5)]
    frames += [(1.0 + 5 / _FPS + i / _FPS, True, True) for i in range(_FPS)]
    events = _run(frames)
    assert all(e.tucked for e in events)


def test_missing_elbow_keypoint_produces_no_event_for_that_arm() -> None:
    bus = EventBus()
    events: list[ElbowStateEvent] = []
    bus.subscribe(ElbowStateEvent, events.append)
    engine = ElbowEngine(bus, get_profile("boxing"), _CALIBRATION, ElbowConfig())
    for i in range(2 * _FPS):
        pose = _pose(True, True, missing=KeypointName.LEFT_ELBOW)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i / _FPS))
    assert all(e.elbow != Limb.LEFT_HAND for e in events)
    assert any(e.elbow == Limb.RIGHT_HAND for e in events)


def test_missing_shoulder_keypoints_produce_no_events() -> None:
    """Without both shoulders there is no torso centerline to measure from."""
    bus = EventBus()
    events: list[ElbowStateEvent] = []
    bus.subscribe(ElbowStateEvent, events.append)
    engine = ElbowEngine(bus, get_profile("boxing"), _CALIBRATION, ElbowConfig())
    for i in range(_FPS):
        pose = _pose(True, True, missing=KeypointName.LEFT_SHOULDER)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i / _FPS))
    assert events == []


def test_relabel_clears_debounce_state_so_the_new_person_gets_an_initial_event() -> None:
    bus = EventBus()
    events: list[ElbowStateEvent] = []
    bus.subscribe(ElbowStateEvent, events.append)
    engine = ElbowEngine(bus, get_profile("boxing"), _CALIBRATION, ElbowConfig())

    t = 0.0
    for _ in range(2 * _FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True, True), timestamp_s=t))
        t += 1 / _FPS
    assert len(events) == 2  # one initial "tucked" event per arm
    events.clear()

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=t, fighter_id="A"))

    for _ in range(_FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True, True), timestamp_s=t))
        t += 1 / _FPS
    assert len(events) == 2
