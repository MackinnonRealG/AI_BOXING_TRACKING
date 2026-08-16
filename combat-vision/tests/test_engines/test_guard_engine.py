"""Guard engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.guard import GuardEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterRelabeledEvent,
    GuardStateEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import GuardConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_FPS = 60
# nose.y=0.20, shoulder.y=0.35 -> span=0.15; default config -> guard line y=0.29.
_UP_Y = 0.25
_DOWN_Y = 0.45


def _pose(left_up: bool, right_up: bool, missing: KeypointName | None = None) -> Pose:
    """A fighter with each wrist above (guard up) or below (guard down) the chin line."""
    keypoints = {
        KeypointName.NOSE: Keypoint(x=0.50, y=0.20),
        KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35),
        KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35),
        KeypointName.LEFT_WRIST: Keypoint(x=0.55, y=_UP_Y if left_up else _DOWN_Y),
        KeypointName.RIGHT_WRIST: Keypoint(x=0.45, y=_UP_Y if right_up else _DOWN_Y),
    }
    keypoints.pop(missing, None)
    return Pose(keypoints=keypoints)


def _run(sequence: list[tuple[float, bool, bool]]) -> list[GuardStateEvent]:
    """Feed (timestamp, left_up?, right_up?) frames through a fresh engine."""
    bus = EventBus()
    events: list[GuardStateEvent] = []
    bus.subscribe(GuardStateEvent, events.append)
    engine = GuardEngine(bus, get_profile("boxing"), _CALIBRATION, GuardConfig())
    for t, left_up, right_up in sequence:
        engine.process(TrackedPose(fighter_id="A", pose=_pose(left_up, right_up), timestamp_s=t))
    return events


def test_guard_held_up_is_sampled_once_per_hand() -> None:
    """Both hands held up from the start yield exactly one 'up' event each."""
    frames = [(i / _FPS, True, True) for i in range(2 * _FPS)]
    events = _run(frames)
    assert {(e.hand, e.guard_up) for e in events} == {
        (Limb.LEFT_HAND, True),
        (Limb.RIGHT_HAND, True),
    }


def test_sustained_drop_is_flagged_after_debounce() -> None:
    """A hand held down for longer than drop_debounce_s flips to guard_up=False."""
    frames = [(i / _FPS, True, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False, True) for i in range(_FPS)]
    events = _run(frames)

    drops = [e for e in events if e.hand == Limb.LEFT_HAND and not e.guard_up]
    assert len(drops) == 1
    assert drops[0].timestamp_s >= 1.0 + GuardConfig().drop_debounce_s
    # The other hand never left guard, so it should only ever report "up".
    assert all(e.guard_up for e in events if e.hand == Limb.RIGHT_HAND)


def test_brief_dip_is_debounced() -> None:
    """A 5-frame dip below the guard line must not register as a drop."""
    frames = [(i / _FPS, True, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False, True) for i in range(5)]
    frames += [(1.0 + 5 / _FPS + i / _FPS, True, True) for i in range(_FPS)]
    events = _run(frames)
    assert all(e.guard_up for e in events)


def test_relabel_clears_debounce_state_so_the_new_person_gets_an_initial_event() -> None:
    """A relabeled fighter must get their own initial event, not go silent.

    Without clearing debounce state on relabel, a new person starting in the
    same guard state the departed one left behind hits the
    ``up_now == current_up`` short-circuit and never gets classified.
    """
    bus = EventBus()
    events: list[GuardStateEvent] = []
    bus.subscribe(GuardStateEvent, events.append)
    engine = GuardEngine(bus, get_profile("boxing"), _CALIBRATION, GuardConfig())

    for i in range(2 * _FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True, True), timestamp_s=i / _FPS))
    assert len(events) == 2  # one initial "up" event per hand
    events.clear()

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    # The new "A" also happens to hold guard up.
    for i in range(_FPS):
        t = 2.0 + i / _FPS
        engine.process(TrackedPose(fighter_id="A", pose=_pose(True, True), timestamp_s=t))
    assert len(events) == 2


def test_missing_face_keypoints_produce_no_events() -> None:
    """Without a nose or both shoulders, there is no guard line to compare against."""
    bus = EventBus()
    events: list[GuardStateEvent] = []
    bus.subscribe(GuardStateEvent, events.append)
    engine = GuardEngine(bus, get_profile("boxing"), _CALIBRATION, GuardConfig())
    for i in range(_FPS):
        pose = _pose(True, True, missing=KeypointName.NOSE)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i / _FPS))
    assert events == []
