"""Stance engine tests — synthetic pose sequences, no camera required."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.stance import StanceEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    Keypoint,
    KeypointName,
    Pose,
    Stance,
    StanceSample,
    StanceSwitchEvent,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import StanceConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_FPS = 60


def _pose(orthodox: bool) -> Pose:
    """A fighter facing +x with the left (orthodox) or right (southpaw) foot lead."""
    left_ankle_x, right_ankle_x = (0.55, 0.44) if orthodox else (0.44, 0.55)
    return Pose(
        keypoints={
            KeypointName.NOSE: Keypoint(x=0.52, y=0.20),
            KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35),
            KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35),
            KeypointName.LEFT_HIP: Keypoint(x=0.53, y=0.60),
            KeypointName.RIGHT_HIP: Keypoint(x=0.47, y=0.60),
            KeypointName.LEFT_ANKLE: Keypoint(x=left_ankle_x, y=0.90),
            KeypointName.RIGHT_ANKLE: Keypoint(x=right_ankle_x, y=0.90),
        }
    )


def _run(sequence: list[tuple[float, bool]]) -> tuple[list[StanceSample], list[StanceSwitchEvent]]:
    """Feed (timestamp, orthodox?) frames through a fresh engine."""
    bus = EventBus()
    samples: list[StanceSample] = []
    switches: list[StanceSwitchEvent] = []
    bus.subscribe(StanceSample, samples.append)
    bus.subscribe(StanceSwitchEvent, switches.append)
    engine = StanceEngine(bus, get_profile("boxing"), _CALIBRATION, StanceConfig())
    for t, orthodox in sequence:
        engine.process(TrackedPose(fighter_id="A", pose=_pose(orthodox), timestamp_s=t))
    return samples, switches


def test_initial_stance_is_sampled_not_switched() -> None:
    """A held orthodox stance yields one sample and no switch events."""
    frames = [(i / _FPS, True) for i in range(2 * _FPS)]
    samples, switches = _run(frames)
    assert [s.stance for s in samples] == [Stance.ORTHODOX]
    assert switches == []


def test_switch_to_southpaw_is_logged_once_with_timestamp() -> None:
    """Orthodox for 1s then southpaw for 1s -> exactly one logged switch."""
    frames = [(i / _FPS, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False) for i in range(_FPS)]
    samples, switches = _run(frames)

    assert [s.stance for s in samples] == [Stance.ORTHODOX, Stance.SOUTHPAW]
    assert len(switches) == 1
    switch = switches[0]
    assert switch.from_stance == Stance.ORTHODOX
    assert switch.to_stance == Stance.SOUTHPAW
    # Accepted only after the debounce window, at the earliest at t = 1.5s.
    assert switch.timestamp_s >= 1.0 + StanceConfig().switch_debounce_s


def test_brief_flicker_is_debounced() -> None:
    """A 5-frame southpaw flicker must not register as a switch."""
    frames = [(i / _FPS, True) for i in range(_FPS)]
    frames += [(1.0 + i / _FPS, False) for i in range(5)]
    frames += [(1.0 + 5 / _FPS + i / _FPS, True) for i in range(_FPS)]
    _, switches = _run(frames)
    assert switches == []
