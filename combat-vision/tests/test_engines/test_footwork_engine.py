"""Footwork engine tests — synthetic step sequences."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.footwork import FootworkEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FootworkSample,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    StepEvent,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import FootworkConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_FPS = 60
_STEP_NORM = 0.2  # left ankle travels 0.2 of frame width = 256 px = 0.512 m


def _pose(left_ankle_x: float) -> Pose:
    return Pose(
        keypoints={
            KeypointName.LEFT_HIP: Keypoint(x=0.53, y=0.60),
            KeypointName.RIGHT_HIP: Keypoint(x=0.47, y=0.60),
            KeypointName.LEFT_ANKLE: Keypoint(x=left_ankle_x, y=0.90),
            KeypointName.RIGHT_ANKLE: Keypoint(x=0.44, y=0.90),
        }
    )


def _step_sequence() -> list[TrackedPose]:
    """Still 0.5s -> left foot slides 0.2 over 0.3s -> still 0.5s."""
    poses = []
    t = 0.0
    for _ in range(int(0.5 * _FPS)):
        poses.append(TrackedPose(fighter_id="A", pose=_pose(0.55), timestamp_s=t))
        t += 1 / _FPS
    move_frames = int(0.3 * _FPS)
    for i in range(move_frames):
        x = 0.55 + _STEP_NORM * (i + 1) / move_frames
        poses.append(TrackedPose(fighter_id="A", pose=_pose(x), timestamp_s=t))
        t += 1 / _FPS
    for _ in range(int(0.5 * _FPS)):
        poses.append(TrackedPose(fighter_id="A", pose=_pose(0.55 + _STEP_NORM), timestamp_s=t))
        t += 1 / _FPS
    return poses


def _run(poses: list[TrackedPose]) -> tuple[list[StepEvent], list[FootworkSample], FootworkEngine]:
    bus = EventBus()
    steps: list[StepEvent] = []
    samples: list[FootworkSample] = []
    bus.subscribe(StepEvent, steps.append)
    bus.subscribe(FootworkSample, samples.append)
    engine = FootworkEngine(bus, get_profile("boxing"), _CALIBRATION, FootworkConfig())
    for pose in poses:
        engine.process(pose)
    engine.finish()
    return steps, samples, engine


def test_single_step_detected_with_correct_displacement() -> None:
    """One slide-and-plant of the left foot -> exactly one StepEvent."""
    steps, _, _ = _run(_step_sequence())

    assert len(steps) == 1
    step = steps[0]
    assert step.foot == Limb.LEFT_FOOT
    expected_m = _STEP_NORM * _CALIBRATION.frame_width_px * 0.002
    assert abs(step.displacement - expected_m) < 0.05 * expected_m
    assert step.from_xy[0] < step.to_xy[0]


def test_still_feet_take_no_steps() -> None:
    """A stationary fighter must produce zero StepEvents."""
    poses = [
        TrackedPose(fighter_id="A", pose=_pose(0.55), timestamp_s=i / _FPS)
        for i in range(2 * _FPS)
    ]
    steps, _, _ = _run(poses)
    assert steps == []


def test_footwork_samples_and_heatmap_accumulate() -> None:
    """Samples flow at the decimated rate and the heat map fills in."""
    steps, samples, engine = _run(_step_sequence())

    assert len(samples) >= 3  # 1.3 s of stream at one sample per 0.2 s
    assert all(-1.0 <= s.weight_shift <= 1.0 for s in samples)
    assert all(s.stance_width > 0 for s in samples)

    heat = engine.heatmap("A")
    assert heat is not None
    assert heat.sum() > 0
    assert engine.heatmap("B") is None
