"""Distance engine tests — two synthetic fighters at a known range."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.distance import DistanceEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import DistanceSample, Keypoint, KeypointName, Pose, TrackedPose
from combat_vision.sports import get_profile
from combat_vision.utils.config import DistanceEngineConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_FPS = 30


def _pose(hip_mid_x: float) -> Pose:
    return Pose(
        keypoints={
            KeypointName.LEFT_HIP: Keypoint(x=hip_mid_x + 0.03, y=0.60),
            KeypointName.RIGHT_HIP: Keypoint(x=hip_mid_x - 0.03, y=0.60),
        }
    )


def test_two_fighters_produce_distance_samples() -> None:
    """Fighters 0.3 frame-widths apart -> samples at the known distance."""
    bus = EventBus()
    samples: list[DistanceSample] = []
    bus.subscribe(DistanceSample, samples.append)
    engine = DistanceEngine(bus, get_profile("boxing"), _CALIBRATION, DistanceEngineConfig())

    for i in range(_FPS):  # one second, both fighters visible each frame
        t = i / _FPS
        engine.process(TrackedPose(fighter_id="A", pose=_pose(0.40), timestamp_s=t))
        engine.process(TrackedPose(fighter_id="B", pose=_pose(0.70), timestamp_s=t))

    assert 4 <= len(samples) <= 7  # ~one per 0.2 s over 1 s
    expected_m = 0.30 * _CALIBRATION.frame_width_px * 0.002
    for sample in samples:
        assert (sample.fighter_id, sample.other_fighter_id) == ("A", "B")
        assert abs(sample.distance - expected_m) < 0.02 * expected_m


def test_single_fighter_produces_no_samples() -> None:
    """Distance is pairwise — one fighter alone yields nothing."""
    bus = EventBus()
    samples: list[DistanceSample] = []
    bus.subscribe(DistanceSample, samples.append)
    engine = DistanceEngine(bus, get_profile("boxing"), _CALIBRATION, DistanceEngineConfig())
    for i in range(_FPS):
        engine.process(TrackedPose(fighter_id="A", pose=_pose(0.40), timestamp_s=i / _FPS))
    assert samples == []
