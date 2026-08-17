"""Kick-balance engine tests — synthetic pose sequences, no camera required.

SpeedPeakEvent candidates are published directly onto the bus (rather than
produced by the speed engine) so each test controls exactly the base-ankle
geometry under evaluation, per the "engines are bus-isolated, independently
testable" design rule already used by test_rotation_engine.py.
"""

from __future__ import annotations

import pytest

from combat_vision.calibration import Calibration
from combat_vision.engines.kick_balance import KickBalanceEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    BalanceFaultEvent,
    CleanTechniqueEvent,
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    Limb,
    Pose,
    SpeedPeakEvent,
    SpeedUnit,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import KickBalanceConfig

_CALIBRATION = Calibration(metres_per_pixel=0.002, frame_width_px=1280, frame_height_px=720)
_MPS = SpeedUnit.METERS_PER_SECOND
# Hips at 0.45/0.55 -> hip_width = 0.10 (matches the convention test_elbow_engine.py uses).
_HIP_WIDTH = 0.10


def _pose(base_ankle_x: float, kicking_ankle_y: float = 0.30) -> Pose:
    """A fighter kicking with the left foot; ``base_ankle_x`` controls the
    right (base) ankle's horizontal position, the thing under test."""
    return Pose(
        keypoints={
            KeypointName.LEFT_HIP: Keypoint(x=0.55, y=0.60),
            KeypointName.RIGHT_HIP: Keypoint(x=0.45, y=0.60),
            KeypointName.RIGHT_ANKLE: Keypoint(x=base_ankle_x, y=0.90),
            KeypointName.LEFT_ANKLE: Keypoint(x=0.60, y=kicking_ankle_y),
        }
    )


def _run(base_ankle_xs: list[float], limb: Limb = Limb.LEFT_FOOT) -> tuple[
    list[BalanceFaultEvent], list[CleanTechniqueEvent]
]:
    """Feed one pose per element of ``base_ankle_xs``, 0.1s apart, then fire
    a matching kick candidate spanning the whole window."""
    bus = EventBus()
    faults: list[BalanceFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(BalanceFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = KickBalanceEngine(bus, get_profile("kickboxing"), _CALIBRATION, KickBalanceConfig())

    for i, x in enumerate(base_ankle_xs):
        t = i * 0.1
        engine.process(TrackedPose(fighter_id="A", pose=_pose(x), timestamp_s=t))
    end_s = (len(base_ankle_xs) - 1) * 0.1
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=end_s,
            fighter_id="A",
            limb=limb,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=end_s,
        )
    )
    return faults, clean


def test_stable_base_leg_publishes_clean_technique() -> None:
    """Base ankle held still (x never changes) -> zero wobble -> clean."""
    faults, clean = _run([0.45, 0.45, 0.45])
    assert faults == []
    assert len(clean) == 1
    assert clean[0].check == "base_balance"
    assert clean[0].limb == Limb.LEFT_FOOT


def test_wobbling_base_leg_is_flagged() -> None:
    """Base ankle drifting by a full hip-width during the kick -> fault."""
    faults, clean = _run([0.40, 0.50, 0.40])  # range 0.10 == hip width -> ratio 1.0
    assert clean == []
    assert len(faults) == 1
    assert faults[0].limb == Limb.LEFT_FOOT
    assert faults[0].wobble_ratio == pytest.approx(1.0)


def test_right_foot_kick_checks_the_left_ankle_as_base() -> None:
    """A right-foot kick's base leg is the left ankle, not the right."""
    bus = EventBus()
    faults: list[BalanceFaultEvent] = []
    bus.subscribe(BalanceFaultEvent, faults.append)
    engine = KickBalanceEngine(bus, get_profile("kickboxing"), _CALIBRATION, KickBalanceConfig())

    # Right ankle (the kicking foot here) swings wildly -- must be ignored.
    # Left ankle (the actual base for a right-foot kick) stays fixed at 0.60.
    poses = [
        Pose(
            keypoints={
                KeypointName.LEFT_HIP: Keypoint(x=0.55, y=0.60),
                KeypointName.RIGHT_HIP: Keypoint(x=0.45, y=0.60),
                KeypointName.LEFT_ANKLE: Keypoint(x=0.60, y=0.90),
                KeypointName.RIGHT_ANKLE: Keypoint(x=x, y=0.30),
            }
        )
        for x in (0.30, 0.60, 0.30)
    ]
    for i, pose in enumerate(poses):
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i * 0.1))
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
    assert faults == []  # the swinging ankle was the kicking foot, not the base


def test_hand_strikes_are_ignored() -> None:
    faults, clean = _run([0.40, 0.50, 0.40], limb=Limb.LEFT_HAND)
    assert faults == []
    assert clean == []


def test_missing_hip_keypoints_produce_no_event() -> None:
    bus = EventBus()
    faults: list[BalanceFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(BalanceFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = KickBalanceEngine(bus, get_profile("kickboxing"), _CALIBRATION, KickBalanceConfig())

    pose = Pose(
        keypoints={
            KeypointName.RIGHT_ANKLE: Keypoint(x=0.45, y=0.90),
            KeypointName.LEFT_ANKLE: Keypoint(x=0.60, y=0.30),
        }
    )
    engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=0.0))
    engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=0.1))
    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.1,
            fighter_id="A",
            limb=Limb.LEFT_FOOT,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,
            end_s=0.1,
        )
    )
    assert faults == []
    assert clean == []


def test_relabel_clears_the_pose_buffer_so_kicks_do_not_mix_fighters() -> None:
    """A kick thrown right after a relabel must not window over the
    departed fighter's poses when measuring base-ankle wobble.
    """
    bus = EventBus()
    faults: list[BalanceFaultEvent] = []
    clean: list[CleanTechniqueEvent] = []
    bus.subscribe(BalanceFaultEvent, faults.append)
    bus.subscribe(CleanTechniqueEvent, clean.append)
    engine = KickBalanceEngine(bus, get_profile("kickboxing"), _CALIBRATION, KickBalanceConfig())

    # Departed fighter "A": base ankle at 0.30, buffered at t=0.0.
    engine.process(TrackedPose(fighter_id="A", pose=_pose(0.30), timestamp_s=0.0))

    engine._on_relabeled(FighterRelabeledEvent(timestamp_s=0.0, fighter_id="A"))

    # New fighter "A": base ankle rock-steady at 0.45 for their whole real window.
    engine.process(TrackedPose(fighter_id="A", pose=_pose(0.45), timestamp_s=0.5))
    engine.process(TrackedPose(fighter_id="A", pose=_pose(0.45), timestamp_s=0.6))

    bus.publish(
        SpeedPeakEvent(
            timestamp_s=0.6,
            fighter_id="A",
            limb=Limb.LEFT_FOOT,
            peak_speed=6.0,
            unit=_MPS,
            start_s=0.0,  # spans the relabel boundary if the old pose were still buffered
            end_s=0.6,
        )
    )

    assert faults == []
    assert len(clean) == 1  # correctly judged from the new fighter's steady base leg alone
