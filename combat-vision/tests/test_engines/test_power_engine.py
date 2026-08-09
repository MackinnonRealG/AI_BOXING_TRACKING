"""Power engine tests, driven by the jab fixture via the speed engine."""

from __future__ import annotations

from combat_vision.engines.power import PowerEngine
from combat_vision.engines.speed import SpeedEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import Limb, PowerEstimateEvent, TrackedPose
from combat_vision.sports import get_profile
from combat_vision.utils.config import PowerEngineConfig, SpeedEngineConfig
from tests.conftest import calibration_from_meta


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
