"""Speed engine tests — recorded pose sequences, no camera required."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.speed import SpeedEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import Limb, SpeedPeakEvent, SpeedUnit, TrackedPose
from combat_vision.sports import get_profile
from combat_vision.utils.config import SpeedEngineConfig
from tests.conftest import calibration_from_meta


def _run_engine(
    poses: list[TrackedPose], calibration: Calibration
) -> list[SpeedPeakEvent]:
    """Replay a pose sequence through a fresh engine, collecting its events."""
    bus = EventBus()
    events: list[SpeedPeakEvent] = []
    bus.subscribe(SpeedPeakEvent, events.append)
    engine = SpeedEngine(bus, get_profile("boxing"), calibration, SpeedEngineConfig())
    for pose in poses:
        engine.process(pose)
    engine.finish()
    return events


def test_detects_single_jab(jab_sequence: tuple[list[TrackedPose], dict]) -> None:
    """One out-and-back jab must produce exactly one candidate on the left hand.

    The retraction stroke is equally fast but must be debounced — a jab is
    one punch, not two.
    """
    poses, meta = jab_sequence
    events = _run_engine(poses, calibration_from_meta(meta))

    assert len(events) == 1
    event = events[0]
    assert event.fighter_id == "A"
    assert event.limb == Limb.LEFT_HAND
    assert event.unit == SpeedUnit.METERS_PER_SECOND
    assert event.start_s < event.end_s


def test_peak_speed_accuracy(jab_sequence: tuple[list[TrackedPose], dict]) -> None:
    """Measured peak must be close to the fixture's known true peak.

    The 5-frame moving average attenuates a sinusoidal peak by ~5%, so we
    accept 80–102% of ground truth.
    """
    poses, meta = jab_sequence
    events = _run_engine(poses, calibration_from_meta(meta))

    true_peak = meta["true_peak_speed_mps"]
    assert len(events) == 1
    assert 0.80 * true_peak <= events[0].peak_speed <= 1.02 * true_peak


def test_idle_sequence_emits_nothing(idle_sequence: tuple[list[TrackedPose], dict]) -> None:
    """Estimator jitter on a still fighter must never register as a punch."""
    poses, meta = idle_sequence
    events = _run_engine(poses, calibration_from_meta(meta))
    assert events == []


def test_uncalibrated_falls_back_to_pixels(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """Without calibration the same jab is still detected, reported in px/s."""
    poses, meta = jab_sequence
    events = _run_engine(poses, calibration_from_meta(meta, calibrated=False))

    assert len(events) == 1
    event = events[0]
    assert event.unit == SpeedUnit.PIXELS_PER_SECOND
    # px/s peak should be the m/s peak divided by the metres-per-pixel scale.
    expected_pxps = meta["true_peak_speed_mps"] / meta["metres_per_pixel"]
    assert 0.80 * expected_pxps <= event.peak_speed <= 1.02 * expected_pxps


def test_right_hand_untouched(jab_sequence: tuple[list[TrackedPose], dict]) -> None:
    """A left jab must not generate events for the stationary right hand."""
    poses, meta = jab_sequence
    events = _run_engine(poses, calibration_from_meta(meta))
    assert all(e.limb == Limb.LEFT_HAND for e in events)


def test_calibrating_mid_stroke_does_not_corrupt_peak_speed(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """Completing calibration mid-punch must not mix px/s and m/s samples.

    Buffered speed samples must stay in raw pixels internally and only be
    scaled to the *current* calibration at comparison/publish time — if a
    stale px/s sample were ever averaged together with an already-scaled
    m/s sample (or vice versa), the smoothed/peak speed would be nonsense.
    """
    poses, meta = jab_sequence
    calibration = calibration_from_meta(meta, calibrated=False)
    bus = EventBus()
    events: list[SpeedPeakEvent] = []
    bus.subscribe(SpeedPeakEvent, events.append)
    engine = SpeedEngine(bus, get_profile("boxing"), calibration, SpeedEngineConfig())

    # The fixture's qualifying stroke runs roughly frames 34-46 (uncalibrated
    # thresholds); calibrating at frame 40 lands squarely inside it.
    midpoint = 40
    for pose in poses[:midpoint]:
        engine.process(pose)
    calibration.set_scale(meta["metres_per_pixel"])  # calibrate mid-stroke
    for pose in poses[midpoint:]:
        engine.process(pose)
    engine.finish()

    true_peak = meta["true_peak_speed_mps"]
    assert len(events) == 1
    assert events[0].unit == SpeedUnit.METERS_PER_SECOND
    assert 0.70 * true_peak <= events[0].peak_speed <= 1.10 * true_peak
