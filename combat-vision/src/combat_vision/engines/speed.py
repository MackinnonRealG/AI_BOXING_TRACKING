"""Speed engine — fully implemented.

Measures hand (wrist) speed per fighter and detects punch candidates as
speed *strokes*:

1. Per (fighter, wrist), convert normalized wrist coordinates to pixels and
   compute frame-to-frame speed, then smooth it with a short moving average
   (``smoothing_window`` frames) to suppress pose-estimator jitter.
2. A stroke opens when smoothed speed exceeds ``start_speed`` and closes when
   it drops below ``end_speed_ratio * peak`` (hysteresis — exactly one event
   per punch, no retriggering on noise).
3. A closed stroke is published as a
   :class:`~combat_vision.events.types.SpeedPeakEvent` if its peak cleared
   ``peak_min_speed``, it did not exceed ``max_event_duration_s`` (guards
   against tracking glides), and it is not within ``min_event_interval_s`` of
   the previous event for that wrist (debounce).

Units: metres/second when calibrated, pixels/second otherwise — thresholds
switch with the calibration state so both paths behave consistently.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    Keypoint,
    KeypointName,
    Limb,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import SpeedEngineConfig

logger = logging.getLogger(__name__)

_LIMB_KEYPOINTS: dict[Limb, KeypointName] = {
    Limb.LEFT_HAND: KeypointName.LEFT_WRIST,
    Limb.RIGHT_HAND: KeypointName.RIGHT_WRIST,
    Limb.LEFT_FOOT: KeypointName.LEFT_ANKLE,
    Limb.RIGHT_FOOT: KeypointName.RIGHT_ANKLE,
    Limb.LEFT_KNEE: KeypointName.LEFT_KNEE,
    Limb.RIGHT_KNEE: KeypointName.RIGHT_KNEE,
}


@dataclass
class _WristState:
    """Per-(fighter, wrist) stroke-detection state.

    Speeds are buffered raw, in pixels/second — never pre-scaled to the
    calibration's unit. Live calibration can complete or change mid-stroke,
    and scaling at write-time would leave already-scaled samples stranded in
    a stale unit once that happens; scaling only at read-time (see
    :meth:`SpeedEngine._step`) keeps every sample consistent with whatever
    calibration is current right now.
    """

    last_position_px: tuple[float, float] | None = None
    last_timestamp_s: float | None = None
    recent_speeds_pxps: deque[float] = field(default_factory=deque)
    in_stroke: bool = False
    stroke_start_s: float = 0.0
    peak_speed_pxps: float = 0.0
    last_event_end_s: float = -math.inf


class SpeedEngine(MetricsEngine):
    """Wrist velocity measurement and punch-candidate detection."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: SpeedEngineConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._states: dict[tuple[FighterId, Limb], _WristState] = {}

    @property
    def _start_speed(self) -> float:
        """Stroke-start threshold in the current calibration's unit.

        Read per call, not cached: live calibration can flip the unit
        mid-session and thresholds must follow immediately.
        """
        if self._calibration.is_calibrated:
            return self._config.start_speed_mps
        return self._config.start_speed_pxps

    @property
    def _peak_min_speed(self) -> float:
        """Minimum qualifying peak in the current calibration's unit."""
        if self._calibration.is_calibrated:
            return self._config.peak_min_speed_mps
        return self._config.peak_min_speed_pxps

    def process(self, tracked: TrackedPose) -> None:
        """Update stroke state for every striking limb of this fighter."""
        for limb, keypoint_name in _LIMB_KEYPOINTS.items():
            if limb not in self._profile.striking_limbs:
                continue
            wrist = tracked.pose.get(keypoint_name)
            if wrist is None:
                continue
            state = self._states.setdefault((tracked.fighter_id, limb), _WristState())
            self._step(state, tracked.fighter_id, limb, wrist, tracked.timestamp_s)

    def finish(self) -> None:
        """Close any stroke still open at end of stream."""
        for (fighter_id, limb), state in self._states.items():
            if state.in_stroke and state.last_timestamp_s is not None:
                self._close_stroke(state, fighter_id, limb, state.last_timestamp_s)

    def _step(
        self,
        state: _WristState,
        fighter_id: FighterId,
        limb: Limb,
        wrist: Keypoint,
        timestamp_s: float,
    ) -> None:
        """Advance the state machine by one sample."""
        position_px = self._calibration.to_pixels(wrist.x, wrist.y)

        if state.last_position_px is None or state.last_timestamp_s is None:
            state.last_position_px, state.last_timestamp_s = position_px, timestamp_s
            return
        dt = timestamp_s - state.last_timestamp_s
        if dt <= 0:
            return

        distance_px = math.hypot(
            position_px[0] - state.last_position_px[0],
            position_px[1] - state.last_position_px[1],
        )
        speed_pxps = distance_px / dt
        state.last_position_px, state.last_timestamp_s = position_px, timestamp_s

        state.recent_speeds_pxps.append(speed_pxps)
        while len(state.recent_speeds_pxps) > self._config.smoothing_window:
            state.recent_speeds_pxps.popleft()
        smoothed_pxps = sum(state.recent_speeds_pxps) / len(state.recent_speeds_pxps)
        smoothed = self._calibration.scale_speed(smoothed_pxps)

        if not state.in_stroke:
            if smoothed >= self._start_speed:
                state.in_stroke = True
                state.stroke_start_s = timestamp_s
                state.peak_speed_pxps = smoothed_pxps
            return

        state.peak_speed_pxps = max(state.peak_speed_pxps, smoothed_pxps)
        peak_speed = self._calibration.scale_speed(state.peak_speed_pxps)
        stroke_over = smoothed <= self._config.end_speed_ratio * peak_speed
        too_long = timestamp_s - state.stroke_start_s > self._config.max_event_duration_s
        if stroke_over or too_long:
            self._close_stroke(state, fighter_id, limb, timestamp_s)

    def _close_stroke(
        self, state: _WristState, fighter_id: FighterId, limb: Limb, end_s: float
    ) -> None:
        """End the current stroke, publishing an event if it qualifies."""
        state.in_stroke = False
        duration = end_s - state.stroke_start_s
        debounced = state.stroke_start_s - state.last_event_end_s < (
            self._config.min_event_interval_s
        )
        peak_speed = self._calibration.scale_speed(state.peak_speed_pxps)
        qualifies = (
            peak_speed >= self._peak_min_speed
            and duration <= self._config.max_event_duration_s
            and not debounced
        )
        if not qualifies:
            state.peak_speed_pxps = 0.0
            return

        state.last_event_end_s = end_s
        event = SpeedPeakEvent(
            timestamp_s=end_s,
            fighter_id=fighter_id,
            limb=limb,
            peak_speed=peak_speed,
            unit=self._calibration.unit,
            start_s=state.stroke_start_s,
            end_s=end_s,
        )
        state.peak_speed_pxps = 0.0
        logger.debug(
            "punch candidate fighter=%s limb=%s peak=%.2f%s",
            fighter_id,
            limb,
            event.peak_speed,
            event.unit,
        )
        self._bus.publish(event)
