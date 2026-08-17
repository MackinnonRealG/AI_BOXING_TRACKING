"""Kick-balance engine: base-leg stability during kicks and knee strikes.

The standing ("base") leg should stay planted while the other leg kicks —
a base ankle that drifts laterally during the stroke means the fighter is
fighting to stay upright rather than committing power through a stable
base, and it telegraphs the kick and limits any follow-up. This engine
measures the base ankle's horizontal range of motion during a kick/knee
stroke window, normalized by hip width at the stroke's start (scale-
invariant against distance from the camera — the same normalization
:mod:`engines.elbow` uses for elbow flare, not raw pixels):

* strokes whose base-ankle wobble ratio clears ``wobble_ratio`` publish a
  :class:`BalanceFaultEvent` — the base leg moved more than a planted foot
  should.
* strokes at or below that ratio publish a :class:`CleanTechniqueEvent`
  (``check="base_balance"``) — the good reps are logged too, not just the
  faults.

Hand strikes are ignored entirely — this is a kicking-sport concept with no
boxing analogue. In a boxing session this engine never fires anything,
since the speed engine never emits foot/knee candidates when the active
sport profile doesn't monitor those limbs — the filter here is a defensive
belt-and-braces check, not something load-bearing on its own.
"""

from __future__ import annotations

from collections import defaultdict, deque

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    BalanceFaultEvent,
    CleanTechniqueEvent,
    FighterId,
    FighterRelabeledEvent,
    KeypointName,
    Limb,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import KickBalanceConfig

_BUFFER_FRAMES = 240  # ~3s at 60 FPS; must outlast max_event_duration_s
_KICK_LIMBS = (Limb.LEFT_FOOT, Limb.RIGHT_FOOT, Limb.LEFT_KNEE, Limb.RIGHT_KNEE)

# The base (standing) leg's ankle, keyed by which limb is striking.
_BASE_ANKLE: dict[Limb, KeypointName] = {
    Limb.LEFT_FOOT: KeypointName.RIGHT_ANKLE,
    Limb.RIGHT_FOOT: KeypointName.LEFT_ANKLE,
    Limb.LEFT_KNEE: KeypointName.RIGHT_ANKLE,
    Limb.RIGHT_KNEE: KeypointName.LEFT_ANKLE,
}


class KickBalanceEngine(MetricsEngine):
    """Flags kicks/knees thrown with excessive base-leg lateral wobble."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: KickBalanceConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._buffers: dict[FighterId, deque[TrackedPose]] = defaultdict(
            lambda: deque(maxlen=_BUFFER_FRAMES)
        )
        bus.subscribe(SpeedPeakEvent, self._on_candidate)
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop buffered poses for a label now held by a different person.

        Buffered poses are keyed by frame timestamp, not by who's wearing
        the label — without clearing this, a kick thrown moments after the
        relabel would window over a mix of the departed fighter's tail-end
        poses and the new fighter's poses, measuring wobble across two
        different people's legs (same failure mode already fixed in
        engines.rotation, engines.power, engines.strike_classifier, and
        engines.knee_bend).
        """
        self._buffers.pop(event.fighter_id, None)

    def process(self, tracked: TrackedPose) -> None:
        """Buffer poses so stroke windows are available when candidates fire."""
        self._buffers[tracked.fighter_id].append(tracked)

    def _on_candidate(self, event: SpeedPeakEvent) -> None:
        """Measure base-ankle wobble over the stroke; flag it if excessive."""
        if event.limb not in _KICK_LIMBS:
            return
        window = [
            p
            for p in self._buffers[event.fighter_id]
            if event.start_s <= p.timestamp_s <= event.end_s
        ]
        if len(window) < 2:
            return

        hip_width = self._hip_width(window[0])
        if hip_width is None or hip_width <= 0:
            return

        base_ankle_name = _BASE_ANKLE[event.limb]
        xs = [
            kp.x
            for tracked in window
            if (kp := tracked.pose.get(base_ankle_name)) is not None
        ]
        if len(xs) < 2:
            return

        wobble_ratio = (max(xs) - min(xs)) / hip_width
        if wobble_ratio <= self._config.wobble_ratio:
            self._bus.publish(
                CleanTechniqueEvent(
                    timestamp_s=event.end_s,
                    fighter_id=event.fighter_id,
                    check="base_balance",
                    limb=event.limb,
                )
            )
            return

        self._bus.publish(
            BalanceFaultEvent(
                timestamp_s=event.end_s,
                fighter_id=event.fighter_id,
                limb=event.limb,
                wobble_ratio=wobble_ratio,
            )
        )

    def _hip_width(self, tracked: TrackedPose) -> float | None:
        """Normalized hip separation at the stroke's start, or None if unknown."""
        l_hip = tracked.pose.get(KeypointName.LEFT_HIP)
        r_hip = tracked.pose.get(KeypointName.RIGHT_HIP)
        if l_hip is None or r_hip is None:
            return None
        return abs(l_hip.x - r_hip.x)
