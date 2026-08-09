"""Stance engine: orthodox/southpaw/square detection and switch logging.

Algorithm:

1. Facing direction: the nose leads the body — ``sign(nose_x - hip_mid_x)``
   gives the horizontal direction the fighter faces (side-on camera).
2. Square check: if the ankles' horizontal separation is below
   ``square_width_ratio`` × shoulder width, the fighter is SQUARE.
3. Otherwise the *lead* foot is the ankle farthest along the facing
   direction: left foot lead = ORTHODOX, right foot lead = SOUTHPAW.
4. Debounce: a new classification must persist for ``switch_debounce_s``
   before being accepted (steps shuffle through square constantly).

On every accepted change a :class:`StanceSample` is published (consumed by
the strike classifier and the overlay); genuine orthodox↔southpaw changes
additionally publish a :class:`StanceSwitchEvent` with timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    KeypointName,
    Stance,
    StanceSample,
    StanceSwitchEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import StanceConfig


@dataclass
class _FighterStance:
    """Debounce state for one fighter."""

    current: Stance | None = None
    candidate: Stance | None = None
    candidate_since_s: float = 0.0


class StanceEngine(MetricsEngine):
    """Classifies stance per fighter and logs every switch with a timestamp."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: StanceConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._fighters: dict[FighterId, _FighterStance] = {}

    def process(self, tracked: TrackedPose) -> None:
        """Classify this frame's stance and run the debounce state machine."""
        stance = self._classify(tracked)
        if stance is None:
            return
        state = self._fighters.setdefault(tracked.fighter_id, _FighterStance())

        if stance != state.candidate:
            state.candidate = stance
            state.candidate_since_s = tracked.timestamp_s
            return
        stable_for = tracked.timestamp_s - state.candidate_since_s
        if stance == state.current or stable_for < self._config.switch_debounce_s:
            return

        previous = state.current
        state.current = stance
        self._bus.publish(
            StanceSample(
                timestamp_s=tracked.timestamp_s, fighter_id=tracked.fighter_id, stance=stance
            )
        )
        # A switch is specifically orthodox <-> southpaw; passing through
        # square (or the initial classification) is not a switch.
        lead_stances = (Stance.ORTHODOX, Stance.SOUTHPAW)
        if previous in lead_stances and stance in lead_stances:
            self._bus.publish(
                StanceSwitchEvent(
                    timestamp_s=tracked.timestamp_s,
                    fighter_id=tracked.fighter_id,
                    from_stance=previous,
                    to_stance=stance,
                )
            )

    def _classify(self, tracked: TrackedPose) -> Stance | None:
        """One-frame stance classification, or None if keypoints are missing."""
        pose = tracked.pose
        needed = (
            KeypointName.NOSE,
            KeypointName.LEFT_SHOULDER,
            KeypointName.RIGHT_SHOULDER,
            KeypointName.LEFT_HIP,
            KeypointName.RIGHT_HIP,
            KeypointName.LEFT_ANKLE,
            KeypointName.RIGHT_ANKLE,
        )
        points = {name: pose.get(name) for name in needed}
        if any(kp is None for kp in points.values()):
            return None
        nose = points[KeypointName.NOSE]
        l_sh, r_sh = points[KeypointName.LEFT_SHOULDER], points[KeypointName.RIGHT_SHOULDER]
        l_hip, r_hip = points[KeypointName.LEFT_HIP], points[KeypointName.RIGHT_HIP]
        l_ankle, r_ankle = points[KeypointName.LEFT_ANKLE], points[KeypointName.RIGHT_ANKLE]
        assert nose and l_sh and r_sh and l_hip and r_hip and l_ankle and r_ankle

        shoulder_width = abs(l_sh.x - r_sh.x)
        ankle_separation = abs(l_ankle.x - r_ankle.x)
        if shoulder_width > 0 and (
            ankle_separation < self._config.square_width_ratio * shoulder_width
        ):
            return Stance.SQUARE

        hip_mid_x = (l_hip.x + r_hip.x) / 2
        facing = nose.x - hip_mid_x
        if facing == 0:
            return None  # perfectly ambiguous frame; wait for a better one
        # The lead foot is the ankle farthest along the facing direction.
        left_lead = (l_ankle.x - r_ankle.x) * facing > 0
        return Stance.ORTHODOX if left_lead else Stance.SOUTHPAW
