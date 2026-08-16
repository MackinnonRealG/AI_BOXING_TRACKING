"""Guard engine: per-hand guard-height fault detection.

Boxing fundamentals hold both hands near the chin between and during
exchanges. This engine estimates each hand's "guard up" state every frame
from face and shoulder geometry alone — there is no separate chin landmark
in the canonical keypoint set:

1. **Reference chin line** — interpolated between the nose and the shoulder
   line by ``chin_ratio`` (the chin sits below the nose, above the
   shoulders).
2. **Guard line** — the chin line pushed down by ``drop_margin_ratio`` of
   the nose-to-shoulder span, so a hand resting slightly below chin height
   (a normal high guard) still reads as "up".
3. **Debounce** — a hand must be below the guard line for
   ``drop_debounce_s`` before the state flips to "down" (a single punch's
   own extension dips the hand briefly; only a sustained drop is a fault),
   and above it for ``recover_debounce_s`` before flipping back to "up".

Deliberately does not exempt the currently-punching hand: a hand that stays
down through a whole combination is exactly the "not resetting your guard"
fault a coach would flag, punch or no punch. What v1 cannot yet do is tell a
genuine drop from a hand held wide at head height (e.g. mid-hook) — that
needs a horizontal/lateral signal this engine does not compute.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    FighterRelabeledEvent,
    GuardStateEvent,
    KeypointName,
    Limb,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import GuardConfig

_HAND_KEYPOINT: dict[Limb, KeypointName] = {
    Limb.LEFT_HAND: KeypointName.LEFT_WRIST,
    Limb.RIGHT_HAND: KeypointName.RIGHT_WRIST,
}


@dataclass
class _HandGuard:
    """Debounce state for one fighter's one hand."""

    current_up: bool | None = None
    candidate_up: bool | None = None
    candidate_since_s: float = 0.0


class GuardEngine(MetricsEngine):
    """Classifies each striking hand as guard-up/guard-down, with debounce."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: GuardConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._hands: dict[tuple[FighterId, Limb], _HandGuard] = {}
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop debounce state for a label now held by a different person.

        Without this, a relabeled fighter starting in the same guard state
        the departed one left behind would hit the ``up_now == current_up``
        short-circuit in :meth:`_update` and never get an initial
        :class:`GuardStateEvent` — leaving the overlay showing stale or
        missing guard status for someone who never actually had it.
        """
        for hand in _HAND_KEYPOINT:
            self._hands.pop((event.fighter_id, hand), None)

    def process(self, tracked: TrackedPose) -> None:
        """Classify this frame's guard line and run the debounce state machine."""
        line_y = self._guard_line_y(tracked)
        if line_y is None:
            return
        for hand, keypoint_name in _HAND_KEYPOINT.items():
            if hand not in self._profile.striking_limbs:
                continue
            wrist = tracked.pose.get(keypoint_name)
            if wrist is None:
                continue
            self._update(tracked.fighter_id, hand, wrist.y <= line_y, tracked.timestamp_s)

    def _guard_line_y(self, tracked: TrackedPose) -> float | None:
        """Normalized y at/above which a hand counts as guarding, or None if unknown."""
        pose = tracked.pose
        nose = pose.get(KeypointName.NOSE)
        l_sh, r_sh = pose.get(KeypointName.LEFT_SHOULDER), pose.get(KeypointName.RIGHT_SHOULDER)
        if nose is None or l_sh is None or r_sh is None:
            return None
        shoulder_y = (l_sh.y + r_sh.y) / 2
        span = shoulder_y - nose.y
        if span <= 0:
            return None  # degenerate frame (extreme camera roll, bad detection)
        chin_y = nose.y + self._config.chin_ratio * span
        return chin_y + self._config.drop_margin_ratio * span

    def _update(
        self, fighter_id: FighterId, hand: Limb, up_now: bool, timestamp_s: float
    ) -> None:
        state = self._hands.setdefault((fighter_id, hand), _HandGuard())
        if up_now != state.candidate_up:
            state.candidate_up = up_now
            state.candidate_since_s = timestamp_s
            return
        stable_for = timestamp_s - state.candidate_since_s
        debounce = self._config.recover_debounce_s if up_now else self._config.drop_debounce_s
        if up_now == state.current_up or stable_for < debounce:
            return

        state.current_up = up_now
        self._bus.publish(
            GuardStateEvent(
                timestamp_s=timestamp_s, fighter_id=fighter_id, hand=hand, guard_up=up_now
            )
        )
