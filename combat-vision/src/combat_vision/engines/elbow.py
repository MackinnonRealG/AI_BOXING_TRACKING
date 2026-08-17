"""Elbow engine: elbows-flared-off-the-ribs fault detection.

Boxing fundamentals keep the elbows tucked close to the torso between and
during exchanges — flared elbows expose the ribs/body and telegraph shots.
This engine estimates each arm's elbow "tucked" state every frame from a
purely 2D signal, since there is no reliable depth channel to build this on
(see :mod:`engines.depth_posture` for why MediaPipe's ``z`` isn't trusted
for a verdict):

1. **Torso centerline** — the mean x of both shoulders.
2. **Flare threshold** — an elbow whose horizontal distance from that
   centerline exceeds ``flare_ratio`` of the half shoulder width counts as
   flared (roughly: pushed noticeably wider than the shoulder line).
3. **Debounce** — mirrors :mod:`engines.guard`: sustained flare for
   ``flare_debounce_s`` before flagging, sustained tuck for
   ``tuck_debounce_s`` before clearing.

Deliberately does not exempt a currently-punching arm, for the same reason
guard.py doesn't: an elbow that stays flared through a whole combination is
exactly the "not resetting your guard" fault a coach would flag. The real
limitation this creates: a hook's own mechanics *require* the elbow to
travel laterally, so a combo built from several wide hooks in a row can
legitimately read as sustained flare. That's a known false-positive mode of
this signal, not a bug — the same tradeoff guard.py already accepts for a
hand held wide mid-hook.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    ElbowStateEvent,
    FighterId,
    FighterRelabeledEvent,
    KeypointName,
    Limb,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import ElbowConfig

_ELBOW_KEYPOINT: dict[Limb, KeypointName] = {
    Limb.LEFT_HAND: KeypointName.LEFT_ELBOW,
    Limb.RIGHT_HAND: KeypointName.RIGHT_ELBOW,
}


@dataclass
class _ElbowGuard:
    """Debounce state for one fighter's one elbow."""

    current_tucked: bool | None = None
    candidate_tucked: bool | None = None
    candidate_since_s: float = 0.0


class ElbowEngine(MetricsEngine):
    """Classifies each arm's elbow as tucked/flared, with debounce."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: ElbowConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._elbows: dict[tuple[FighterId, Limb], _ElbowGuard] = {}
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop debounce state for a label now held by a different person.

        Same reasoning as :meth:`engines.guard.GuardEngine._on_relabeled`: a
        relabeled fighter starting in the same tucked/flared state the
        departed one left behind would never get an initial
        :class:`ElbowStateEvent`.
        """
        for arm in _ELBOW_KEYPOINT:
            self._elbows.pop((event.fighter_id, arm), None)

    def process(self, tracked: TrackedPose) -> None:
        """Classify this frame's elbow-tuck state and run the debounce machine."""
        reference = self._centerline_and_half_width(tracked)
        if reference is None:
            return
        centerline_x, half_width = reference
        if half_width <= 0:
            return  # degenerate frame (shoulders collapsed to one point)

        for arm, keypoint_name in _ELBOW_KEYPOINT.items():
            if arm not in self._profile.striking_limbs:
                continue
            elbow = tracked.pose.get(keypoint_name)
            if elbow is None:
                continue
            offset_ratio = abs(elbow.x - centerline_x) / half_width
            tucked = offset_ratio <= self._config.flare_ratio
            self._update(tracked.fighter_id, arm, tucked, tracked.timestamp_s)

    def _centerline_and_half_width(self, tracked: TrackedPose) -> tuple[float, float] | None:
        """Torso centerline x and half shoulder width, or None if unknown."""
        pose = tracked.pose
        l_sh, r_sh = pose.get(KeypointName.LEFT_SHOULDER), pose.get(KeypointName.RIGHT_SHOULDER)
        if l_sh is None or r_sh is None:
            return None
        return (l_sh.x + r_sh.x) / 2, abs(l_sh.x - r_sh.x) / 2

    def _update(
        self, fighter_id: FighterId, arm: Limb, tucked_now: bool, timestamp_s: float
    ) -> None:
        state = self._elbows.setdefault((fighter_id, arm), _ElbowGuard())
        if tucked_now != state.candidate_tucked:
            state.candidate_tucked = tucked_now
            state.candidate_since_s = timestamp_s
            return
        stable_for = timestamp_s - state.candidate_since_s
        debounce = self._config.tuck_debounce_s if tucked_now else self._config.flare_debounce_s
        if tucked_now == state.current_tucked or stable_for < debounce:
            return

        state.current_tucked = tucked_now
        self._bus.publish(
            ElbowStateEvent(
                timestamp_s=timestamp_s, fighter_id=fighter_id, elbow=arm, tucked=tucked_now
            )
        )
