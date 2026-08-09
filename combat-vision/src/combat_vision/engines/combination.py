"""Combination engine: assembles strikes into combos.

Unlike the pose-driven engines, this one consumes the *strike event stream*:
it subscribes to StrikeEvent on the bus and needs no pose input at all.

Per fighter, consecutive strikes separated by at most ``max_gap_s`` chain
into an open combo; a longer gap (or end of stream) closes it. Chains of at
least ``min_length`` strikes publish a :class:`ComboEvent` with the ordered
strike-type sequence, from which reports count most-used combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import ComboEvent, FighterId, StrikeEvent, TrackedPose
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import CombinationConfig


@dataclass
class _Chain:
    """A combo currently being assembled for one fighter."""

    strikes: list[StrikeEvent] = field(default_factory=list)


class CombinationEngine(MetricsEngine):
    """Chains strikes into combinations and publishes them as ComboEvents."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: CombinationConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._chains: dict[FighterId, _Chain] = {}
        bus.subscribe(StrikeEvent, self._on_strike)

    def process(self, tracked: TrackedPose) -> None:
        """No pose input needed — this engine is strike-event driven."""

    def finish(self) -> None:
        """Close every open chain at end of stream."""
        for fighter_id in list(self._chains):
            self._close(fighter_id)

    def _on_strike(self, event: StrikeEvent) -> None:
        """Append to the fighter's open chain, closing it first on a long gap."""
        chain = self._chains.setdefault(event.fighter_id, _Chain())
        if chain.strikes and (
            event.timestamp_s - chain.strikes[-1].timestamp_s > self._config.max_gap_s
        ):
            self._close(event.fighter_id)
            chain = self._chains.setdefault(event.fighter_id, _Chain())
        chain.strikes.append(event)

    def _close(self, fighter_id: FighterId) -> None:
        """Publish the chain as a ComboEvent if it is long enough."""
        chain = self._chains.pop(fighter_id, None)
        if chain is None or len(chain.strikes) < self._config.min_length:
            return
        self._bus.publish(
            ComboEvent(
                timestamp_s=chain.strikes[-1].timestamp_s,
                fighter_id=fighter_id,
                sequence=tuple(s.strike_type for s in chain.strikes),
                strike_timestamps=tuple(s.timestamp_s for s in chain.strikes),
            )
        )
