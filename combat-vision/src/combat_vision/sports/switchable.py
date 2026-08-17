"""A SportProfile that can hot-swap the active sport at runtime.

Powers the live boxing/kickboxing toggle: the overlay's ``s`` key calls
:meth:`SwitchableSportProfile.switch`, and every engine already reads
``profile.striking_limbs`` / ``profile.allows(...)`` / ``profile.target_zones``
at use-time inside ``process()`` — none of them cache these at construction
— so swapping the delegate here makes every engine see the new sport's
rules starting on the very next frame. No pipeline rebuild needed, mirroring
how :class:`~combat_vision.tracking.switchable.SwitchableTracker` already
hot-swaps trackers.

Kickboxing is a strict superset of boxing's strike types in this codebase
(see ``sports/kickboxing.py``), so "switching sport" in practice means
"also start/stop monitoring kicks and knees" — there is no meaningful state
where both are simultaneously active or simultaneously inactive, which is
why this is one toggle between two modes rather than two independent
on/off switches.
"""

from __future__ import annotations

from combat_vision.events.types import Limb, StrikeType
from combat_vision.sports import get_profile
from combat_vision.sports.base import BodyZone, SportProfile


class SwitchableSportProfile(SportProfile):
    """Delegates every read to whichever concrete profile is currently active."""

    def __init__(self, initial: str) -> None:
        self._active = get_profile(initial)
        self.name = self._active.name

    def switch(self, name: str) -> str:
        """Swap the active sport; returns the new sport's name."""
        self._active = get_profile(name)
        self.name = self._active.name
        return self.name

    @property
    def strike_types(self) -> frozenset[StrikeType]:
        return self._active.strike_types

    @property
    def striking_limbs(self) -> frozenset[Limb]:
        return self._active.striking_limbs

    @property
    def target_zones(self) -> tuple[BodyZone, ...]:
        return self._active.target_zones
