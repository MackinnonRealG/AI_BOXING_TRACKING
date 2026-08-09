"""Boxing sport profile."""

from __future__ import annotations

from combat_vision.events.types import Limb, StrikeType
from combat_vision.sports.base import BodyZone, SportProfile


class BoxingProfile(SportProfile):
    """Hands only; jab/cross/hook/uppercut; head and body targets."""

    name = "boxing"

    @property
    def strike_types(self) -> frozenset[StrikeType]:
        """Boxing's four punch classes."""
        return frozenset(
            {StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK, StrikeType.UPPERCUT}
        )

    @property
    def striking_limbs(self) -> frozenset[Limb]:
        """Hands only."""
        return frozenset({Limb.LEFT_HAND, Limb.RIGHT_HAND})

    @property
    def target_zones(self) -> tuple[BodyZone, ...]:
        """Head and torso above the belt."""
        return (
            BodyZone("head", "front/side of the head"),
            BodyZone("body", "torso above the belt line"),
        )
