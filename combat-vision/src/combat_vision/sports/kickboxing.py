"""Kickboxing sport profile."""

from __future__ import annotations

from combat_vision.events.types import Limb, StrikeType
from combat_vision.sports.base import BodyZone, SportProfile


class KickboxingProfile(SportProfile):
    """All boxing strikes plus kicks and knees; adds leg targets."""

    name = "kickboxing"

    @property
    def strike_types(self) -> frozenset[StrikeType]:
        """Boxing set plus front/roundhouse/side kicks and knees."""
        return frozenset(
            {
                StrikeType.JAB,
                StrikeType.CROSS,
                StrikeType.HOOK,
                StrikeType.UPPERCUT,
                StrikeType.FRONT_KICK,
                StrikeType.ROUNDHOUSE_LOW,
                StrikeType.ROUNDHOUSE_MID,
                StrikeType.ROUNDHOUSE_HIGH,
                StrikeType.SIDE_KICK,
                StrikeType.KNEE,
            }
        )

    @property
    def striking_limbs(self) -> frozenset[Limb]:
        """Hands, feet, and knees."""
        return frozenset(
            {
                Limb.LEFT_HAND,
                Limb.RIGHT_HAND,
                Limb.LEFT_FOOT,
                Limb.RIGHT_FOOT,
                Limb.LEFT_KNEE,
                Limb.RIGHT_KNEE,
            }
        )

    @property
    def target_zones(self) -> tuple[BodyZone, ...]:
        """Head, torso, and legs."""
        return (
            BodyZone("head", "front/side of the head"),
            BodyZone("body", "torso above the belt line"),
            BodyZone("legs", "thighs and calves (low-kick targets)"),
        )
