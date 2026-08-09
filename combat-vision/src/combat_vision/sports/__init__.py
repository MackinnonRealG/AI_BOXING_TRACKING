"""Sport profiles: sport as configuration, not branching."""

from combat_vision.sports.base import SportProfile
from combat_vision.sports.boxing import BoxingProfile
from combat_vision.sports.kickboxing import KickboxingProfile

_PROFILES: dict[str, type[SportProfile]] = {
    "boxing": BoxingProfile,
    "kickboxing": KickboxingProfile,
}


def get_profile(name: str) -> SportProfile:
    """Look up a sport profile by name (case-insensitive).

    Adding a new sport = adding one profile module and registering it here.
    """
    try:
        return _PROFILES[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unknown sport {name!r}; choose from {sorted(_PROFILES)}") from exc


__all__ = ["BoxingProfile", "KickboxingProfile", "SportProfile", "get_profile"]
