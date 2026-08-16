"""Built-in combo sequences for guided drill/practice mode.

Deliberately a small, fixed built-in list rather than YAML-configurable —
the bar for v1 is "solo training gets on-screen prompts to follow instead
of only passive logging," not a full drill-authoring system.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.events.types import StrikeType
from combat_vision.sports.base import SportProfile


@dataclass(frozen=True, slots=True)
class Drill:
    """A named sequence of strikes to throw in order."""

    name: str
    sequence: tuple[StrikeType, ...]


DEFAULT_DRILLS: tuple[Drill, ...] = (
    Drill("Jab-Cross", (StrikeType.JAB, StrikeType.CROSS)),
    Drill("Jab-Jab-Cross", (StrikeType.JAB, StrikeType.JAB, StrikeType.CROSS)),
    Drill("Jab-Cross-Hook", (StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK)),
    Drill("Cross-Hook-Uppercut", (StrikeType.CROSS, StrikeType.HOOK, StrikeType.UPPERCUT)),
    Drill(
        "Jab-Cross-Hook-Cross",
        (StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK, StrikeType.CROSS),
    ),
)


def drills_for_profile(profile: SportProfile) -> tuple[Drill, ...]:
    """Only drills whose every strike type the active sport profile allows."""
    return tuple(d for d in DEFAULT_DRILLS if all(profile.allows(t) for t in d.sequence))
