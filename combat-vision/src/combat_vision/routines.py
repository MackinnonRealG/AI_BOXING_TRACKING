"""The reference training-routine library — real, named boxing/kickboxing combos.

Distinct from :mod:`drills` (which drives the live guided-drill prompt/grade
loop): this is the seed data for the persisted "training routine library"
(``storage.models.TrainingRoutine``) — every routine here gets a stable
database id and name the first time the app runs, via
``SessionRepository.ensure_reference_routines``. Fighters also *discover*
their own routines automatically (see :mod:`analytics.routine_matching`):
the first time a fighter throws a sequence not already in this list, it gets
auto-named and added with ``source="discovered"``, so the library grows from
what you actually throw, not just what's seeded here.

Uses the standard boxing-gym numbering convention (1=jab, 2=cross, 3=lead
hook, 4=rear hook, 5=lead uppercut, 6=rear uppercut) — this app's
:class:`~combat_vision.events.types.StrikeType` has one ``HOOK`` and one
``UPPERCUT`` token each (no lead/rear distinction), so 3/4 both map to
``HOOK`` and 5/6 both map to ``UPPERCUT``.

Boxing combos and the numbering convention verified against
precisionstriking.com, blog.joinfightcamp.com, and legendsboxing.com.
Kickboxing combos 1-9 sourced from infighting.ca's published combination
list (its spinning-kick/crescent-kick/Superman-punch combos were excluded —
no matching ``StrikeType`` token exists for them); combos 10-12 are
standard, widely-taught combinations not tied to one specific source.
``SIDE_KICK`` and ``KNEE`` coverage is intentionally thin (1-2 combos each)
— neither source emphasized them.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.events.types import StrikeType

_J = StrikeType.JAB
_C = StrikeType.CROSS
_H = StrikeType.HOOK
_U = StrikeType.UPPERCUT
_FK = StrikeType.FRONT_KICK
_RL = StrikeType.ROUNDHOUSE_LOW
_RM = StrikeType.ROUNDHOUSE_MID
_RH = StrikeType.ROUNDHOUSE_HIGH
_SK = StrikeType.SIDE_KICK
_KN = StrikeType.KNEE


@dataclass(frozen=True, slots=True)
class RoutineSeed:
    """One reference routine: name, sport, sequence, and rough difficulty."""

    name: str
    sport: str  # "boxing" | "kickboxing"
    sequence: tuple[StrikeType, ...]
    difficulty: str  # "beginner" | "intermediate" | "advanced"


REFERENCE_ROUTINES: tuple[RoutineSeed, ...] = (
    # -- Boxing (gym numbering shown in the name) --
    RoutineSeed("The One-Two (1-2)", "boxing", (_J, _C), "beginner"),
    RoutineSeed("Jab-Cross-Hook (1-2-3)", "boxing", (_J, _C, _H), "beginner"),
    RoutineSeed("Cross-Hook-Cross (2-3-2)", "boxing", (_C, _H, _C), "beginner"),
    RoutineSeed("Jab-Uppercut-Hook (1-6-3)", "boxing", (_J, _U, _H), "intermediate"),
    RoutineSeed("Hook-Uppercut-Cross (3-6-2)", "boxing", (_H, _U, _C), "intermediate"),
    RoutineSeed(
        "Jab-Cross-Uppercut-Cross (1-2-5-2)", "boxing", (_J, _C, _U, _C), "intermediate"
    ),
    RoutineSeed("Four-Punch Combo (1-2-3-4)", "boxing", (_J, _C, _H, _H), "intermediate"),
    RoutineSeed("Uppercut-Cross-Hook (5-2-3)", "boxing", (_U, _C, _H), "intermediate"),
    RoutineSeed(
        "Hook-Uppercut-Cross-Hook (3-6-2-3)", "boxing", (_H, _U, _C, _H), "advanced"
    ),
    RoutineSeed(
        "Jab-Cross-Double Uppercut (1-2-5-6)", "boxing", (_J, _C, _U, _U), "advanced"
    ),
    RoutineSeed(
        "Cross-Hook-Cross-Uppercut (2-3-2-5)", "boxing", (_C, _H, _C, _U), "advanced"
    ),
    # -- Kickboxing: hands + kicks/knees mixed --
    RoutineSeed(
        "Kick Entry (aka Body Kick Setup)", "kickboxing", (_FK, _J, _C, _RM), "beginner"
    ),
    RoutineSeed("Foot Jab Finish", "kickboxing", (_FK, _J, _C, _FK), "beginner"),
    RoutineSeed("Kick-Punch-Kick", "kickboxing", (_RM, _J, _C, _RM), "intermediate"),
    RoutineSeed(
        "High/Low/High Body Rip (aka Body Hook to Kick)",
        "kickboxing",
        (_J, _C, _H, _RM),
        "advanced",
    ),
    RoutineSeed("Low-High-Low", "kickboxing", (_J, _J, _C, _FK), "intermediate"),
    RoutineSeed("Jab-Cross-Low Kick", "kickboxing", (_J, _C, _RL), "beginner"),
    RoutineSeed("Jab-Cross-Hook-Low Kick", "kickboxing", (_J, _C, _H, _RL), "intermediate"),
    RoutineSeed("Teep-Cross-Roundhouse", "kickboxing", (_FK, _C, _RH), "intermediate"),
    RoutineSeed("Jab-Cross-Knee", "kickboxing", (_J, _C, _KN), "beginner"),
    RoutineSeed("Cross-Hook-Knee", "kickboxing", (_C, _H, _KN), "intermediate"),
    RoutineSeed("Side Kick Stop", "kickboxing", (_J, _SK), "beginner"),
    RoutineSeed("Low-High Kick Combo", "kickboxing", (_RL, _RH), "advanced"),
)


def seed_rows() -> list[tuple[str, str, str, list[str], str | None]]:
    """``REFERENCE_ROUTINES`` in the row shape ``ensure_reference_routines`` expects."""
    return [
        (
            seed.name,
            seed.sport,
            "-".join(t.value for t in seed.sequence),
            [t.value for t in seed.sequence],
            seed.difficulty,
        )
        for seed in REFERENCE_ROUTINES
    ]
