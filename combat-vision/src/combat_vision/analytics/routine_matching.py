"""Matches thrown combos against the training-routine library and logs hits.

Every :class:`~combat_vision.events.types.ComboEvent` already carries the
exact strike sequence a fighter threw (see :mod:`engines.combination`); this
module is the bridge from "a sequence was thrown" to "routine #14
'Jab-Cross-LowKick' was thrown, and now has a stable id you can look up
again" — matching against the seeded reference library
(:mod:`combat_vision.routines`) first, and auto-naming/creating a new
"discovered" routine the first time a fighter throws something not already
in the library. Runs once per session, right after events are saved, same
timing as :func:`analytics.baseline.personal_best_notes`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from combat_vision.events.types import ComboEvent
from combat_vision.storage.repository import SessionRepository


def sequence_key(sequence: tuple[str, ...]) -> str:
    """The human-readable dedup/lookup key for a strike sequence."""
    return "-".join(sequence)


def _fallback_name(sequence: tuple[str, ...]) -> str:
    """Auto-generated name for a newly-discovered routine, e.g. 'Jab-Cross-Hook'."""
    return "-".join(part.replace("_", " ").title() for part in sequence)


@dataclass(frozen=True, slots=True)
class RoutineHitSummary:
    """One (fighter, routine) pairing logged for a session."""

    routine_id: int
    name: str
    fighter_label: str
    count: int
    newly_discovered: bool


def record_routine_occurrences(
    repo: SessionRepository, session_id: int, sport: str, combos: list[ComboEvent]
) -> list[RoutineHitSummary]:
    """Match every combo thrown in a session against the routine library.

    Combos are grouped by ``(fighter_id, sequence)`` first so a routine
    thrown five times in one session is one lookup/insert plus a count of
    5, not five round-trips.
    """
    tallies: Counter[tuple[str, tuple[str, ...]]] = Counter()
    for combo in combos:
        tallies[(combo.fighter_id, tuple(t.value for t in combo.sequence))] += 1

    summaries: list[RoutineHitSummary] = []
    for (fighter_label, sequence), count in tallies.items():
        key = sequence_key(sequence)
        routine_id, name, created = repo.find_or_create_routine(
            sport=sport,
            sequence_key=key,
            sequence=list(sequence),
            fallback_name=_fallback_name(sequence),
        )
        repo.record_routine_hit(session_id, routine_id, fighter_label, count)
        summaries.append(
            RoutineHitSummary(
                routine_id=routine_id,
                name=name,
                fighter_label=fighter_label,
                count=count,
                newly_discovered=created,
            )
        )
    return summaries
