"""Routine-matching tests — combos thrown must map onto named, id'd routines."""

from __future__ import annotations

from pathlib import Path

from combat_vision.analytics.routine_matching import (
    record_routine_occurrences,
    sequence_key,
)
from combat_vision.events.types import ComboEvent, StrikeType
from combat_vision.storage.repository import SessionRepository


def _combo(fighter_id: str, *strikes: StrikeType, t: float = 1.0) -> ComboEvent:
    return ComboEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        sequence=strikes,
        strike_timestamps=tuple(t + i * 0.1 for i in range(len(strikes))),
    )


def test_sequence_key_is_dash_joined() -> None:
    assert sequence_key(("jab", "cross")) == "jab-cross"


def test_new_combo_is_discovered_and_given_a_stable_id(tmp_path: Path) -> None:
    repo = SessionRepository(f"sqlite:///{tmp_path}/discover.db")
    session_id = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    combos = [_combo("A", StrikeType.JAB, StrikeType.CROSS)]

    summaries = record_routine_occurrences(repo, session_id, "boxing", combos)

    assert len(summaries) == 1
    hit = summaries[0]
    assert hit.newly_discovered is True
    assert hit.count == 1
    assert hit.name == "Jab-Cross"

    routines = repo.list_routines("boxing")
    assert len(routines) == 1
    assert routines[0].id == hit.routine_id
    assert routines[0].source == "discovered"


def test_repeated_combo_in_one_session_is_tallied_not_duplicated(tmp_path: Path) -> None:
    repo = SessionRepository(f"sqlite:///{tmp_path}/tally.db")
    session_id = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    combos = [
        _combo("A", StrikeType.JAB, StrikeType.CROSS, t=1.0),
        _combo("A", StrikeType.JAB, StrikeType.CROSS, t=5.0),
        _combo("A", StrikeType.JAB, StrikeType.CROSS, t=9.0),
    ]

    summaries = record_routine_occurrences(repo, session_id, "boxing", combos)

    assert len(summaries) == 1
    assert summaries[0].count == 3
    assert len(repo.list_routines("boxing")) == 1


def test_seeded_reference_routine_is_matched_not_recreated(tmp_path: Path) -> None:
    """A combo matching a pre-seeded reference routine reuses its id/name."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/seeded.db")
    repo.ensure_reference_routines(
        [("The One-Two", "boxing", "jab-cross", ["jab", "cross"], "beginner")]
    )
    session_id = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)

    summaries = record_routine_occurrences(
        repo, session_id, "boxing", [_combo("A", StrikeType.JAB, StrikeType.CROSS)]
    )

    assert summaries[0].newly_discovered is False
    assert summaries[0].name == "The One-Two"
    assert len(repo.list_routines("boxing")) == 1  # matched, not a second row


def test_different_fighters_are_tallied_separately(tmp_path: Path) -> None:
    repo = SessionRepository(f"sqlite:///{tmp_path}/fighters.db")
    session_id = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    combos = [
        _combo("A", StrikeType.JAB, StrikeType.CROSS, t=1.0),
        _combo("B", StrikeType.JAB, StrikeType.CROSS, t=2.0),
    ]

    summaries = record_routine_occurrences(repo, session_id, "boxing", combos)

    assert {(s.fighter_label, s.count) for s in summaries} == {("A", 1), ("B", 1)}
    # Same routine (one row), two per-fighter hit rows.
    assert len(repo.list_routines("boxing")) == 1
