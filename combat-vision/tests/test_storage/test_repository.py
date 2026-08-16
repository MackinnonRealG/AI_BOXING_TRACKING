"""Repository batch-query tests — grouped results must match the per-item calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from combat_vision.events.types import Limb, SpeedPeakEvent, SpeedUnit
from combat_vision.storage.repository import SessionRepository


def _repo(tmp_path: Path, name: str) -> SessionRepository:
    return SessionRepository(f"sqlite:///{tmp_path}/{name}.db")


def test_labels_for_fighter_covers_only_sessions_the_fighter_joined(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "labels")
    fighter_id = repo.get_or_create_fighter("Alex")
    other_id = repo.get_or_create_fighter("Sam")

    session_a = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    session_b = repo.create_session(sport="boxing", mode="review", source="b.mp4", calibrated=True)
    session_c = repo.create_session(sport="boxing", mode="review", source="c.mp4", calibrated=True)

    repo.link_fighter(session_a, fighter_id, "A")
    repo.link_fighter(session_b, fighter_id, "B")  # same fighter, different label
    repo.link_fighter(session_c, other_id, "A")  # a different fighter entirely

    labels = repo.labels_for_fighter(fighter_id)

    assert labels == {session_a: "A", session_b: "B"}
    assert session_c not in labels


def test_link_fighter_rejects_a_second_label_for_the_same_pair(tmp_path: Path) -> None:
    """One fighter must hold at most one label per session.

    Without the unique constraint, label_for_fighter could return either row
    for a fighter linked twice in one session — an arbitrary, silent pick.
    """
    repo = _repo(tmp_path, "dup_pair")
    fighter_id = repo.get_or_create_fighter("Alex")
    session_id = repo.create_session(
        sport="boxing", mode="review", source="a.mp4", calibrated=True
    )

    repo.link_fighter(session_id, fighter_id, "A")
    with pytest.raises(IntegrityError):
        repo.link_fighter(session_id, fighter_id, "B")


def test_link_fighter_rejects_two_fighters_sharing_a_label(tmp_path: Path) -> None:
    """One label must map to at most one fighter per session.

    Without the unique constraint, two fighters could both hold label "A" in
    the same session and labels_for_fighter would silently overwrite one.
    """
    repo = _repo(tmp_path, "dup_label")
    fighter_a = repo.get_or_create_fighter("Alex")
    fighter_b = repo.get_or_create_fighter("Sam")
    session_id = repo.create_session(
        sport="boxing", mode="review", source="a.mp4", calibrated=True
    )

    repo.link_fighter(session_id, fighter_a, "A")
    with pytest.raises(IntegrityError):
        repo.link_fighter(session_id, fighter_b, "A")


def test_events_for_sessions_matches_per_session_calls(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "events")
    session_a = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    session_b = repo.create_session(sport="boxing", mode="review", source="b.mp4", calibrated=True)

    def _event(t: float) -> SpeedPeakEvent:
        return SpeedPeakEvent(
            timestamp_s=t,
            fighter_id="A",
            limb=Limb.LEFT_HAND,
            peak_speed=5.0,
            unit=SpeedUnit.METERS_PER_SECOND,
            start_s=t - 0.1,
            end_s=t,
        )

    repo.save_events(session_a, [_event(1.0), _event(2.0)])
    repo.save_events(session_b, [_event(3.0)])

    grouped = repo.events_for_sessions([session_a, session_b])

    assert [r.timestamp_s for r in grouped[session_a]] == [1.0, 2.0]
    assert [r.timestamp_s for r in grouped[session_b]] == [3.0]
    assert [r.id for r in grouped[session_a]] == [
        r.id for r in repo.events_for_session(session_a)
    ]

    # A session with no events still gets an (empty) entry, not a KeyError.
    session_c = repo.create_session(sport="boxing", mode="review", source="c.mp4", calibrated=True)
    grouped = repo.events_for_sessions([session_a, session_c])
    assert grouped[session_c] == []

    assert repo.events_for_sessions([]) == {}


def test_events_for_sessions_respects_event_type_filter(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "events_filtered")
    session_id = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    event = SpeedPeakEvent(
        timestamp_s=1.0,
        fighter_id="A",
        limb=Limb.LEFT_HAND,
        peak_speed=5.0,
        unit=SpeedUnit.METERS_PER_SECOND,
        start_s=0.9,
        end_s=1.0,
    )
    repo.save_events(session_id, [event])

    grouped = repo.events_for_sessions([session_id], event_type="SpeedPeakEvent")
    assert len(grouped[session_id]) == 1

    grouped = repo.events_for_sessions([session_id], event_type="StrikeEvent")
    assert grouped[session_id] == []


def test_rounds_for_sessions_matches_per_session_calls(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "rounds")
    session_a = repo.create_session(sport="boxing", mode="review", source="a.mp4", calibrated=True)
    session_b = repo.create_session(sport="boxing", mode="review", source="b.mp4", calibrated=True)

    repo.create_round(session_a, number=2, start_s=60.0, end_s=120.0)
    repo.create_round(session_a, number=1, start_s=0.0, end_s=60.0)
    repo.create_round(session_b, number=1, start_s=0.0, end_s=60.0)

    grouped = repo.rounds_for_sessions([session_a, session_b])

    assert [r.number for r in grouped[session_a]] == [1, 2]  # ordered by round number
    assert [r.number for r in grouped[session_b]] == [1]
    assert [r.id for r in grouped[session_a]] == [
        r.id for r in repo.rounds_for_session(session_a)
    ]

    assert repo.rounds_for_sessions([]) == {}
