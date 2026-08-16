"""Personal-baseline tests: comparing a fighter against their own history."""

from __future__ import annotations

from pathlib import Path

from combat_vision.analytics.baseline import (
    MIN_BASELINE_SESSIONS,
    compute_baseline,
    personal_best_notes,
)
from combat_vision.events.types import Limb, SpeedUnit, StrikeEvent, StrikeType
from combat_vision.storage.repository import SessionRepository

_MPS = SpeedUnit.METERS_PER_SECOND


def _jab(t: float, speed: float, fighter_id: str = "A") -> StrikeEvent:
    return StrikeEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        strike_type=StrikeType.JAB,
        limb=Limb.LEFT_HAND,
        speed=speed,
        unit=_MPS,
    )


def _repo_with_history(tmp_path: Path, name: str) -> tuple[SessionRepository, int]:
    """A fighter with 3 prior sessions of jab speeds: [4, 5, 6], [5, 5, 5], [4, 6, 6]."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/{name}.db")
    fighter_id = repo.get_or_create_fighter("Alex")
    for speeds in ([4.0, 5.0, 6.0], [5.0, 5.0, 5.0], [4.0, 6.0, 6.0]):
        session = repo.create_session(
            sport="boxing", mode="review", source="s.mp4", calibrated=True
        )
        repo.link_fighter(session, fighter_id, "A")
        repo.save_events(session, [_jab(float(i), s) for i, s in enumerate(speeds)])
        repo.finish_session(session, 60.0)
    return repo, fighter_id


def test_baseline_needs_a_minimum_sample_count_per_strike_type(tmp_path: Path) -> None:
    """Two historical jabs (below MIN_SAMPLES_PER_STRIKE_TYPE) yields no baseline for jab."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/sparse.db")
    fighter_id = repo.get_or_create_fighter("Alex")
    session = repo.create_session(sport="boxing", mode="review", source="s.mp4", calibrated=True)
    repo.link_fighter(session, fighter_id, "A")
    repo.save_events(session, [_jab(1.0, 5.0), _jab(2.0, 5.0)])
    repo.finish_session(session, 60.0)

    baseline = compute_baseline(repo, fighter_id)
    assert baseline.by_strike_type == {}


def test_baseline_computes_best_and_typical_speed(tmp_path: Path) -> None:
    repo, fighter_id = _repo_with_history(tmp_path, "history")
    baseline = compute_baseline(repo, fighter_id)

    key = ("jab", "m/s")
    assert baseline.session_count == 3
    assert key in baseline.by_strike_type
    entry = baseline.by_strike_type[key]
    assert entry.sample_count == 9
    assert entry.best_speed == 6.0
    assert entry.typical_speed == 5.0  # median of [4,4,5,5,5,5,6,6,6]


def test_excluded_session_is_not_counted_in_its_own_baseline(tmp_path: Path) -> None:
    """A session must not be compared against itself."""
    repo, fighter_id = _repo_with_history(tmp_path, "exclude")
    all_sessions = [s.id for s in repo.list_sessions()]
    target = all_sessions[0]

    baseline = compute_baseline(repo, fighter_id, exclude_session_id=target)
    assert baseline.session_count == 2


def test_personal_best_notes_flags_a_new_best(tmp_path: Path) -> None:
    repo, fighter_id = _repo_with_history(tmp_path, "newbest")
    new_session = repo.create_session(
        sport="boxing", mode="review", source="new.mp4", calibrated=True
    )
    repo.link_fighter(new_session, fighter_id, "A")
    repo.save_events(new_session, [_jab(1.0, 7.0)])  # beats the historical best of 6.0
    repo.finish_session(new_session, 60.0)

    notes = personal_best_notes(repo, fighter_id, new_session)
    assert any("New personal best JAB" in n and "7.0" in n for n in notes)


def test_personal_best_notes_flags_a_below_typical_session(tmp_path: Path) -> None:
    repo, fighter_id = _repo_with_history(tmp_path, "belowtypical")
    new_session = repo.create_session(
        sport="boxing", mode="review", source="new.mp4", calibrated=True
    )
    repo.link_fighter(new_session, fighter_id, "A")
    # Well below the 5.0 m/s typical, none of which beats the 6.0 best.
    repo.save_events(new_session, [_jab(1.0, 3.0), _jab(2.0, 3.0)])
    repo.finish_session(new_session, 60.0)

    notes = personal_best_notes(repo, fighter_id, new_session)
    assert any("below your typical" in n for n in notes)
    assert not any("New personal best" in n for n in notes)


def test_no_notes_before_minimum_session_history(tmp_path: Path) -> None:
    """A fighter's very first session must not be judged against a near-empty baseline."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/first.db")
    fighter_id = repo.get_or_create_fighter("Alex")
    session = repo.create_session(sport="boxing", mode="review", source="s.mp4", calibrated=True)
    repo.link_fighter(session, fighter_id, "A")
    repo.save_events(session, [_jab(1.0, 5.0)])
    repo.finish_session(session, 60.0)

    assert MIN_BASELINE_SESSIONS >= 1  # sanity: the guard below is meaningful
    assert personal_best_notes(repo, fighter_id, session) == []


def test_units_are_never_mixed(tmp_path: Path) -> None:
    """A px/s session must not be compared against an m/s baseline, or vice versa."""
    repo, fighter_id = _repo_with_history(tmp_path, "units")  # all m/s
    pxps_session = repo.create_session(
        sport="boxing", mode="review", source="uncal.mp4", calibrated=False
    )
    repo.link_fighter(pxps_session, fighter_id, "A")
    repo.save_events(
        pxps_session,
        [
            StrikeEvent(
                timestamp_s=1.0,
                fighter_id="A",
                strike_type=StrikeType.JAB,
                limb=Limb.LEFT_HAND,
                speed=1400.0,
                unit=SpeedUnit.PIXELS_PER_SECOND,
            )
        ],
    )
    repo.finish_session(pxps_session, 60.0)

    notes = personal_best_notes(repo, fighter_id, pxps_session)
    assert notes == []  # no m/s baseline entry matches a px/s key, so nothing is compared
