"""Training calendar tests — sessions grouped and summarized by day."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from combat_vision.analytics.calendar import build_calendar
from combat_vision.storage.repository import SessionRepository


def _repo(tmp_path: Path, name: str) -> SessionRepository:
    return SessionRepository(f"sqlite:///{tmp_path}/{name}.db")


def _stamp_session(db_path: Path, session_id: int, started_at: str, duration_s: float) -> None:
    """Backdate a session's started_at -- create_session always uses "now"."""
    con = sqlite3.connect(db_path)
    con.execute(
        "UPDATE sessions SET started_at = ?, duration_s = ? WHERE id = ?",
        (started_at, duration_s, session_id),
    )
    con.commit()
    con.close()


def test_empty_month_reports_no_sessions(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "empty")
    report = build_calendar(repo, 2026, 8)
    assert report.days == ()
    assert report.session_count == 0
    assert "No sessions recorded" in report.to_text()


def test_sessions_are_grouped_by_calendar_day(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "grouped")
    db_path = tmp_path / "grouped.db"

    s1 = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, s1, "2026-08-11 10:00:00", 600.0)
    s2 = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, s2, "2026-08-11 18:00:00", 300.0)
    s3 = repo.create_session(sport="kickboxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, s3, "2026-08-13 10:00:00", 900.0)

    report = build_calendar(repo, 2026, 8)

    assert report.session_count == 3
    assert len(report.days) == 2
    day11 = report.days[0]
    assert day11.session_count == 2
    assert day11.total_duration_s == 900.0
    assert day11.sports == ("boxing",)
    day13 = report.days[1]
    assert day13.sports == ("kickboxing",)


def test_sessions_outside_the_month_are_excluded(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "filtered")
    db_path = tmp_path / "filtered.db"
    s1 = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, s1, "2026-07-31 23:59:00", 60.0)
    s2 = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, s2, "2026-08-01 00:01:00", 60.0)

    report = build_calendar(repo, 2026, 8)

    assert report.session_count == 1


def test_top_routines_per_day_come_from_routine_hits(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "routines")
    db_path = tmp_path / "routines.db"
    session_id = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, session_id, "2026-08-11 10:00:00", 600.0)
    routine_id, _, _ = repo.find_or_create_routine(
        sport="boxing",
        sequence_key="jab-cross",
        sequence=["jab", "cross"],
        fallback_name="Jab-Cross",
    )
    repo.record_routine_hit(session_id, routine_id, "A", 14)

    report = build_calendar(repo, 2026, 8)

    assert report.days[0].top_routines == (("Jab-Cross", 14),)
    assert "Jab-Cross ×14" in report.to_text()


def test_streak_counts_consecutive_trained_days(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "streak")
    db_path = tmp_path / "streak.db"
    for day in ("2026-08-10", "2026-08-11", "2026-08-12"):
        sid = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
        _stamp_session(db_path, sid, f"{day} 10:00:00", 60.0)
    # A gap day, then one more trained day.
    sid = repo.create_session(sport="boxing", mode="live", source="cam0", calibrated=True)
    _stamp_session(db_path, sid, "2026-08-15 10:00:00", 60.0)

    report = build_calendar(repo, 2026, 8)

    assert report.streak_days == 1  # the 15th is isolated -- the 10-12 streak ended before it
