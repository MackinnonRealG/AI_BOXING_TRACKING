"""Trend computation tests — cross-session identity and per-session labels."""

from __future__ import annotations

from pathlib import Path

from combat_vision.analytics.trends import compute_trends
from combat_vision.events.types import Limb, SpeedPeakEvent, SpeedUnit
from combat_vision.storage.repository import SessionRepository


def _event(t: float, fighter_id: str) -> SpeedPeakEvent:
    return SpeedPeakEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        limb=Limb.LEFT_HAND,
        peak_speed=5.0,
        unit=SpeedUnit.METERS_PER_SECOND,
        start_s=t - 0.1,
        end_s=t,
    )


def test_trends_follow_the_fighter_across_relabelled_sessions(tmp_path: Path) -> None:
    """The same physical fighter labelled "B" in one session, "A" in another."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/trends.db")
    fighter_id = repo.get_or_create_fighter("Alex")

    session_1 = repo.create_session(sport="boxing", mode="review", source="s1.mp4", calibrated=True)
    repo.link_fighter(session_1, fighter_id, "A")
    repo.save_events(session_1, [_event(1.0, "A"), _event(2.0, "A")])
    repo.finish_session(session_1, 60.0)

    session_2 = repo.create_session(sport="boxing", mode="review", source="s2.mp4", calibrated=True)
    repo.link_fighter(session_2, fighter_id, "B")  # different label this time
    repo.save_events(session_2, [_event(1.0, "B"), _event(1.0, "A")])  # "A" here is someone else
    repo.finish_session(session_2, 60.0)

    trends = compute_trends(repo, fighter_id)

    assert trends.fighter_id == fighter_id
    assert [p.session_id for p in trends.points] == [session_1, session_2]
    assert [p.punch_candidates for p in trends.points] == [2, 1]


def test_trends_skip_sessions_the_fighter_never_joined(tmp_path: Path) -> None:
    repo = SessionRepository(f"sqlite:///{tmp_path}/trends_skip.db")
    fighter_id = repo.get_or_create_fighter("Alex")
    other_id = repo.get_or_create_fighter("Sam")

    joined = repo.create_session(sport="boxing", mode="review", source="j.mp4", calibrated=True)
    repo.link_fighter(joined, fighter_id, "A")
    repo.save_events(joined, [_event(1.0, "A")])
    repo.finish_session(joined, 60.0)

    unrelated = repo.create_session(sport="boxing", mode="review", source="u.mp4", calibrated=True)
    repo.link_fighter(unrelated, other_id, "A")
    repo.save_events(unrelated, [_event(1.0, "A")])
    repo.finish_session(unrelated, 60.0)

    trends = compute_trends(repo, fighter_id)

    assert [p.session_id for p in trends.points] == [joined]
