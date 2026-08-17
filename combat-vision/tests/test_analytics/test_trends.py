"""Trend computation tests — cross-session identity and per-session labels."""

from __future__ import annotations

from pathlib import Path

from combat_vision.analytics.trends import compute_trends
from combat_vision.events.types import (
    BalanceFaultEvent,
    ElbowStateEvent,
    GuardStateEvent,
    KneeBendStateEvent,
    Limb,
    RotationFaultEvent,
    SpeedPeakEvent,
    SpeedUnit,
)
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


def test_technique_faults_count_only_the_bad_direction_of_state_events(
    tmp_path: Path,
) -> None:
    """Guard-up and knees-bent are not faults; only the drop/lock direction counts."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/faults.db")
    fighter_id = repo.get_or_create_fighter("Alex")

    session = repo.create_session(sport="boxing", mode="review", source="s.mp4", calibrated=True)
    repo.link_fighter(session, fighter_id, "A")
    repo.save_events(
        session,
        [
            GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False),
            GuardStateEvent(timestamp_s=2.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=True),
            ElbowStateEvent(timestamp_s=2.5, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=False),
            ElbowStateEvent(timestamp_s=2.6, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=True),
            KneeBendStateEvent(timestamp_s=3.0, fighter_id="A", locked=True),
            KneeBendStateEvent(timestamp_s=4.0, fighter_id="A", locked=False),
            RotationFaultEvent(
                timestamp_s=5.0,
                fighter_id="A",
                limb=Limb.RIGHT_HAND,
                shoulder_rotation_deg=40.0,
                hip_rotation_deg=5.0,
            ),
            BalanceFaultEvent(
                timestamp_s=5.5, fighter_id="A", limb=Limb.LEFT_FOOT, wobble_ratio=0.8
            ),
            # A different fighter's fault in the same session must not be counted.
            GuardStateEvent(timestamp_s=6.0, fighter_id="B", hand=Limb.LEFT_HAND, guard_up=False),
        ],
    )
    repo.finish_session(session, 60.0)

    trends = compute_trends(repo, fighter_id)

    assert len(trends.points) == 1
    point = trends.points[0]
    # 1 guard drop + 1 elbow flare + 1 locked-knee event + 1 rotation fault
    # + 1 balance fault = 5; the guard-up, elbow-tucked, knees-bent, and
    # other-fighter's-drop events are correctly excluded.
    assert point.technique_faults == 5
    assert point.faults_per_minute == 5.0
