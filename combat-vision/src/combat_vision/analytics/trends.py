"""Per-fighter progression across sessions — reads storage only."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from combat_vision.storage.repository import SessionRepository


@dataclass(frozen=True, slots=True)
class TrendPoint:
    """One session's aggregate for a fighter."""

    session_id: int
    started_at: str
    punch_candidates: int
    avg_peak_speed: float | None
    max_peak_speed: float | None
    punches_per_minute: float | None


@dataclass(frozen=True, slots=True)
class FighterTrends:
    """Chronological trend series for one fighter label."""

    fighter_label: str
    points: list[TrendPoint]


def compute_trends(repo: SessionRepository, fighter_label: str) -> FighterTrends:
    """Build the trend series for ``fighter_label`` across all sessions.

    Metrics are recomputed from persisted raw events, so improving an
    aggregate later never requires re-running video.
    """
    points: list[TrendPoint] = []
    for session in repo.list_sessions():
        records = repo.events_for_session(session.id, event_type="SpeedPeakEvent")
        speeds = [
            r.payload["peak_speed"] for r in records if r.fighter_label == fighter_label
        ]
        count = len(speeds)
        per_minute = None
        if session.duration_s:
            per_minute = count / (session.duration_s / 60.0)
        points.append(
            TrendPoint(
                session_id=session.id,
                started_at=session.started_at.isoformat(),
                punch_candidates=count,
                avg_peak_speed=statistics.fmean(speeds) if speeds else None,
                max_peak_speed=max(speeds) if speeds else None,
                punches_per_minute=per_minute,
            )
        )
    return FighterTrends(fighter_label=fighter_label, points=points)
