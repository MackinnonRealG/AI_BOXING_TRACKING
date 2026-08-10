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
    """Chronological trend series for one physical fighter."""

    fighter_id: int
    points: list[TrendPoint]


def compute_trends(repo: SessionRepository, fighter_id: int) -> FighterTrends:
    """Build the trend series for ``fighter_id`` across all sessions they appear in.

    Per-session labels ("A"/"B") are assigned independently each session, so
    each session's events are filtered by the label ``fighter_id`` actually
    held *in that session* (via ``SessionRepository.labels_for_fighter``)
    rather than by matching the raw label across sessions — otherwise two
    different people who both happened to be labelled "A" in different
    sessions would be blended into one trend series.

    Labels and events are fetched in two batched queries up front (not one
    query per session in the loop below) so this scales with session count
    without an N+1 round-trip per fighter lookup.

    Metrics are recomputed from persisted raw events, so improving an
    aggregate later never requires re-running video.
    """
    sessions = repo.list_sessions()
    labels = repo.labels_for_fighter(fighter_id)
    relevant_ids = [s.id for s in sessions if s.id in labels]
    events_by_session = repo.events_for_sessions(relevant_ids, event_type="SpeedPeakEvent")

    points: list[TrendPoint] = []
    for session in sessions:
        label = labels.get(session.id)
        if label is None:
            continue  # this fighter did not appear in this session
        records = events_by_session.get(session.id, [])
        speeds = [r.payload["peak_speed"] for r in records if r.fighter_label == label]
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
    return FighterTrends(fighter_id=fighter_id, points=points)
