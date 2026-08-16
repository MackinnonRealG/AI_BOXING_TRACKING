"""Per-fighter progression across sessions — reads storage only."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from combat_vision.storage.models import EventRecord
from combat_vision.storage.repository import SessionRepository

# Event types counted as a technique fault, and how to tell a fault instance
# from a benign one within that type's stored payload. GuardStateEvent and
# KneeBendStateEvent are continuous state events — only the "bad" direction
# (guard down, knees locked) counts; RotationFaultEvent/LegDriveFaultEvent
# are only ever published for the fault case, so every stored one counts.
_FAULT_EVENT_TYPES: tuple[str, ...] = (
    "RotationFaultEvent",
    "LegDriveFaultEvent",
    "GuardStateEvent",
    "KneeBendStateEvent",
)


def _is_fault_instance(event_type: str, payload: dict) -> bool:
    """Whether one stored event of ``event_type`` represents a fault."""
    if event_type == "GuardStateEvent":
        return not payload.get("guard_up", True)
    if event_type == "KneeBendStateEvent":
        return bool(payload.get("locked", False))
    return True


@dataclass(frozen=True, slots=True)
class TrendPoint:
    """One session's aggregate for a fighter."""

    session_id: int
    started_at: str
    punch_candidates: int
    avg_peak_speed: float | None
    max_peak_speed: float | None
    punches_per_minute: float | None
    technique_faults: int
    faults_per_minute: float | None


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
    aggregate later never requires re-running video. Fault types are fetched
    one batched call each (rather than one call per session, per the same
    N+1 concern as the speed-event fetch above) — still a handful of queries
    total, not one per session.
    """
    sessions = repo.list_sessions()
    labels = repo.labels_for_fighter(fighter_id)
    relevant_ids = [s.id for s in sessions if s.id in labels]
    events_by_session = repo.events_for_sessions(relevant_ids, event_type="SpeedPeakEvent")
    fault_events_by_type = {
        event_type: repo.events_for_sessions(relevant_ids, event_type=event_type)
        for event_type in _FAULT_EVENT_TYPES
    }

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

        faults = _count_faults(fault_events_by_type, session.id, label)
        faults_per_minute = None
        if session.duration_s:
            faults_per_minute = faults / (session.duration_s / 60.0)

        points.append(
            TrendPoint(
                session_id=session.id,
                started_at=session.started_at.isoformat(),
                punch_candidates=count,
                avg_peak_speed=statistics.fmean(speeds) if speeds else None,
                max_peak_speed=max(speeds) if speeds else None,
                punches_per_minute=per_minute,
                technique_faults=faults,
                faults_per_minute=faults_per_minute,
            )
        )
    return FighterTrends(fighter_id=fighter_id, points=points)


def _count_faults(
    fault_events_by_type: dict[str, dict[int, list[EventRecord]]],
    session_id: int,
    label: str,
) -> int:
    """Total technique-fault instances for one fighter's label in one session."""
    total = 0
    for event_type, by_session in fault_events_by_type.items():
        for record in by_session.get(session_id, []):
            if record.fighter_label == label and _is_fault_instance(event_type, record.payload):
                total += 1
    return total
