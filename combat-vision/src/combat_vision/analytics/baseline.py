"""Per-fighter personal baselines — reads storage only.

Fixed global thresholds (``config/default.yaml``) tell you whether a punch
cleared some universal bar. A personal baseline instead compares a fighter
against *their own* history — "is my jab getting faster than it used to be"
— which is what self-improvement tracking over time actually needs. Built
the same way :mod:`analytics.trends` is: from persisted events, no camera
attached, so it works identically whether "now" is right after a live
session or a review run months later.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from combat_vision.storage.repository import SessionRepository

MIN_SAMPLES_PER_STRIKE_TYPE = 3
"""Below this many historical strikes of a type, there isn't a real baseline yet."""

MIN_BASELINE_SESSIONS = 2
"""Below this many prior sessions, comparisons would just be noise."""

_NOTABLE_GAP_PCT = 10.0
"""Session-average vs. typical must differ by at least this percent to mention."""


@dataclass(frozen=True, slots=True)
class StrikeTypeBaseline:
    """Personal speed baseline for one strike type, in one speed unit."""

    strike_type: str
    unit: str
    sample_count: int
    best_speed: float
    typical_speed: float
    """Median historical speed — a robust "typical good rep", less swayed by
    one outlier reading than the mean would be."""


@dataclass(frozen=True, slots=True)
class FighterBaseline:
    """A fighter's personal baseline across their prior session history."""

    fighter_id: int
    session_count: int
    by_strike_type: dict[tuple[str, str], StrikeTypeBaseline]
    """Keyed by (strike_type, unit) — speeds are never mixed across units."""


def compute_baseline(
    repo: SessionRepository, fighter_id: int, exclude_session_id: int | None = None
) -> FighterBaseline:
    """Build ``fighter_id``'s baseline from every session but ``exclude_session_id``.

    Excluding the session under comparison matters: without it, a fighter's
    first-ever session would trivially match "their own best" every time.
    """
    sessions = repo.list_sessions()
    labels = repo.labels_for_fighter(fighter_id)
    relevant_ids = [
        s.id for s in sessions if s.id in labels and s.id != exclude_session_id
    ]
    events_by_session = repo.events_for_sessions(relevant_ids, event_type="StrikeEvent")

    speeds_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for session_id in relevant_ids:
        label = labels[session_id]
        for record in events_by_session.get(session_id, []):
            if record.fighter_label != label:
                continue
            key = (record.payload["strike_type"], record.payload["unit"])
            speeds_by_key[key].append(record.payload["speed"])

    by_strike_type: dict[tuple[str, str], StrikeTypeBaseline] = {}
    for (strike_type, unit), speeds in speeds_by_key.items():
        if len(speeds) < MIN_SAMPLES_PER_STRIKE_TYPE:
            continue
        by_strike_type[(strike_type, unit)] = StrikeTypeBaseline(
            strike_type=strike_type,
            unit=unit,
            sample_count=len(speeds),
            best_speed=max(speeds),
            typical_speed=statistics.median(speeds),
        )

    return FighterBaseline(
        fighter_id=fighter_id,
        session_count=len(relevant_ids),
        by_strike_type=by_strike_type,
    )


def personal_best_notes(repo: SessionRepository, fighter_id: int, session_id: int) -> list[str]:
    """Plain-language "vs. your personal best" notes for one session.

    Returns an empty list rather than a note when there isn't enough prior
    history yet — a fighter's first couple of sessions shouldn't be told
    they're above or below a "baseline" that barely exists.
    """
    baseline = compute_baseline(repo, fighter_id, exclude_session_id=session_id)
    if baseline.session_count < MIN_BASELINE_SESSIONS:
        return []

    label = repo.label_for_fighter(session_id, fighter_id)
    if label is None:
        return []
    records = repo.events_for_session(session_id, event_type="StrikeEvent")
    speeds_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.fighter_label != label:
            continue
        key = (record.payload["strike_type"], record.payload["unit"])
        speeds_by_key[key].append(record.payload["speed"])

    notes: list[str] = []
    for (strike_type, unit), speeds in sorted(speeds_by_key.items()):
        base = baseline.by_strike_type.get((strike_type, unit))
        if base is None:
            continue
        session_best = max(speeds)
        session_avg = statistics.fmean(speeds)
        label_text = strike_type.upper()

        if session_best > base.best_speed:
            notes.append(
                f"New personal best {label_text} speed: {session_best:.1f} {unit} "
                f"(previous best {base.best_speed:.1f} {unit})."
            )

        if base.typical_speed > 0:
            gap_pct = (session_avg - base.typical_speed) / base.typical_speed * 100.0
            if abs(gap_pct) >= _NOTABLE_GAP_PCT:
                direction = "above" if gap_pct > 0 else "below"
                notes.append(
                    f"Average {label_text} speed this session ({session_avg:.1f} {unit}) "
                    f"is {abs(gap_pct):.0f}% {direction} your typical "
                    f"({base.typical_speed:.1f} {unit})."
                )
    return notes
