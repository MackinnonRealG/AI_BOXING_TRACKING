"""Training calendar — every session, grouped by the day it happened.

Built the same way :mod:`analytics.trends` and :mod:`analytics.baseline`
are: reads storage only, recomputed on demand from persisted sessions and
routine hits, no camera attached. Answers the "did I actually train this
week" question a raw session list doesn't make easy to see at a glance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from combat_vision.storage.repository import SessionRepository


@dataclass(frozen=True, slots=True)
class DayEntry:
    """One calendar day's training summary."""

    day: date
    session_count: int
    total_duration_s: float
    sports: tuple[str, ...]
    top_routines: tuple[tuple[str, int], ...]
    """(routine name, total times thrown that day), most-thrown first."""


@dataclass(frozen=True, slots=True)
class CalendarReport:
    """One month's training calendar."""

    year: int
    month: int
    days: tuple[DayEntry, ...]
    """Only days with at least one session, oldest first."""
    session_count: int
    total_duration_s: float
    streak_days: int
    """Consecutive trained days ending on this month's most recent training
    day. Scoped to the queried month — a streak spanning a month boundary
    is undercounted, a known v1 simplification."""

    def to_text(self) -> str:
        lines = [f"Training calendar — {date(self.year, self.month, 1):%B %Y}", ""]
        if not self.days:
            lines.append("No sessions recorded this month.")
            return "\n".join(lines)
        for entry in self.days:
            minutes = entry.total_duration_s / 60.0
            sports = "/".join(entry.sports)
            plural = "s" if entry.session_count != 1 else ""
            header = (
                f"  {entry.day:%a %d}  — {entry.session_count} session{plural} "
                f"({sports}, {minutes:.0f} min)"
            )
            if entry.top_routines:
                routines = ", ".join(f"{name} ×{count}" for name, count in entry.top_routines)
                header += f"  {routines}"
            lines.append(header)
        lines.append("")
        total_minutes = self.total_duration_s / 60.0
        lines.append(
            f"Streak: {self.streak_days} day{'s' if self.streak_days != 1 else ''} | "
            f"This month: {self.session_count} sessions, {total_minutes:.0f} min"
        )
        return "\n".join(lines)


def build_calendar(repo: SessionRepository, year: int, month: int) -> CalendarReport:
    """Aggregate every session in ``year``/``month`` into a per-day report."""
    sessions = [
        s
        for s in repo.list_sessions()
        if s.started_at.year == year and s.started_at.month == month
    ]
    session_ids = [s.id for s in sessions]
    hits_by_session = repo.routine_hits_for_sessions(session_ids)

    by_day: dict[date, list] = defaultdict(list)
    for session in sessions:
        by_day[session.started_at.date()].append(session)

    days: list[DayEntry] = []
    for day in sorted(by_day):
        day_sessions = by_day[day]
        routine_totals: Counter[str] = Counter()
        for session in day_sessions:
            for name, _label, count in hits_by_session.get(session.id, []):
                routine_totals[name] += count
        days.append(
            DayEntry(
                day=day,
                session_count=len(day_sessions),
                total_duration_s=sum(s.duration_s or 0.0 for s in day_sessions),
                sports=tuple(sorted({s.sport for s in day_sessions})),
                top_routines=tuple(routine_totals.most_common(3)),
            )
        )

    return CalendarReport(
        year=year,
        month=month,
        days=tuple(days),
        session_count=len(sessions),
        total_duration_s=sum(s.duration_s or 0.0 for s in sessions),
        streak_days=_current_streak(sorted(by_day)),
    )


def _current_streak(trained_days: list[date]) -> int:
    """Consecutive-day streak ending on the last (most recent) trained day."""
    if not trained_days:
        return 0
    streak = 1
    pairs = zip(reversed(trained_days), reversed(trained_days[:-1]), strict=False)
    for later, earlier in pairs:
        if later - earlier == timedelta(days=1):
            streak += 1
        else:
            break
    return streak
