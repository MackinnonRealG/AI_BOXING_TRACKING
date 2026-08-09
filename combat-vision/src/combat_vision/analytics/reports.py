"""Session report generation from recorded events.

Consumed by review mode (fresh events straight off the bus) and by the
analytics layer (events reloaded from storage) — same builder for both.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from combat_vision.events.types import (
    ComboEvent,
    DistanceSample,
    Event,
    PowerEstimateEvent,
    SpeedPeakEvent,
    StanceSwitchEvent,
    StepEvent,
    StrikeEvent,
)


@dataclass(frozen=True, slots=True)
class FighterSummary:
    """Per-fighter numbers for one session."""

    fighter_id: str
    punch_candidates: int
    strikes_by_type: dict[str, int]
    landed: int | None
    accuracy: float | None
    avg_peak_speed: float | None
    max_peak_speed: float | None
    speed_unit: str | None
    avg_power_score: float | None
    max_power_score: float | None
    steps: int
    stance_switches: int
    top_combinations: list[tuple[str, int]]


@dataclass(frozen=True, slots=True)
class SessionReport:
    """The structured output of review mode."""

    sport: str
    source: str
    duration_s: float
    fighters: list[FighterSummary]
    avg_fighter_distance: float | None = None
    distance_unit: str | None = None
    coaching_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable form."""
        return asdict(self)

    def to_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Combat Vision session report — {self.sport}",
            f"Source: {self.source}   Duration: {self.duration_s:.1f}s",
            "",
        ]
        for f in self.fighters:
            lines.append(f"Fighter {f.fighter_id}")
            lines.append(f"  Punch candidates: {f.punch_candidates}")
            if f.avg_peak_speed is not None:
                lines.append(
                    f"  Hand speed: avg {f.avg_peak_speed:.2f} / max "
                    f"{f.max_peak_speed:.2f} {f.speed_unit}"
                )
            if f.strikes_by_type:
                counts = ", ".join(f"{k}×{v}" for k, v in sorted(f.strikes_by_type.items()))
                lines.append(f"  Strikes: {counts}")
            if f.accuracy is not None:
                lines.append(f"  Accuracy: {f.accuracy:.0%}")
            if f.avg_power_score is not None:
                lines.append(
                    f"  Estimated power: avg {f.avg_power_score:.0f} / max "
                    f"{f.max_power_score:.0f} (score 0-100, estimate)"
                )
            lines.append(f"  Steps: {f.steps}   Stance switches: {f.stance_switches}")
            for sequence, count in f.top_combinations:
                lines.append(f"  Combo {sequence} ×{count}")
            lines.append("")
        if self.avg_fighter_distance is not None:
            lines.append(
                f"Average fighter distance: {self.avg_fighter_distance:.2f} {self.distance_unit}"
            )
            lines.append("")
        if self.coaching_notes:
            lines.append("Coaching notes:")
            lines.extend(f"  • {note}" for note in self.coaching_notes)
        return "\n".join(lines)


def build_session_report(
    sport: str, source: str, duration_s: float, events: Sequence[Event]
) -> SessionReport:
    """Aggregate a session's event stream into a report.

    Works with whatever engines were active: sections whose engines are still
    stubs simply come out zero/None rather than failing.
    """
    fighter_ids = sorted({e.fighter_id for e in events})
    summaries = [_summarize(fid, events) for fid in fighter_ids]
    distances = [e for e in events if isinstance(e, DistanceSample)]
    return SessionReport(
        sport=sport,
        source=source,
        duration_s=duration_s,
        fighters=summaries,
        avg_fighter_distance=(
            statistics.fmean(d.distance for d in distances) if distances else None
        ),
        distance_unit=distances[0].unit.value if distances else None,
        coaching_notes=_coaching_notes(summaries, duration_s),
    )


def _summarize(fighter_id: str, events: Sequence[Event]) -> FighterSummary:
    """Build one fighter's summary from the mixed event stream."""
    mine = [e for e in events if e.fighter_id == fighter_id]
    candidates = [e for e in mine if isinstance(e, SpeedPeakEvent)]
    strikes = [e for e in mine if isinstance(e, StrikeEvent)]
    steps = [e for e in mine if isinstance(e, StepEvent)]
    switches = [e for e in mine if isinstance(e, StanceSwitchEvent)]
    combos = [e for e in mine if isinstance(e, ComboEvent)]
    powers = [e.score for e in mine if isinstance(e, PowerEstimateEvent)]

    speeds = [c.peak_speed for c in candidates]
    landed_known = [s for s in strikes if s.landed is not None]
    landed: int | None = None
    accuracy: float | None = None
    if landed_known:
        landed = sum(1 for s in landed_known if s.landed)
        accuracy = landed / len(landed_known)

    combo_counter: Counter[str] = Counter(
        "–".join(t.value for t in combo.sequence) for combo in combos
    )
    return FighterSummary(
        fighter_id=fighter_id,
        punch_candidates=len(candidates),
        strikes_by_type=dict(Counter(s.strike_type.value for s in strikes)),
        landed=landed,
        accuracy=accuracy,
        avg_peak_speed=statistics.fmean(speeds) if speeds else None,
        max_peak_speed=max(speeds) if speeds else None,
        speed_unit=candidates[0].unit.value if candidates else None,
        avg_power_score=statistics.fmean(powers) if powers else None,
        max_power_score=max(powers) if powers else None,
        steps=len(steps),
        stance_switches=len(switches),
        top_combinations=combo_counter.most_common(3),
    )


def _coaching_notes(summaries: Sequence[FighterSummary], duration_s: float) -> list[str]:
    """Plain-language feedback derived purely from the numbers."""
    notes: list[str] = []
    minutes = max(duration_s / 60.0, 1e-6)
    for f in summaries:
        rate = f.punch_candidates / minutes
        if f.punch_candidates == 0:
            notes.append(f"Fighter {f.fighter_id}: no punch candidates detected — check "
                         "framing/calibration or activity level.")
            continue
        if rate < 10:
            notes.append(
                f"Fighter {f.fighter_id}: low output ({rate:.0f} punches/min) — "
                "consider higher work rate in sparring."
            )
        if f.avg_peak_speed and f.max_peak_speed and f.max_peak_speed > 1.5 * f.avg_peak_speed:
            notes.append(
                f"Fighter {f.fighter_id}: big gap between average and best hand speed — "
                "focus on consistent snap, not occasional bursts."
            )
        if f.accuracy is not None and f.accuracy < 0.3:
            notes.append(
                f"Fighter {f.fighter_id}: connect rate {f.accuracy:.0%} — work distance "
                "management and setups before volume."
            )
    return notes
