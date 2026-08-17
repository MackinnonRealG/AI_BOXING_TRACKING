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
    BalanceFaultEvent,
    CleanTechniqueEvent,
    ComboEvent,
    DepthPostureSample,
    DistanceSample,
    ElbowStateEvent,
    Event,
    GuardStateEvent,
    HeadPostureSample,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    PowerEstimateEvent,
    RotationFaultEvent,
    SpeedPeakEvent,
    StanceSwitchEvent,
    StepEvent,
    StrikeEvent,
)


@dataclass(frozen=True, slots=True)
class FighterSummary:
    """Per-fighter numbers for one session."""

    fighter_id: str
    name: str | None
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
    guard_drops: int
    elbow_flares: int
    rotation_faults: int
    leg_drive_faults: int
    locked_knee_events: int
    balance_faults: int
    clean_hip_turns: int
    clean_leg_drives: int
    clean_base_balance: int
    avg_head_tilt_deg: float | None
    avg_head_lateral_movement: float | None
    avg_torso_lean: float | None


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
            lines.append(f.name or f"Fighter {f.fighter_id}")
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
            if (
                f.guard_drops
                or f.elbow_flares
                or f.rotation_faults
                or f.leg_drive_faults
                or f.locked_knee_events
                or f.balance_faults
            ):
                lines.append(
                    f"  Technique faults: guard dropped ×{f.guard_drops}, "
                    f"elbow flared ×{f.elbow_flares}, no hip turn ×{f.rotation_faults}, "
                    f"no leg drive ×{f.leg_drive_faults}, "
                    f"locked-knee stretches ×{f.locked_knee_events}, "
                    f"base wobbled ×{f.balance_faults}"
                )
            hip_judged = f.clean_hip_turns + f.rotation_faults
            if hip_judged:
                lines.append(
                    f"  Hip turn: {f.clean_hip_turns} clean / {f.rotation_faults} no-turn "
                    f"({f.clean_hip_turns / hip_judged:.0%} clean)"
                )
            leg_judged = f.clean_leg_drives + f.leg_drive_faults
            if leg_judged:
                lines.append(
                    f"  Leg drive: {f.clean_leg_drives} clean / {f.leg_drive_faults} no-drive "
                    f"({f.clean_leg_drives / leg_judged:.0%} clean)"
                )
            balance_judged = f.clean_base_balance + f.balance_faults
            if balance_judged:
                lines.append(
                    f"  Kick/knee base balance: {f.clean_base_balance} clean / "
                    f"{f.balance_faults} wobbled "
                    f"({f.clean_base_balance / balance_judged:.0%} clean)"
                )
            if f.avg_head_tilt_deg is not None:
                lines.append(f"  Avg head tilt: {f.avg_head_tilt_deg:.0f}° (approximate)")
            if f.avg_head_lateral_movement is not None:
                lines.append(
                    f"  Avg head movement: {f.avg_head_lateral_movement:.2f}x shoulder width "
                    "(higher = more head movement; not judged as good or bad)"
                )
            if f.avg_torso_lean is not None:
                direction = "forward" if f.avg_torso_lean > 0 else "back"
                lines.append(
                    f"  Avg torso lean: {direction} "
                    "(approximate, unitless single-camera depth signal)"
                )
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
    sport: str,
    source: str,
    duration_s: float,
    events: Sequence[Event],
    names: dict[str, str] | None = None,
) -> SessionReport:
    """Aggregate a session's event stream into a report.

    ``names`` maps fighter labels ("A"/"B") to display names. Sections whose
    engines produced no events simply come out zero/None rather than failing.
    """
    names = names or {}
    fighter_ids = sorted({e.fighter_id for e in events})
    summaries = [_summarize(fid, events, names.get(fid)) for fid in fighter_ids]
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


def _summarize(
    fighter_id: str, events: Sequence[Event], name: str | None = None
) -> FighterSummary:
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

    guard_drops = sum(1 for e in mine if isinstance(e, GuardStateEvent) and not e.guard_up)
    elbow_flares = sum(1 for e in mine if isinstance(e, ElbowStateEvent) and not e.tucked)
    rotation_faults = sum(1 for e in mine if isinstance(e, RotationFaultEvent))
    leg_drive_faults = sum(1 for e in mine if isinstance(e, LegDriveFaultEvent))
    locked_knee_events = sum(1 for e in mine if isinstance(e, KneeBendStateEvent) and e.locked)
    balance_faults = sum(1 for e in mine if isinstance(e, BalanceFaultEvent))
    clean = [e for e in mine if isinstance(e, CleanTechniqueEvent)]
    clean_hip_turns = sum(1 for e in clean if e.check == "hip_rotation")
    clean_leg_drives = sum(1 for e in clean if e.check == "leg_drive")
    clean_base_balance = sum(1 for e in clean if e.check == "base_balance")
    head_tilts = [e.tilt_deg for e in mine if isinstance(e, HeadPostureSample)]
    head_movements = [
        e.lateral_movement
        for e in mine
        if isinstance(e, HeadPostureSample) and e.lateral_movement is not None
    ]
    torso_leans = [
        e.torso_lean
        for e in mine
        if isinstance(e, DepthPostureSample) and e.torso_lean is not None
    ]

    return FighterSummary(
        fighter_id=fighter_id,
        name=name,
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
        guard_drops=guard_drops,
        elbow_flares=elbow_flares,
        rotation_faults=rotation_faults,
        leg_drive_faults=leg_drive_faults,
        locked_knee_events=locked_knee_events,
        balance_faults=balance_faults,
        clean_hip_turns=clean_hip_turns,
        clean_leg_drives=clean_leg_drives,
        clean_base_balance=clean_base_balance,
        avg_head_tilt_deg=statistics.fmean(head_tilts) if head_tilts else None,
        avg_head_lateral_movement=statistics.fmean(head_movements) if head_movements else None,
        avg_torso_lean=statistics.fmean(torso_leans) if torso_leans else None,
    )


def _coaching_notes(summaries: Sequence[FighterSummary], duration_s: float) -> list[str]:
    """Plain-language feedback derived purely from the numbers."""
    notes: list[str] = []
    minutes = max(duration_s / 60.0, 1e-6)
    for f in summaries:
        who = f.name or f"Fighter {f.fighter_id}"
        rate = f.punch_candidates / minutes
        if f.punch_candidates == 0:
            notes.append(f"{who}: no punch candidates detected — check "
                         "framing/calibration or activity level.")
            continue
        if rate < 10:
            notes.append(
                f"{who}: low output ({rate:.0f} punches/min) — "
                "consider higher work rate in sparring."
            )
        if f.avg_peak_speed and f.max_peak_speed and f.max_peak_speed > 1.5 * f.avg_peak_speed:
            notes.append(
                f"{who}: big gap between average and best hand speed — "
                "focus on consistent snap, not occasional bursts."
            )
        if f.accuracy is not None and f.accuracy < 0.3:
            notes.append(
                f"{who}: connect rate {f.accuracy:.0%} — work distance "
                "management and setups before volume."
            )
        if f.guard_drops / minutes > 3:
            notes.append(
                f"{who}: guard dropped {f.guard_drops} times this session — "
                "make resetting your hands after every exchange automatic."
            )
        if f.elbow_flares / minutes > 3:
            notes.append(
                f"{who}: elbows flared out {f.elbow_flares} times this session — "
                "keep them tucked to your ribs to protect the body."
            )
        hip_judged = f.clean_hip_turns + f.rotation_faults
        if hip_judged and f.rotation_faults / hip_judged > 0.25:
            notes.append(
                f"{who}: {f.rotation_faults} of {hip_judged} judged punches had no hip "
                "turn — you're throwing with your arm, not your hips."
            )
        leg_judged = f.clean_leg_drives + f.leg_drive_faults
        if leg_judged and f.leg_drive_faults / leg_judged > 0.25:
            notes.append(
                f"{who}: {f.leg_drive_faults} of {leg_judged} judged punches were thrown "
                "with locked knees — stay bent so you can push off the floor."
            )
        balance_judged = f.clean_base_balance + f.balance_faults
        if balance_judged and f.balance_faults / balance_judged > 0.25:
            notes.append(
                f"{who}: {f.balance_faults} of {balance_judged} judged kicks/knees had a "
                "wobbly base leg — work stance drills to stay grounded through the kick."
            )
    return notes
