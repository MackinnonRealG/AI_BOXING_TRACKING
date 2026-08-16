"""Session report tests: metric aggregation and plain-language coaching notes."""

from __future__ import annotations

from combat_vision.analytics.reports import build_session_report
from combat_vision.events.types import (
    CleanTechniqueEvent,
    DepthPostureSample,
    GuardStateEvent,
    HeadPostureSample,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    Limb,
    RotationFaultEvent,
    SpeedPeakEvent,
    SpeedUnit,
    StrikeEvent,
    StrikeType,
)

_MPS = SpeedUnit.METERS_PER_SECOND


def _candidate(t: float) -> SpeedPeakEvent:
    return SpeedPeakEvent(
        timestamp_s=t,
        fighter_id="A",
        limb=Limb.LEFT_HAND,
        peak_speed=5.0,
        unit=_MPS,
        start_s=t - 0.1,
        end_s=t,
    )


def test_fault_counts_exclude_the_benign_direction() -> None:
    """Guard-up and knees-bent samples must not inflate the fault counts."""
    events = [
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False),
        GuardStateEvent(timestamp_s=2.0, fighter_id="A", hand=Limb.RIGHT_HAND, guard_up=True),
        KneeBendStateEvent(timestamp_s=3.0, fighter_id="A", locked=True),
        KneeBendStateEvent(timestamp_s=4.0, fighter_id="A", locked=False),
        RotationFaultEvent(
            timestamp_s=5.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        ),
        LegDriveFaultEvent(
            timestamp_s=6.0, fighter_id="A", limb=Limb.LEFT_HAND, knee_angle_deg=178.0
        ),
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)

    summary = report.fighters[0]
    assert summary.guard_drops == 1
    assert summary.locked_knee_events == 1
    assert summary.rotation_faults == 1
    assert summary.leg_drive_faults == 1


def test_clean_technique_events_are_counted_alongside_faults() -> None:
    """Good reps are logged and reported, not just the mistakes."""
    events = [
        CleanTechniqueEvent(
            timestamp_s=1.0, fighter_id="A", check="hip_rotation", limb=Limb.RIGHT_HAND
        ),
        CleanTechniqueEvent(
            timestamp_s=2.0, fighter_id="A", check="hip_rotation", limb=Limb.RIGHT_HAND
        ),
        RotationFaultEvent(
            timestamp_s=3.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        ),
        CleanTechniqueEvent(
            timestamp_s=1.0, fighter_id="A", check="leg_drive", limb=Limb.LEFT_HAND
        ),
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    summary = report.fighters[0]
    assert summary.clean_hip_turns == 2
    assert summary.clean_leg_drives == 1

    text = report.to_text()
    assert "Hip turn: 2 clean / 1 no-turn (67% clean)" in text
    assert "Leg drive: 1 clean / 0 no-drive (100% clean)" in text


def test_approximate_measurements_average_correctly_and_ignore_missing_fields() -> None:
    events = [
        HeadPostureSample(timestamp_s=1.0, fighter_id="A", tilt_deg=10.0),
        HeadPostureSample(timestamp_s=2.0, fighter_id="A", tilt_deg=20.0),
        DepthPostureSample(
            timestamp_s=1.0,
            fighter_id="A",
            left_elbow_flare=0.1,
            right_elbow_flare=None,
            torso_lean=0.2,
        ),
        DepthPostureSample(
            timestamp_s=2.0,
            fighter_id="A",
            left_elbow_flare=None,
            right_elbow_flare=None,
            torso_lean=None,  # must not be averaged in as 0
        ),
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)

    summary = report.fighters[0]
    assert summary.avg_head_tilt_deg == 15.0
    assert summary.avg_torso_lean == 0.2


def test_no_faults_omits_the_technique_fault_line() -> None:
    events = [_candidate(1.0)]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    text = report.to_text()
    assert "Technique faults" not in text


def test_technique_fault_line_appears_when_faults_exist() -> None:
    events = [
        _candidate(1.0),
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False),
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    text = report.to_text()
    assert "Technique faults" in text
    assert "guard dropped ×1" in text


def test_coaching_note_fires_when_rotation_fault_rate_is_high() -> None:
    """4 of 4 punches with no hip turn (>25%) should trigger the plain-language note."""
    events = [_candidate(float(i)) for i in range(1, 5)]
    events += [
        RotationFaultEvent(
            timestamp_s=float(i),
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        )
        for i in range(1, 5)
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    assert any("no hip turn" in note for note in report.coaching_notes)


def test_coaching_note_fires_when_guard_drops_frequently() -> None:
    """More than 3 guard drops per minute should trigger the plain-language note."""
    events = [_candidate(1.0)]
    events += [
        GuardStateEvent(timestamp_s=float(i), fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False)
        for i in range(4)
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    assert any("guard dropped" in note for note in report.coaching_notes)


def test_landed_strike_reporting_is_unaffected_by_new_fields() -> None:
    """Regression guard: extending FighterSummary must not break existing aggregation."""
    events = [
        _candidate(1.0),
        StrikeEvent(
            timestamp_s=1.0,
            fighter_id="A",
            strike_type=StrikeType.JAB,
            limb=Limb.LEFT_HAND,
            speed=5.0,
            unit=_MPS,
            landed=True,
        ),
    ]
    report = build_session_report("boxing", "cam0", duration_s=60.0, events=events)
    summary = report.fighters[0]
    assert summary.landed == 1
    assert summary.accuracy == 1.0
