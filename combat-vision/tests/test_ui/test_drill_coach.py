"""DrillCoach state machine tests — no camera or overlay window required."""

from __future__ import annotations

from combat_vision.drills import Drill, drills_for_profile
from combat_vision.events.types import Limb, SpeedUnit, StrikeEvent, StrikeType
from combat_vision.sports import get_profile
from combat_vision.ui.drill_coach import DrillCoach

_MPS = SpeedUnit.METERS_PER_SECOND
_JAB_CROSS = Drill("Jab-Cross", (StrikeType.JAB, StrikeType.CROSS))


def _strike(strike_type: StrikeType, t: float, fighter_id: str = "A") -> StrikeEvent:
    return StrikeEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        strike_type=strike_type,
        limb=Limb.LEFT_HAND,  # unused by DrillCoach, but a required field
        speed=5.0,
        unit=_MPS,
    )


def test_starts_idle_and_start_begins_a_countdown() -> None:
    coach = DrillCoach(countdown_s=2.0)
    assert not coach.active
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    assert coach.active
    assert "ready in" in (coach.prompt(0.0) or "")


def test_countdown_transitions_to_active_after_the_configured_duration() -> None:
    coach = DrillCoach(countdown_s=2.0)
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.tick(1.0)
    assert "ready in" in (coach.prompt(1.0) or "")

    coach.tick(2.0)
    prompt = coach.prompt(2.0)
    assert prompt is not None
    assert "throw" in prompt
    assert "[JAB]" in prompt


def test_correct_sequence_finishes_clean() -> None:
    coach = DrillCoach(countdown_s=0.0, result_hold_s=5.0)
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.tick(0.0)  # countdown_s=0 -> immediately active

    coach.on_strike(_strike(StrikeType.JAB, 1.0))
    prompt = coach.prompt(1.0)
    assert prompt is not None
    assert "JAB-[CROSS]" in prompt

    coach.on_strike(_strike(StrikeType.CROSS, 2.0))
    assert "CLEAN" in (coach.prompt(2.0) or "")


def test_wrong_strike_breaks_the_sequence() -> None:
    coach = DrillCoach(countdown_s=0.0, result_hold_s=5.0)
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.tick(0.0)

    coach.on_strike(_strike(StrikeType.HOOK, 1.0))
    prompt = coach.prompt(1.0)
    assert prompt is not None
    assert "broke the sequence" in prompt


def test_result_returns_to_idle_after_result_hold_s() -> None:
    coach = DrillCoach(countdown_s=0.0, result_hold_s=1.0)
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.tick(0.0)
    coach.on_strike(_strike(StrikeType.HOOK, 0.5))  # breaks -> result state
    assert coach.active

    coach.tick(1.5)  # past result_hold_s since the break at t=0.5
    assert not coach.active
    assert coach.prompt(1.5) is None


def test_strikes_from_a_different_fighter_are_ignored() -> None:
    coach = DrillCoach(countdown_s=0.0)
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.tick(0.0)

    coach.on_strike(_strike(StrikeType.JAB, 1.0, fighter_id="B"))
    prompt = coach.prompt(1.0)
    assert prompt is not None
    assert "[JAB]" in prompt  # still waiting on the first strike, unaffected


def test_stop_clears_the_drill() -> None:
    coach = DrillCoach()
    coach.start(_JAB_CROSS, fighter_id="A", now_s=0.0)
    coach.stop()
    assert not coach.active
    assert coach.prompt(0.0) is None
    assert coach.fighter_id is None


def test_drills_for_profile_excludes_kick_only_combos_in_boxing() -> None:
    boxing_drills = drills_for_profile(get_profile("boxing"))
    assert all(
        strike_type not in (StrikeType.FRONT_KICK, StrikeType.KNEE)
        for drill in boxing_drills
        for strike_type in drill.sequence
    )
    assert len(boxing_drills) > 0


def test_drills_for_profile_allows_all_built_in_drills_in_kickboxing() -> None:
    from combat_vision.drills import DEFAULT_DRILLS

    kickboxing_drills = drills_for_profile(get_profile("kickboxing"))
    assert len(kickboxing_drills) == len(DEFAULT_DRILLS)
