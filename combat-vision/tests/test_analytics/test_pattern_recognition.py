"""Pattern recognition tests over a seeded storage layer — no camera."""

from __future__ import annotations

from pathlib import Path

from combat_vision.analytics.pattern_recognition import PatternRecognizer
from combat_vision.events.types import Limb, SpeedPeakEvent, SpeedUnit
from combat_vision.storage.repository import SessionRepository

_ROUND_S = 60.0


def _seed(repo: SessionRepository) -> int:
    """12 labelled rounds: fighter A wins exactly the high-output rounds.

    Returns the seeded fighter's stable DB id (not the per-session "A" label).
    """
    session_id = repo.create_session(
        sport="boxing", mode="review", source="seed.mp4", calibrated=True
    )
    fighter_id = repo.get_or_create_fighter("Alex")
    repo.link_fighter(session_id, fighter_id, "A")
    events = []
    for round_number in range(12):
        start = round_number * _ROUND_S
        high_output = round_number % 2 == 0
        repo.create_round(
            session_id,
            number=round_number + 1,
            start_s=start,
            end_s=start + _ROUND_S,
            outcome="A" if high_output else "B",
        )
        punches = 20 if high_output else 5
        for i in range(punches):
            t = start + 1.0 + i * (_ROUND_S - 2.0) / punches
            events.append(
                SpeedPeakEvent(
                    timestamp_s=t,
                    fighter_id="A",
                    limb=Limb.LEFT_HAND,
                    peak_speed=6.0,
                    unit=SpeedUnit.METERS_PER_SECOND,
                    start_s=t - 0.2,
                    end_s=t,
                )
            )
    repo.save_events(session_id, events)
    repo.finish_session(session_id, 12 * _ROUND_S)
    return fighter_id


def test_output_rate_pattern_is_discovered(tmp_path: Path) -> None:
    """High punch output perfectly separates wins -> a strong pattern."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/patterns.db")
    fighter_id = _seed(repo)

    patterns = PatternRecognizer(repo).discover(fighter_id, min_support=3)

    assert patterns, "expected at least one discovered pattern"
    top = patterns[0]
    assert "punch output" in top.description
    assert "Alex" in top.description
    assert top.confidence == 1.0  # 100% vs 0% split in the seeded data
    assert top.support == 12


def test_cross_session_label_reuse_does_not_blend_fighters(tmp_path: Path) -> None:
    """Two different physical fighters both labelled "A" must not mix data."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/patterns_cross.db")
    fighter_id = _seed(repo)

    # A second, unrelated fighter also happens to be labelled "A" in a
    # session with the *opposite* correlation. If the cross-session join
    # were bypassed, this would dilute/flip the first fighter's pattern.
    other_session_id = repo.create_session(
        sport="boxing", mode="review", source="other.mp4", calibrated=True
    )
    other_fighter_id = repo.get_or_create_fighter("Sam")
    repo.link_fighter(other_session_id, other_fighter_id, "A")
    other_events = []
    for round_number in range(12):
        start = round_number * _ROUND_S
        high_output = round_number % 2 == 0
        repo.create_round(
            other_session_id,
            number=round_number + 1,
            start_s=start,
            end_s=start + _ROUND_S,
            outcome="B" if high_output else "A",  # inverted correlation
        )
        punches = 20 if high_output else 5
        for i in range(punches):
            t = start + 1.0 + i * (_ROUND_S - 2.0) / punches
            other_events.append(
                SpeedPeakEvent(
                    timestamp_s=t,
                    fighter_id="A",
                    limb=Limb.LEFT_HAND,
                    peak_speed=6.0,
                    unit=SpeedUnit.METERS_PER_SECOND,
                    start_s=t - 0.2,
                    end_s=t,
                )
            )
    repo.save_events(other_session_id, other_events)
    repo.finish_session(other_session_id, 12 * _ROUND_S)

    patterns = PatternRecognizer(repo).discover(fighter_id, min_support=3)
    assert patterns, "expected the original fighter's pattern to survive unmixed"
    top = patterns[0]
    assert top.confidence == 1.0
    assert top.support == 12  # only the seeded fighter's 12 rounds, not 24


def test_unlabelled_storage_returns_no_patterns(tmp_path: Path) -> None:
    """Without user-labelled outcomes there is nothing to mine."""
    repo = SessionRepository(f"sqlite:///{tmp_path}/empty.db")
    session_id = repo.create_session(
        sport="boxing", mode="review", source="x.mp4", calibrated=False
    )
    fighter_id = repo.get_or_create_fighter("Nobody")
    repo.link_fighter(session_id, fighter_id, "A")
    repo.create_round(session_id, number=1, start_s=0.0, end_s=_ROUND_S)  # no outcome
    assert PatternRecognizer(repo).discover(fighter_id) == []
