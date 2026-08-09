"""Combination engine tests — pure event-stream input, no poses needed."""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.combination import CombinationEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    ComboEvent,
    Limb,
    SpeedUnit,
    StrikeEvent,
    StrikeType,
)
from combat_vision.sports import get_profile
from combat_vision.utils.config import CombinationConfig

_CALIBRATION = Calibration(metres_per_pixel=None, frame_width_px=1280, frame_height_px=720)


def _strike(t: float, strike_type: StrikeType, fighter: str = "A") -> StrikeEvent:
    return StrikeEvent(
        timestamp_s=t,
        fighter_id=fighter,
        strike_type=strike_type,
        limb=Limb.LEFT_HAND,
        speed=2000.0,
        unit=SpeedUnit.PIXELS_PER_SECOND,
    )


def _setup() -> tuple[EventBus, CombinationEngine, list[ComboEvent]]:
    bus = EventBus()
    combos: list[ComboEvent] = []
    bus.subscribe(ComboEvent, combos.append)
    engine = CombinationEngine(bus, get_profile("boxing"), _CALIBRATION, CombinationConfig())
    return bus, engine, combos


def test_close_strikes_chain_into_a_combo() -> None:
    """jab-cross-hook within the gap window -> one three-strike combo."""
    bus, engine, combos = _setup()
    bus.publish(_strike(0.0, StrikeType.JAB))
    bus.publish(_strike(0.3, StrikeType.CROSS))
    bus.publish(_strike(0.6, StrikeType.HOOK))
    bus.publish(_strike(2.5, StrikeType.JAB))  # gap > max_gap_s closes the chain
    engine.finish()

    assert len(combos) == 1
    combo = combos[0]
    assert combo.sequence == (StrikeType.JAB, StrikeType.CROSS, StrikeType.HOOK)
    assert combo.strike_timestamps == (0.0, 0.3, 0.6)


def test_isolated_strikes_are_not_combos() -> None:
    """Single strikes separated by long gaps never publish a ComboEvent."""
    bus, engine, combos = _setup()
    bus.publish(_strike(0.0, StrikeType.JAB))
    bus.publish(_strike(5.0, StrikeType.CROSS))
    bus.publish(_strike(10.0, StrikeType.HOOK))
    engine.finish()
    assert combos == []


def test_fighters_chain_independently() -> None:
    """Interleaved strikes from two fighters keep separate chains."""
    bus, engine, combos = _setup()
    bus.publish(_strike(0.0, StrikeType.JAB, "A"))
    bus.publish(_strike(0.1, StrikeType.CROSS, "B"))
    bus.publish(_strike(0.3, StrikeType.CROSS, "A"))
    bus.publish(_strike(0.4, StrikeType.HOOK, "B"))
    engine.finish()

    assert len(combos) == 2
    by_fighter = {c.fighter_id: c.sequence for c in combos}
    assert by_fighter["A"] == (StrikeType.JAB, StrikeType.CROSS)
    assert by_fighter["B"] == (StrikeType.CROSS, StrikeType.HOOK)
