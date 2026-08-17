"""SwitchableSportProfile tests: hot-swapping the active sport at runtime."""

from __future__ import annotations

from combat_vision.events.types import Limb, StrikeType
from combat_vision.sports import get_profile
from combat_vision.sports.switchable import SwitchableSportProfile


def test_starts_on_the_requested_sport() -> None:
    profile = SwitchableSportProfile("boxing")
    assert profile.name == "boxing"
    assert profile.strike_types == get_profile("boxing").strike_types
    assert profile.striking_limbs == get_profile("boxing").striking_limbs
    assert profile.target_zones == get_profile("boxing").target_zones


def test_switch_changes_every_delegated_property_immediately() -> None:
    profile = SwitchableSportProfile("boxing")
    assert not profile.allows(StrikeType.FRONT_KICK)
    assert Limb.LEFT_FOOT not in profile.striking_limbs

    returned = profile.switch("kickboxing")

    assert returned == "kickboxing"
    assert profile.name == "kickboxing"
    assert profile.allows(StrikeType.FRONT_KICK)
    assert Limb.LEFT_FOOT in profile.striking_limbs
    assert profile.target_zones == get_profile("kickboxing").target_zones


def test_switch_back_and_forth_is_idempotent() -> None:
    profile = SwitchableSportProfile("kickboxing")
    profile.switch("boxing")
    profile.switch("kickboxing")
    assert profile.name == "kickboxing"
    assert profile.strike_types == get_profile("kickboxing").strike_types


def test_unknown_sport_raises_and_leaves_the_active_profile_unchanged() -> None:
    profile = SwitchableSportProfile("boxing")
    try:
        profile.switch("mma")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown sport")
    assert profile.name == "boxing"  # the failed switch must not have partially applied
