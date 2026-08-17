"""ui/overlay.py tests: per-label HUD state must not survive a relabel.

No window is opened — every behavior under test is driven through the event
bus, which is the overlay's only input besides ``render``.
"""

from __future__ import annotations

import time

from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    BalanceFaultEvent,
    ElbowStateEvent,
    FighterRelabeledEvent,
    GuardStateEvent,
    Keypoint,
    KeypointName,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    Limb,
    Pose,
    RotationFaultEvent,
    SpeedPeakEvent,
    SpeedUnit,
    StepEvent,
    StrikeEvent,
    StrikeType,
    TrackedPose,
)
from combat_vision.sports.switchable import SwitchableSportProfile
from combat_vision.ui.overlay import LiveOverlay
from combat_vision.utils.config import UiConfig

_MPS = SpeedUnit.METERS_PER_SECOND


def _overlay() -> tuple[LiveOverlay, EventBus]:
    bus = EventBus()
    return LiveOverlay(bus=bus, config=UiConfig()), bus


def _speed(fighter_id: str, t: float = 1.0) -> SpeedPeakEvent:
    return SpeedPeakEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        limb=Limb.LEFT_HAND,
        peak_speed=5.0,
        unit=_MPS,
        start_s=t - 0.2,
        end_s=t,
    )


def _strike(fighter_id: str, t: float = 1.0) -> StrikeEvent:
    return StrikeEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        strike_type=StrikeType.JAB,
        limb=Limb.LEFT_HAND,
        speed=5.0,
        unit=_MPS,
    )


def _step(fighter_id: str, t: float = 1.0) -> StepEvent:
    return StepEvent(
        timestamp_s=t,
        fighter_id=fighter_id,
        foot=Limb.LEFT_FOOT,
        from_xy=(0.0, 0.0),
        to_xy=(0.1, 0.0),
        displacement=0.1,
        unit=_MPS,
    )


def _visible_pose(fighter_id: str) -> TrackedPose:
    """A pose visible enough that _guidance never short-circuits on framing."""
    keypoints = {
        name: Keypoint(x=0.5, y=0.5, visibility=1.0)
        for name in (
            KeypointName.LEFT_WRIST,
            KeypointName.RIGHT_WRIST,
            KeypointName.LEFT_ANKLE,
            KeypointName.RIGHT_ANKLE,
        )
    }
    return TrackedPose(fighter_id=fighter_id, pose=Pose(keypoints=keypoints), timestamp_s=1.0)


def test_guard_drop_appears_in_guidance() -> None:
    """A flagged guard drop surfaces as an on-screen coaching cue."""
    overlay, bus = _overlay()
    bus.publish(
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False)
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "left hand" in message.lower()


def test_guard_up_produces_no_guidance() -> None:
    """A hand reported guard-up must not trigger the drop cue."""
    overlay, bus = _overlay()
    bus.publish(
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=True)
    )
    assert overlay._guidance([_visible_pose("A")]) is None


def test_elbow_flare_appears_in_guidance() -> None:
    """A flagged elbow flare surfaces as an on-screen coaching cue."""
    overlay, bus = _overlay()
    bus.publish(
        ElbowStateEvent(timestamp_s=1.0, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=False)
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "right hand" in message.lower()
    assert "elbow" in message.lower()


def test_elbow_tucked_produces_no_guidance() -> None:
    """An arm reported elbow-tucked must not trigger the flare cue."""
    overlay, bus = _overlay()
    bus.publish(
        ElbowStateEvent(timestamp_s=1.0, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=True)
    )
    assert overlay._guidance([_visible_pose("A")]) is None


def test_guard_drop_takes_priority_over_elbow_flare() -> None:
    """Guard drop is checked first — a dropped hand is the more urgent cue."""
    overlay, bus = _overlay()
    bus.publish(
        ElbowStateEvent(timestamp_s=1.0, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=False)
    )
    bus.publish(
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False)
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "dropping" in message.lower()


def test_rotation_fault_appears_in_guidance_and_expires() -> None:
    """A fresh rotation fault surfaces as a cue, then clears after it expires."""
    overlay, bus = _overlay()
    bus.publish(
        RotationFaultEvent(
            timestamp_s=1.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        )
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "hips" in message.lower()

    # Simulate expiry by back-dating when the fault was observed.
    text, _ = overlay._fighters["A"].fault_notes[-1]
    overlay._fighters["A"].fault_notes = [(text, time.monotonic() - 10.0)]
    assert overlay._guidance([_visible_pose("A")]) is None


def test_rotation_fault_takes_priority_over_guard_cue() -> None:
    """A fresh per-rep fault is more actionable right now than the standing guard cue."""
    overlay, bus = _overlay()
    bus.publish(
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False)
    )
    bus.publish(
        RotationFaultEvent(
            timestamp_s=1.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        )
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "hips" in message.lower()


def test_relabel_clears_guard_state() -> None:
    """Guard state is per-hand HUD state and must not survive a relabel either."""
    overlay, bus = _overlay()
    bus.publish(
        GuardStateEvent(timestamp_s=1.0, fighter_id="A", hand=Limb.LEFT_HAND, guard_up=False)
    )
    assert overlay._fighters["A"].guard_up[Limb.LEFT_HAND] is False

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].guard_up == {}


def test_relabel_clears_elbow_state() -> None:
    """Elbow-tuck state is per-arm HUD state and must not survive a relabel either."""
    overlay, bus = _overlay()
    bus.publish(
        ElbowStateEvent(timestamp_s=1.0, fighter_id="A", elbow=Limb.RIGHT_HAND, tucked=False)
    )
    assert overlay._fighters["A"].elbow_tucked[Limb.RIGHT_HAND] is False

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].elbow_tucked == {}


def test_relabel_clears_rotation_fault_state() -> None:
    """A rotation fault cue is per-fighter HUD state and must not survive a relabel."""
    overlay, bus = _overlay()
    bus.publish(
        RotationFaultEvent(
            timestamp_s=1.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        )
    )
    assert overlay._fighters["A"].fault_notes != []

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].fault_notes == []


def test_locked_knees_appear_in_guidance_below_guard_and_rotation() -> None:
    """A standing locked-knee warning is lower priority than a fresh per-rep fault."""
    overlay, bus = _overlay()
    bus.publish(KneeBendStateEvent(timestamp_s=1.0, fighter_id="A", locked=True))
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "knees" in message.lower()

    bus.publish(
        RotationFaultEvent(
            timestamp_s=1.0,
            fighter_id="A",
            limb=Limb.RIGHT_HAND,
            shoulder_rotation_deg=40.0,
            hip_rotation_deg=5.0,
        )
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "hips" in message.lower()


def test_leg_drive_fault_appears_in_guidance() -> None:
    """A no-leg-drive punch surfaces the same way a rotation fault does."""
    overlay, bus = _overlay()
    bus.publish(
        LegDriveFaultEvent(
            timestamp_s=1.0, fighter_id="A", limb=Limb.LEFT_HAND, knee_angle_deg=170.0
        )
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "leg drive" in message.lower()


def test_balance_fault_appears_in_guidance() -> None:
    """A wobbly-base kick surfaces the same way a leg-drive fault does."""
    overlay, bus = _overlay()
    bus.publish(
        BalanceFaultEvent(timestamp_s=1.0, fighter_id="A", limb=Limb.LEFT_FOOT, wobble_ratio=0.8)
    )
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert "wobbled" in message.lower()


def test_toggle_drill_starts_and_stops_via_the_d_key() -> None:
    """The 'd' key starts the next drill in rotation, then stops it on a second press."""
    bus = EventBus()
    overlay = LiveOverlay(
        bus=bus, config=UiConfig(), sport_profile=SwitchableSportProfile("boxing")
    )
    assert not overlay._drill_coach.active

    overlay._toggle_drill()
    assert overlay._drill_coach.active
    assert overlay._drill_coach.fighter_id == "A"

    overlay._toggle_drill()
    assert not overlay._drill_coach.active


def test_toggle_sport_flips_between_boxing_and_kickboxing() -> None:
    """The 's' key flips the shared profile so every engine sees the new
    sport immediately — this overlay just reads .name back afterward."""
    bus = EventBus()
    profile = SwitchableSportProfile("boxing")
    overlay = LiveOverlay(bus=bus, config=UiConfig(), sport_profile=profile)

    overlay._toggle_sport()
    assert profile.name == "kickboxing"

    overlay._toggle_sport()
    assert profile.name == "boxing"


def test_toggle_sport_refreshes_the_drill_list() -> None:
    """The drill list is recomputed for whichever sport is now active.

    (The built-in drills all happen to be hand-only, so boxing's and
    kickboxing's lists are currently identical in content -- this test
    checks the list is *recomputed against the new profile*, not that its
    contents differ, which drills_for_profile's own tests already cover.)
    """
    from combat_vision.drills import drills_for_profile

    bus = EventBus()
    profile = SwitchableSportProfile("boxing")
    overlay = LiveOverlay(bus=bus, config=UiConfig(), sport_profile=profile)

    overlay._toggle_sport()

    assert overlay._drills == drills_for_profile(profile)


def test_toggle_sport_stops_a_running_drill() -> None:
    """A drill built from the old sport's combo list is stopped, not left
    running against a profile that may no longer support its strikes."""
    bus = EventBus()
    profile = SwitchableSportProfile("boxing")
    overlay = LiveOverlay(bus=bus, config=UiConfig(), sport_profile=profile)
    overlay._toggle_drill()
    assert overlay._drill_coach.active

    overlay._toggle_sport()

    assert not overlay._drill_coach.active


def test_s_key_does_nothing_without_a_sport_profile() -> None:
    """Overlays built without a sport (e.g. most other tests in this file)
    must not crash on 's' -- there's nothing to toggle."""
    overlay, _bus = _overlay()
    assert overlay._handle_key(ord("s"), frame_width=100) is True


def test_toggle_drill_targets_the_lone_fighter_in_frame() -> None:
    """Solo training as fighter B must not silently start a drill for 'A'."""
    bus = EventBus()
    overlay = LiveOverlay(
        bus=bus, config=UiConfig(), sport_profile=SwitchableSportProfile("boxing")
    )
    overlay._now_s = 5.0
    overlay._fighters["B"].last_seen_s = 5.0  # only B is currently in frame

    overlay._toggle_drill()

    assert overlay._drill_coach.fighter_id == "B"


def test_toggle_drill_defaults_to_a_when_both_fighters_are_in_frame() -> None:
    bus = EventBus()
    overlay = LiveOverlay(
        bus=bus, config=UiConfig(), sport_profile=SwitchableSportProfile("boxing")
    )
    overlay._now_s = 5.0
    overlay._fighters["A"].last_seen_s = 5.0
    overlay._fighters["B"].last_seen_s = 5.0

    overlay._toggle_drill()

    assert overlay._drill_coach.fighter_id == "A"


def test_drill_prompt_appears_in_guidance() -> None:
    """A running drill's prompt takes priority in the guidance line."""
    bus = EventBus()
    overlay = LiveOverlay(
        bus=bus, config=UiConfig(), sport_profile=SwitchableSportProfile("boxing")
    )
    overlay._toggle_drill()
    message = overlay._guidance([_visible_pose("A")])
    assert message is not None
    assert message.startswith("DRILL")


def test_relabel_clears_knee_state() -> None:
    """Knee posture is per-fighter HUD state and must not survive a relabel."""
    overlay, bus = _overlay()
    bus.publish(KneeBendStateEvent(timestamp_s=1.0, fighter_id="A", locked=True))
    assert overlay._fighters["A"].knees_locked is True

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].knees_locked is None


def test_relabel_clears_that_fighters_counts() -> None:
    """A recycled label must not inherit the departed fighter's stats.

    Otherwise the HUD credits the previous person's punches, steps and strike
    histogram to whoever now holds the label — someone who has thrown nothing
    displays a full card.
    """
    overlay, bus = _overlay()
    bus.publish(_speed("A"))
    bus.publish(_speed("A"))
    bus.publish(_strike("A"))
    bus.publish(_step("A"))
    assert overlay._fighters["A"].candidates == 2
    assert overlay._fighters["A"].steps == 1

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].candidates == 0
    assert overlay._fighters["A"].steps == 0
    assert overlay._fighters["A"].last_strike is None
    assert overlay._fighters["A"].last_speed is None
    assert overlay._fighters["A"].strike_counts == {}


def test_relabel_leaves_other_fighters_alone() -> None:
    """Only the recycled label is cleared; the other fighter is untouched."""
    overlay, bus = _overlay()
    bus.publish(_speed("A"))
    bus.publish(_speed("B"))

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._fighters["A"].candidates == 0
    assert overlay._fighters["B"].candidates == 1


def test_relabel_drops_pending_toasts_for_that_fighter_only() -> None:
    """A departed fighter's strike must not pop over the new fighter's head."""
    overlay, bus = _overlay()
    bus.publish(_strike("A"))
    bus.publish(_strike("B"))
    assert len(overlay._toasts) == 2

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert [event.fighter_id for event, _ in overlay._toasts] == ["B"]


def test_relabel_clears_the_foot_heat_map() -> None:
    """Footwork accumulated by the previous person must not carry over."""
    overlay, bus = _overlay()
    overlay._foot_heat["A"][0, 0] = 42.0

    bus.publish(FighterRelabeledEvent(timestamp_s=2.0, fighter_id="A"))

    assert overlay._foot_heat["A"].max() == 0.0
