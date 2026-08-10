"""PoseSmoother tests — filter state persistence and reset."""

from __future__ import annotations

from combat_vision.events.types import Keypoint, KeypointName, Pose, TrackedPose
from combat_vision.filtering.smoother import PoseSmoother


def _pose(x: float, y: float, t: float, fighter_id: str = "A") -> TrackedPose:
    return TrackedPose(
        fighter_id=fighter_id,
        pose=Pose(keypoints={KeypointName.NOSE: Keypoint(x=x, y=y)}),
        timestamp_s=t,
    )


def test_smoothing_lags_behind_a_sudden_jump() -> None:
    """A big frame-to-frame jump must be damped, not passed straight through."""
    smoother = PoseSmoother(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    smoother.smooth(_pose(0.5, 0.5, 0.0))
    smoothed = smoother.smooth(_pose(0.9, 0.9, 0.1))
    assert smoothed.pose.get(KeypointName.NOSE).x != 0.9


def test_reset_drops_filter_state_for_that_fighter_only() -> None:
    """After reset, a fighter's next sample passes through unfiltered.

    This is the behavior a tracker relies on when it recycles a label onto
    a different physical person: without a reset, the new person's first
    frames would be smoothed against the departed fighter's stale
    position/velocity.
    """
    smoother = PoseSmoother(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    smoother.smooth(_pose(0.5, 0.5, 0.0, fighter_id="A"))
    smoother.smooth(_pose(0.9, 0.9, 0.1, fighter_id="A"))
    smoother.smooth(_pose(0.2, 0.2, 0.0, fighter_id="B"))

    smoother.reset("A")

    # A fresh sample for the reset fighter passes through unfiltered — a
    # brand-new OneEuroFilter has no prior value to blend against.
    reset_result = smoother.smooth(_pose(0.1, 0.1, 0.2, fighter_id="A"))
    assert reset_result.pose.get(KeypointName.NOSE).x == 0.1
    assert reset_result.pose.get(KeypointName.NOSE).y == 0.1

    # The untouched fighter's state must survive the other fighter's reset.
    still_smoothed = smoother.smooth(_pose(0.9, 0.9, 0.1, fighter_id="B"))
    assert still_smoothed.pose.get(KeypointName.NOSE).x != 0.9


def test_reset_of_unknown_fighter_is_a_no_op() -> None:
    """Resetting a label with no filter state yet must not raise."""
    smoother = PoseSmoother(min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
    smoother.reset("Z")  # no prior state for "Z" — must be a harmless no-op
