"""Applies One-Euro filtering to every keypoint of every tracked fighter."""

from __future__ import annotations

from collections import defaultdict

from combat_vision.events.types import FighterId, Keypoint, KeypointName, Pose, TrackedPose
from combat_vision.filtering.one_euro import OneEuroFilter


class PoseSmoother:
    """Stateful per-(fighter, keypoint, axis) smoothing of the pose stream.

    Sits between the tracker and the metrics engines so every engine sees the
    same de-jittered signal.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._params = (min_cutoff, beta, d_cutoff)
        # fighter_id -> keypoint -> (x filter, y filter)
        self._filters: dict[str, dict[KeypointName, tuple[OneEuroFilter, OneEuroFilter]]] = (
            defaultdict(dict)
        )

    def smooth(self, tracked: TrackedPose) -> TrackedPose:
        """Return a copy of ``tracked`` with filtered keypoint coordinates."""
        fighter_filters = self._filters[tracked.fighter_id]
        smoothed: dict[KeypointName, Keypoint] = {}
        for name, kp in tracked.pose.keypoints.items():
            if name not in fighter_filters:
                fighter_filters[name] = (OneEuroFilter(*self._params), OneEuroFilter(*self._params))
            fx, fy = fighter_filters[name]
            smoothed[name] = Keypoint(
                x=fx.filter(kp.x, tracked.timestamp_s),
                y=fy.filter(kp.y, tracked.timestamp_s),
                z=kp.z,
                visibility=kp.visibility,
            )
        return TrackedPose(
            fighter_id=tracked.fighter_id,
            pose=Pose(keypoints=smoothed),
            timestamp_s=tracked.timestamp_s,
            camera_id=tracked.camera_id,
        )

    def reset(self, fighter_id: FighterId) -> None:
        """Drop all filter state for one fighter label.

        Call this when a tracker reassigns the label to a different physical
        person (see :meth:`~combat_vision.tracking.base.Tracker.consume_relabeled`)
        — otherwise the new person's first frames get smoothed against the
        departed fighter's stale position/velocity, producing a spurious
        speed spike that corrupts downstream speed/strike metrics.
        """
        self._filters.pop(fighter_id, None)
