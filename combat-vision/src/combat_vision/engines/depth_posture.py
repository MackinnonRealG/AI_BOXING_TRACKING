"""Depth-posture engine: approximate elbow-flare and torso-lean measurement.

MediaPipe's pose landmarks include a ``z`` coordinate (roughly the same
scale as ``x``, smaller = closer to the camera) that every other engine in
this codebase ignores — coordinates flow through the pipeline as normalized
image-plane ``(x, y)`` everywhere else. This engine is the first consumer of
that depth channel, for two things a single 2D camera plane genuinely can't
see on its own:

* **elbow flare** — an elbow pushed forward, out of the torso's frontal
  plane, reads as *closer to the camera* than the shoulder/hip reference
  even when its (x, y) position looks unremarkable from a front-on or
  3/4 camera angle.
* **torso lean** — shoulders sitting closer to (or farther from) the camera
  than the hips, i.e. leaning into or away from an exchange along the
  camera's depth axis rather than sideways in the image.

Both are published as raw, unitless measurements (:class:`DepthPostureSample`)
rather than faults, for the same reason :mod:`engines.head_posture` doesn't
grade what it measures: MediaPipe's ``z`` is a rough, un-calibrated,
single-view estimate — noisier and less trustworthy than the (x, y) geometry
every other engine relies on — and its meaning depends on which way the
fighter is oriented to the camera. Treat these numbers as directional hints
for a coach watching the feed, not verified biomechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    DepthPostureSample,
    FighterId,
    KeypointName,
    Pose,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils.config import DepthPostureConfig

_TORSO_KEYPOINTS = (
    KeypointName.LEFT_SHOULDER,
    KeypointName.RIGHT_SHOULDER,
    KeypointName.LEFT_HIP,
    KeypointName.RIGHT_HIP,
)


@dataclass
class _FighterDepthPosture:
    """Per-fighter sampling state."""

    last_sample_s: float = -1.0e9


class DepthPostureEngine(MetricsEngine):
    """Publishes a decimated, approximate depth-based posture reading."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: DepthPostureConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._fighters: dict[FighterId, _FighterDepthPosture] = {}

    def process(self, tracked: TrackedPose) -> None:
        """Sample elbow flare and torso lean at the configured cadence."""
        state = self._fighters.setdefault(tracked.fighter_id, _FighterDepthPosture())
        if tracked.timestamp_s - state.last_sample_s < self._config.sample_interval_s:
            return

        pose = tracked.pose
        torso_z = self._torso_reference_z(pose)
        left_flare = self._elbow_flare(pose, KeypointName.LEFT_ELBOW, torso_z)
        right_flare = self._elbow_flare(pose, KeypointName.RIGHT_ELBOW, torso_z)
        torso_lean = self._torso_lean(pose)
        if left_flare is None and right_flare is None and torso_lean is None:
            return  # nothing measurable this frame; don't publish an empty sample

        state.last_sample_s = tracked.timestamp_s
        self._bus.publish(
            DepthPostureSample(
                timestamp_s=tracked.timestamp_s,
                fighter_id=tracked.fighter_id,
                left_elbow_flare=left_flare,
                right_elbow_flare=right_flare,
                torso_lean=torso_lean,
            )
        )

    def _torso_reference_z(self, pose: Pose) -> float | None:
        """Mean z of shoulders/hips — the torso's depth reference plane."""
        zs = [
            kp.z
            for name in _TORSO_KEYPOINTS
            if (kp := pose.get(name)) is not None and kp.z is not None
        ]
        return sum(zs) / len(zs) if zs else None

    def _elbow_flare(
        self, pose: Pose, elbow_name: KeypointName, torso_z: float | None
    ) -> float | None:
        """Positive when this elbow is closer to the camera than the torso."""
        if torso_z is None:
            return None
        kp = pose.get(elbow_name)
        if kp is None or kp.z is None:
            return None
        return torso_z - kp.z

    def _torso_lean(self, pose: Pose) -> float | None:
        """Positive when the shoulders are closer to the camera than the hips."""
        l_sh, r_sh = pose.get(KeypointName.LEFT_SHOULDER), pose.get(KeypointName.RIGHT_SHOULDER)
        l_hip, r_hip = pose.get(KeypointName.LEFT_HIP), pose.get(KeypointName.RIGHT_HIP)
        if l_sh is None or r_sh is None or l_hip is None or r_hip is None:
            return None
        if l_sh.z is None or r_sh.z is None or l_hip.z is None or r_hip.z is None:
            return None
        shoulder_z = (l_sh.z + r_sh.z) / 2
        hip_z = (l_hip.z + r_hip.z) / 2
        return hip_z - shoulder_z
