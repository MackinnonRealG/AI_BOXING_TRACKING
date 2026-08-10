"""Power engine — estimated strike power from kinematics.

IMPORTANT: power is an *estimate* derived from pose kinematics, never
measured force. All UI/reports must label it "estimated power score".

For each :class:`SpeedPeakEvent` the engine computes three normalized
components over the stroke window and blends them with config weights:

* **speed** — peak limb speed relative to ``speed_ceiling`` (a fast hand is
  the dominant term, weight 0.5 by default),
* **extension** — elbow (or knee, for kicks) angle at maximum reach; a
  straighter limb at impact transfers more of the kinetic chain,
* **rotation** — shoulder-line angular speed over the stroke; torso torque
  is what separates an arm punch from a whole-body punch.

``score = 100 * (w_s * speed + w_e * extension + w_r * rotation)`` published
as a :class:`PowerEstimateEvent` keyed to the stroke by
``(fighter_id, limb, end_s)``.
"""

from __future__ import annotations

from collections import defaultdict, deque

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    KeypointName,
    Limb,
    Pose,
    PowerEstimateEvent,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import PowerEngineConfig

_BUFFER_FRAMES = 240

_EXTENSION_MIN_DEG = 90.0  # elbow/knee angle mapping to extension component 0
_EXTENSION_MAX_DEG = 180.0  # fully straight limb maps to component 1

# (proximal, joint, distal) triple per limb for the extension component.
_LIMB_JOINTS: dict[Limb, tuple[KeypointName, KeypointName, KeypointName]] = {
    Limb.LEFT_HAND: (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
    Limb.RIGHT_HAND: (
        KeypointName.RIGHT_SHOULDER,
        KeypointName.RIGHT_ELBOW,
        KeypointName.RIGHT_WRIST,
    ),
    Limb.LEFT_FOOT: (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    Limb.RIGHT_FOOT: (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
    Limb.LEFT_KNEE: (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    Limb.RIGHT_KNEE: (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
}


class PowerEngine(MetricsEngine):
    """Computes a 0–100 estimated power score for each strike candidate."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: PowerEngineConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._buffers: dict[FighterId, deque[TrackedPose]] = defaultdict(
            lambda: deque(maxlen=_BUFFER_FRAMES)
        )
        bus.subscribe(SpeedPeakEvent, self._on_candidate)

    @property
    def _speed_ceiling(self) -> float:
        """Speed-normalization ceiling in the current calibration's unit."""
        if self._calibration.is_calibrated:
            return self._config.speed_ceiling_mps
        return self._config.speed_ceiling_pxps

    def process(self, tracked: TrackedPose) -> None:
        """Buffer poses so stroke windows are available when candidates fire."""
        self._buffers[tracked.fighter_id].append(tracked)

    def _on_candidate(self, event: SpeedPeakEvent) -> None:
        """Score the stroke and publish a PowerEstimateEvent."""
        window = [
            p
            for p in self._buffers[event.fighter_id]
            if event.start_s <= p.timestamp_s <= event.end_s
        ]
        if len(window) < 2:
            return

        speed_component = min(event.peak_speed / self._speed_ceiling, 1.0)
        extension_component = self._extension(event.limb, window)
        rotation_component = self._rotation(window, event.end_s - event.start_s)

        score = 100.0 * (
            self._config.speed_weight * speed_component
            + self._config.extension_weight * extension_component
            + self._config.rotation_weight * rotation_component
        )
        self._bus.publish(
            PowerEstimateEvent(
                timestamp_s=event.end_s,
                fighter_id=event.fighter_id,
                limb=event.limb,
                score=min(score, 100.0),
                start_s=event.start_s,
                end_s=event.end_s,
            )
        )

    def _extension(self, limb: Limb, window: list[TrackedPose]) -> float:
        """Joint straightness (0..1) at the stroke's point of maximum reach.

        Returns 0.0 (no extension credit, not full credit) when the limb's
        proximal/distal keypoints are never both visible over the window, or
        when the joint vertex itself is missing at the best-reach frame — an
        occluded limb must not score as if it were fully extended.
        """
        proximal, joint, distal = _LIMB_JOINTS[limb]

        def reach(tracked: TrackedPose) -> float:
            p, d = tracked.pose.get(proximal), tracked.pose.get(distal)
            if p is None or d is None:
                return -1.0
            return geometry.distance(
                self._calibration.to_pixels(p.x, p.y), self._calibration.to_pixels(d.x, d.y)
            )

        best = max(window, key=reach)
        if reach(best) < 0:
            return 0.0
        angle = self._joint_angle(best.pose, proximal, joint, distal)
        if angle is None:
            return 0.0
        normalized = (angle - _EXTENSION_MIN_DEG) / (_EXTENSION_MAX_DEG - _EXTENSION_MIN_DEG)
        return max(0.0, min(normalized, 1.0))

    def _rotation(self, window: list[TrackedPose], duration_s: float) -> float:
        """Shoulder-line angular speed over the stroke, normalized 0..1."""
        if duration_s <= 0:
            return 0.0
        angles = []
        for tracked in window:
            l_sh = tracked.pose.get(KeypointName.LEFT_SHOULDER)
            r_sh = tracked.pose.get(KeypointName.RIGHT_SHOULDER)
            if l_sh is None or r_sh is None:
                continue
            angles.append(
                geometry.line_angle(
                    self._calibration.to_pixels(l_sh.x, l_sh.y),
                    self._calibration.to_pixels(r_sh.x, r_sh.y),
                )
            )
        if len(angles) < 2:
            return 0.0
        angular_speed = geometry.angle_delta(angles[0], angles[-1]) / duration_s
        return min(angular_speed / self._config.rotation_ceiling_dps, 1.0)

    def _joint_angle(
        self, pose: Pose, a: KeypointName, vertex: KeypointName, b: KeypointName
    ) -> float | None:
        """Angle at ``vertex`` in degrees, or None if any keypoint is missing."""
        pa, pv, pb = pose.get(a), pose.get(vertex), pose.get(b)
        if pa is None or pv is None or pb is None:
            return None
        return geometry.angle_at(
            self._calibration.to_pixels(pv.x, pv.y),
            self._calibration.to_pixels(pa.x, pa.y),
            self._calibration.to_pixels(pb.x, pb.y),
        )
