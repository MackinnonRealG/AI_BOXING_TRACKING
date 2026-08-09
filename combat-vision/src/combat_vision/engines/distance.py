"""Distance engine: inter-fighter range over time.

Buffers the latest hip-midpoint per fighter; whenever two fighters have
poses within ``max_pose_age_s`` of each other, publishes a
:class:`DistanceSample` decimated to ``sample_interval_s`` (full-rate
distance would swamp storage without adding information). Fighter labels in
the sample are ordered (A then B) so consumers get one canonical series.
"""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import DistanceSample, FighterId, KeypointName, TrackedPose
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import DistanceEngineConfig


class DistanceEngine(MetricsEngine):
    """Computes fighter ↔ fighter distance samples."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: DistanceEngineConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._latest: dict[FighterId, tuple[float, geometry.Point]] = {}
        self._last_sample_s = -1.0e9

    def process(self, tracked: TrackedPose) -> None:
        """Update this fighter's position and sample pairwise distance."""
        l_hip = tracked.pose.get(KeypointName.LEFT_HIP)
        r_hip = tracked.pose.get(KeypointName.RIGHT_HIP)
        if l_hip is None or r_hip is None:
            return
        hip_mid = geometry.midpoint(
            self._calibration.to_pixels(l_hip.x, l_hip.y),
            self._calibration.to_pixels(r_hip.x, r_hip.y),
        )
        self._latest[tracked.fighter_id] = (tracked.timestamp_s, hip_mid)

        if tracked.timestamp_s - self._last_sample_s < self._config.sample_interval_s:
            return
        for other_id, (other_t, other_mid) in self._latest.items():
            if other_id == tracked.fighter_id:
                continue
            if tracked.timestamp_s - other_t > self._config.max_pose_age_s:
                continue
            first, second = sorted((tracked.fighter_id, other_id))
            self._last_sample_s = tracked.timestamp_s
            self._bus.publish(
                DistanceSample(
                    timestamp_s=tracked.timestamp_s,
                    fighter_id=first,
                    other_fighter_id=second,
                    distance=self._calibration.scale_length(
                        geometry.distance(hip_mid, other_mid)
                    ),
                    unit=self._calibration.unit,
                )
            )
            return
