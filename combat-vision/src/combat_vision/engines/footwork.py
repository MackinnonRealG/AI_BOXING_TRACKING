"""Footwork engine: steps, stance width, weight shift, foot-placement heat map.

Per foot, a small state machine detects steps:

1. Each foot has an *anchor* — its last planted position. When the foot
   moves farther than ``step_min_displacement`` from the anchor it is
   airborne (a step in progress).
2. When the moving foot's speed stays below ``plant_speed`` for
   ``plant_frames`` consecutive frames, it has re-planted: a
   :class:`StepEvent` is published from anchor to the new position and the
   anchor moves.

Continuously, the engine also:

* accumulates every foot position into a per-fighter 2D histogram
  (``heatmap_bins``) — exposed via :meth:`heatmap` for the overlay and
  session reports, and
* publishes a decimated :class:`FootworkSample` (every
  ``sample_interval_s``) carrying stance width (ankle separation) and a
  weight-shift estimate: the hip midpoint projected onto the ankle base
  line, -1 = fully over the left foot, +1 = fully over the right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    FootworkSample,
    Keypoint,
    KeypointName,
    Limb,
    StepEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import FootworkConfig

_FEET: dict[Limb, KeypointName] = {
    Limb.LEFT_FOOT: KeypointName.LEFT_ANKLE,
    Limb.RIGHT_FOOT: KeypointName.RIGHT_ANKLE,
}


@dataclass
class _FootState:
    """Step-detection state for one foot."""

    anchor_px: geometry.Point | None = None
    anchor_norm: tuple[float, float] = (0.0, 0.0)
    last_px: geometry.Point | None = None
    last_t: float | None = None
    airborne: bool = False
    slow_frames: int = 0


@dataclass
class _FighterFootwork:
    """Per-fighter footwork state."""

    feet: dict[Limb, _FootState] = field(default_factory=dict)
    heatmap: np.ndarray | None = None
    last_sample_s: float = -1.0e9


class FootworkEngine(MetricsEngine):
    """Tracks both feet continuously per fighter."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: FootworkConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._fighters: dict[FighterId, _FighterFootwork] = {}
        if calibration.is_calibrated:
            self._step_min = config.step_min_displacement_m
            self._plant_speed = config.plant_speed_mps
        else:
            self._step_min = config.step_min_displacement_px
            self._plant_speed = config.plant_speed_pxps

    def process(self, tracked: TrackedPose) -> None:
        """Advance step detection, the heat map, and periodic samples."""
        state = self._fighters.setdefault(tracked.fighter_id, _FighterFootwork())
        for limb, keypoint_name in _FEET.items():
            kp = tracked.pose.get(keypoint_name)
            if kp is None:
                continue
            self._accumulate_heat(state, kp)
            self._step_machine(
                state.feet.setdefault(limb, _FootState()),
                tracked.fighter_id,
                limb,
                kp,
                tracked.timestamp_s,
            )
        self._maybe_sample(state, tracked)

    def heatmap(self, fighter_id: FighterId) -> np.ndarray | None:
        """The fighter's accumulated foot-placement histogram (y-bins, x-bins)."""
        state = self._fighters.get(fighter_id)
        return None if state is None else state.heatmap

    # -- internals ---------------------------------------------------------

    def _step_machine(
        self,
        foot: _FootState,
        fighter_id: FighterId,
        limb: Limb,
        kp: Keypoint,
        timestamp_s: float,
    ) -> None:
        """One tick of the anchored/airborne/replant state machine."""
        position_px = self._calibration.to_pixels(kp.x, kp.y)
        if foot.anchor_px is None or foot.last_px is None or foot.last_t is None:
            foot.anchor_px, foot.anchor_norm = position_px, (kp.x, kp.y)
            foot.last_px, foot.last_t = position_px, timestamp_s
            return
        dt = timestamp_s - foot.last_t
        if dt <= 0:
            return
        speed = self._calibration.scale_speed(
            geometry.distance(position_px, foot.last_px) / dt
        )
        foot.last_px, foot.last_t = position_px, timestamp_s

        displacement = self._calibration.scale_length(
            geometry.distance(position_px, foot.anchor_px)
        )
        if not foot.airborne:
            if displacement > self._step_min:
                foot.airborne = True
                foot.slow_frames = 0
            return

        foot.slow_frames = foot.slow_frames + 1 if speed < self._plant_speed else 0
        if foot.slow_frames >= self._config.plant_frames:
            self._bus.publish(
                StepEvent(
                    timestamp_s=timestamp_s,
                    fighter_id=fighter_id,
                    foot=limb,
                    from_xy=foot.anchor_norm,
                    to_xy=(kp.x, kp.y),
                    displacement=displacement,
                    unit=self._calibration.unit,
                )
            )
            foot.anchor_px, foot.anchor_norm = position_px, (kp.x, kp.y)
            foot.airborne = False
            foot.slow_frames = 0

    def _accumulate_heat(self, state: _FighterFootwork, kp: Keypoint) -> None:
        """Add one foot observation to the placement histogram."""
        bins_x, bins_y = self._config.heatmap_bins
        if state.heatmap is None:
            state.heatmap = np.zeros((bins_y, bins_x), dtype=np.float32)
        bx = min(max(int(kp.x * bins_x), 0), bins_x - 1)
        by = min(max(int(kp.y * bins_y), 0), bins_y - 1)
        state.heatmap[by, bx] += 1.0

    def _maybe_sample(self, state: _FighterFootwork, tracked: TrackedPose) -> None:
        """Publish a decimated FootworkSample with width and weight shift."""
        if tracked.timestamp_s - state.last_sample_s < self._config.sample_interval_s:
            return
        pose = tracked.pose
        l_ankle, r_ankle = pose.get(KeypointName.LEFT_ANKLE), pose.get(KeypointName.RIGHT_ANKLE)
        l_hip, r_hip = pose.get(KeypointName.LEFT_HIP), pose.get(KeypointName.RIGHT_HIP)
        if l_ankle is None or r_ankle is None or l_hip is None or r_hip is None:
            return
        state.last_sample_s = tracked.timestamp_s

        l_px = self._calibration.to_pixels(l_ankle.x, l_ankle.y)
        r_px = self._calibration.to_pixels(r_ankle.x, r_ankle.y)
        width = self._calibration.scale_length(geometry.distance(l_px, r_px))

        # Project the hip midpoint onto the ankle base line: t in [0, 1]
        # (0 = left ankle, 1 = right ankle) -> weight shift in [-1, 1].
        hip_mid = geometry.midpoint(
            self._calibration.to_pixels(l_hip.x, l_hip.y),
            self._calibration.to_pixels(r_hip.x, r_hip.y),
        )
        base = (r_px[0] - l_px[0], r_px[1] - l_px[1])
        base_sq = base[0] ** 2 + base[1] ** 2
        if base_sq == 0:
            weight_shift = 0.0
        else:
            t = ((hip_mid[0] - l_px[0]) * base[0] + (hip_mid[1] - l_px[1]) * base[1]) / base_sq
            weight_shift = max(-1.0, min(2.0 * t - 1.0, 1.0))

        self._bus.publish(
            FootworkSample(
                timestamp_s=tracked.timestamp_s,
                fighter_id=tracked.fighter_id,
                stance_width=width,
                unit=self._calibration.unit,
                weight_shift=weight_shift,
            )
        )
