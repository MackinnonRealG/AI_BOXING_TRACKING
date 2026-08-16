"""Rotation engine: hip-shoulder separation ("power source") fault detection.

Real punching power comes from hip rotation driving through the shoulders,
not the arm alone — hooks and crosses thrown "square" (shoulders turn, hips
stay put) look fast on camera but transfer little of the kinetic chain. This
engine flags exactly that gap.

For each hand-strike candidate (:class:`SpeedPeakEvent`), it measures
shoulder-line and hip-line angular displacement over the stroke window (the
same line-angle geometry :mod:`engines.power` already uses for its rotation
component) and compares them:

* strokes with little shoulder rotation at all (jabs, most lead-hand work)
  are skipped entirely — a jab isn't supposed to need hip drive, so judging
  it by this metric would just be noise.
* strokes where the shoulders visibly turned but the hip rotation trailed
  far behind (below ``min_hip_ratio`` of the shoulder rotation) publish a
  :class:`RotationFaultEvent` — shoulders turned, hips didn't follow.
* strokes that clear that ratio publish a :class:`CleanTechniqueEvent`
  instead — the good reps are logged too, not just the faults.

Like `power.py`, this reads shoulder/hip *line angle* in the image plane,
not true 3D hip rotation — a side-on camera reads it well; a fighter facing
the camera square-on reads as less rotation than they actually produced.
"""

from __future__ import annotations

from collections import defaultdict, deque

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    CleanTechniqueEvent,
    FighterId,
    FighterRelabeledEvent,
    KeypointName,
    Limb,
    RotationFaultEvent,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import RotationEngineConfig

_BUFFER_FRAMES = 240  # ~3s at 60 FPS; must outlast max_event_duration_s
_HAND_LIMBS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)


class RotationEngine(MetricsEngine):
    """Flags hand strikes thrown with shoulder turn but no matching hip turn."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: RotationEngineConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._buffers: dict[FighterId, deque[TrackedPose]] = defaultdict(
            lambda: deque(maxlen=_BUFFER_FRAMES)
        )
        bus.subscribe(SpeedPeakEvent, self._on_candidate)
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop buffered poses for a label now held by a different person.

        Buffered poses are keyed by frame timestamp, not by who's wearing
        the label — without clearing this, a strike thrown moments after
        the relabel would window over a mix of the departed fighter's
        tail-end poses and the new fighter's poses, computing a shoulder/hip
        rotation delta across two different people's torsos.
        """
        self._buffers.pop(event.fighter_id, None)

    def process(self, tracked: TrackedPose) -> None:
        """Buffer poses so stroke windows are available when candidates fire."""
        self._buffers[tracked.fighter_id].append(tracked)

    def _on_candidate(self, event: SpeedPeakEvent) -> None:
        """Compare hip vs shoulder rotation over the stroke; flag the gap."""
        if event.limb not in _HAND_LIMBS:
            return
        window = [
            p
            for p in self._buffers[event.fighter_id]
            if event.start_s <= p.timestamp_s <= event.end_s
        ]
        if len(window) < 2:
            return

        shoulder_rotation = self._line_rotation(
            window, KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER
        )
        if (
            shoulder_rotation is None
            or shoulder_rotation < self._config.min_shoulder_rotation_deg
        ):
            return  # not enough shoulder turn for the hip comparison to mean anything

        hip_rotation = self._line_rotation(window, KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP)
        if hip_rotation is None:
            return

        if hip_rotation / shoulder_rotation >= self._config.min_hip_ratio:
            self._bus.publish(
                CleanTechniqueEvent(
                    timestamp_s=event.end_s,
                    fighter_id=event.fighter_id,
                    check="hip_rotation",
                    limb=event.limb,
                )
            )
            return

        self._bus.publish(
            RotationFaultEvent(
                timestamp_s=event.end_s,
                fighter_id=event.fighter_id,
                limb=event.limb,
                shoulder_rotation_deg=shoulder_rotation,
                hip_rotation_deg=hip_rotation,
            )
        )

    def _line_rotation(
        self, window: list[TrackedPose], left: KeypointName, right: KeypointName
    ) -> float | None:
        """Absolute line-angle rotation of a keypoint pair over the window."""
        angles = []
        for tracked in window:
            l_kp, r_kp = tracked.pose.get(left), tracked.pose.get(right)
            if l_kp is None or r_kp is None:
                continue
            angles.append(
                geometry.line_angle(
                    self._calibration.to_pixels(l_kp.x, l_kp.y),
                    self._calibration.to_pixels(r_kp.x, r_kp.y),
                )
            )
        if len(angles) < 2:
            return None
        return geometry.angle_delta(angles[0], angles[-1])
