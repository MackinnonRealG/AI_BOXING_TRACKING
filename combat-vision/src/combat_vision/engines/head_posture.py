"""Head-posture engine: periodic head-roll measurement, no verdict attached.

The eye line (left eye to right eye) compared against the shoulder line
gives a simple, camera-plane read of head roll — how level the head is
relative to the torso. Unlike :mod:`engines.guard`, :mod:`engines.rotation`,
and :mod:`engines.knee_bend`, this engine does **not** publish a fault: head
tilt alone can't distinguish sloppy head position from a deliberate slip
(head movement off the centerline is good defensive technique, not a
mistake), and there is no reliable way from a single 2D camera to tell
those apart without more context than this engine has. v1 only exposes the
raw measurement — :class:`HeadPostureSample`, decimated like
:class:`FootworkSample` — for a fighter or coach to interpret themselves, or
for a future engine with combo/exchange context to judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    HeadPostureSample,
    KeypointName,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import HeadPostureConfig


@dataclass
class _FighterHeadPosture:
    """Per-fighter sampling state."""

    last_sample_s: float = -1.0e9


class HeadPostureEngine(MetricsEngine):
    """Publishes a decimated head-roll measurement per fighter."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: HeadPostureConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._fighters: dict[FighterId, _FighterHeadPosture] = {}

    def process(self, tracked: TrackedPose) -> None:
        """Sample head roll at the configured cadence, when the face/torso is visible."""
        state = self._fighters.setdefault(tracked.fighter_id, _FighterHeadPosture())
        if tracked.timestamp_s - state.last_sample_s < self._config.sample_interval_s:
            return

        pose = tracked.pose
        l_eye, r_eye = pose.get(KeypointName.LEFT_EYE), pose.get(KeypointName.RIGHT_EYE)
        l_sh = pose.get(KeypointName.LEFT_SHOULDER)
        r_sh = pose.get(KeypointName.RIGHT_SHOULDER)
        if l_eye is None or r_eye is None or l_sh is None or r_sh is None:
            return
        state.last_sample_s = tracked.timestamp_s

        eye_angle = geometry.line_angle(
            self._calibration.to_pixels(l_eye.x, l_eye.y),
            self._calibration.to_pixels(r_eye.x, r_eye.y),
        )
        shoulder_angle = geometry.line_angle(
            self._calibration.to_pixels(l_sh.x, l_sh.y),
            self._calibration.to_pixels(r_sh.x, r_sh.y),
        )
        self._bus.publish(
            HeadPostureSample(
                timestamp_s=tracked.timestamp_s,
                fighter_id=tracked.fighter_id,
                tilt_deg=geometry.angle_delta(eye_angle, shoulder_angle),
            )
        )
