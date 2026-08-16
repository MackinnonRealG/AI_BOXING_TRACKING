"""Knee-bend engine: locked-knee posture and no-leg-drive fault detection.

Punching power and mobility come from bent, "loaded" knees — locked-straight
legs can't push off the floor or absorb impact. This engine watches knee
angle (hip-knee-ankle, the same joint-angle geometry :mod:`engines.power`
already uses for kick extension) in two ways:

1. **Continuous stance posture** — both knees are classified locked/bent
   every frame (mirrors :mod:`engines.guard`'s debounce state machine, so a
   brief straightening between reps doesn't false-positive), publishing
   :class:`KneeBendStateEvent` on debounced change. A fighter standing with
   locked knees between exchanges can't react or move.
2. **Per-punch leg drive** — on each hand-strike candidate, the knee angle
   at the *start* of the stroke (before the arm accelerates) is checked; both
   legs already locked at that moment means the punch was thrown from a
   static, flat-footed stance with nothing pushing through it, and publishes
   :class:`LegDriveFaultEvent`. A punch thrown with at least one knee bent
   publishes :class:`CleanTechniqueEvent` instead — the good reps are
   logged too, not just the faults. Kicks are excluded — a kicking leg
   extending is the strike itself, not a fault.

The fault fires only when *both* knees are near-locked, not either leg alone
— picking out the specific "rear driving leg" would need a reliable
lead/rear read per sport (stance.py already derives one, but this engine
keeps its input independent for now), and requiring both is a stronger,
less occlusion-prone signal in the meantime.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    CleanTechniqueEvent,
    FighterId,
    KeypointName,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    Limb,
    Pose,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import KneeBendConfig

_BUFFER_FRAMES = 240  # ~3s at 60 FPS; must outlast max_event_duration_s
_HAND_LIMBS = (Limb.LEFT_HAND, Limb.RIGHT_HAND)
_KNEES: tuple[tuple[KeypointName, KeypointName, KeypointName], ...] = (
    (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
)


@dataclass
class _PostureState:
    """Debounce state for one fighter's continuous knee posture."""

    current_locked: bool | None = None
    candidate_locked: bool | None = None
    candidate_since_s: float = 0.0


class KneeBendEngine(MetricsEngine):
    """Flags locked-knee posture and hand strikes thrown with no leg drive."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: KneeBendConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._buffers: dict[FighterId, deque[TrackedPose]] = defaultdict(
            lambda: deque(maxlen=_BUFFER_FRAMES)
        )
        self._posture: dict[FighterId, _PostureState] = {}
        bus.subscribe(SpeedPeakEvent, self._on_candidate)

    def process(self, tracked: TrackedPose) -> None:
        """Buffer poses and advance the continuous locked-knee posture check."""
        self._buffers[tracked.fighter_id].append(tracked)
        angles = self._knee_angles(tracked.pose)
        if len(angles) < 2:
            return
        locked = all(a >= self._config.locked_angle_deg for a in angles)
        self._update_posture(tracked.fighter_id, locked, tracked.timestamp_s)

    def _on_candidate(self, event: SpeedPeakEvent) -> None:
        """Flag a hand strike whose legs were already locked at the stroke's start."""
        if event.limb not in _HAND_LIMBS:
            return
        window = [
            p
            for p in self._buffers[event.fighter_id]
            if event.start_s <= p.timestamp_s <= event.end_s
        ]
        if not window:
            return
        angles = self._knee_angles(window[0].pose)
        if len(angles) < 2:
            return  # can't judge leg drive without both legs measured
        if not all(a >= self._config.locked_angle_deg for a in angles):
            self._bus.publish(
                CleanTechniqueEvent(
                    timestamp_s=event.end_s,
                    fighter_id=event.fighter_id,
                    check="leg_drive",
                    limb=event.limb,
                )
            )
            return
        self._bus.publish(
            LegDriveFaultEvent(
                timestamp_s=event.end_s,
                fighter_id=event.fighter_id,
                limb=event.limb,
                knee_angle_deg=min(angles),
            )
        )

    def _knee_angles(self, pose: Pose) -> list[float]:
        """Hip-knee-ankle angle for each leg with all three keypoints visible."""
        angles = []
        for hip, knee, ankle in _KNEES:
            h, k, a = pose.get(hip), pose.get(knee), pose.get(ankle)
            if h is None or k is None or a is None:
                continue
            angles.append(
                geometry.angle_at(
                    self._calibration.to_pixels(k.x, k.y),
                    self._calibration.to_pixels(h.x, h.y),
                    self._calibration.to_pixels(a.x, a.y),
                )
            )
        return angles

    def _update_posture(
        self, fighter_id: FighterId, locked_now: bool, timestamp_s: float
    ) -> None:
        state = self._posture.setdefault(fighter_id, _PostureState())
        if locked_now != state.candidate_locked:
            state.candidate_locked = locked_now
            state.candidate_since_s = timestamp_s
            return
        stable_for = timestamp_s - state.candidate_since_s
        debounce = self._config.lock_debounce_s if locked_now else self._config.bend_debounce_s
        if locked_now == state.current_locked or stable_for < debounce:
            return

        state.current_locked = locked_now
        self._bus.publish(
            KneeBendStateEvent(
                timestamp_s=timestamp_s, fighter_id=fighter_id, locked=locked_now
            )
        )
