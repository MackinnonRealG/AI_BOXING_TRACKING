"""Strike classifier: speed-peak candidates + pose context → typed strikes.

The classifier keeps a short pose buffer per fighter. When the speed engine
publishes a :class:`SpeedPeakEvent`, it windows the fighter's poses over the
stroke and classifies with geometric heuristics:

Hand strikes
    * extension frame = wrist farthest from the shoulder during the stroke
    * elbow straighter than ``straight_elbow_min_deg`` → jab or cross
      (lead hand from the current stance = jab, rear hand = cross)
    * dominant upward motion (``uppercut_vertical_ratio``) with a bent
      elbow → uppercut
    * curved wrist path (path/straight > ``path_curve_ratio``) or a bent
      elbow with lateral motion → hook

Kicks / knees (only reachable when the sport profile monitors those limbs)
    * knee-keypoint candidates → knee strike
    * ankle apex above the shoulder → high roundhouse
    * apex between hip and shoulder: strong torso rotation → side kick,
      curved path → mid roundhouse, otherwise front kick
    * apex below the hip → low roundhouse

Landed detection: if the opponent's pose is fresh, the strike endpoint is
compared against the opponent's profile target zones; within
``landed_max_distance`` = landed. With no opponent in frame it stays None.

Classes are filtered through ``profile.strike_types``; confidence below
``min_confidence`` demotes the label to UNKNOWN. v2 replaces these
heuristics with a small temporal model trained on labelled sparring clips.
"""

from __future__ import annotations

from collections import defaultdict, deque

from combat_vision.calibration import Calibration
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    FighterRelabeledEvent,
    KeypointName,
    Limb,
    Pose,
    SpeedPeakEvent,
    Stance,
    StanceSample,
    StrikeEvent,
    StrikeType,
    TrackedPose,
)
from combat_vision.sports.base import SportProfile
from combat_vision.utils import geometry
from combat_vision.utils.config import StrikeClassifierConfig

_BUFFER_FRAMES = 240  # ~3s at 60 FPS; must outlast max_event_duration_s

_LIMB_KEYPOINT: dict[Limb, KeypointName] = {
    Limb.LEFT_HAND: KeypointName.LEFT_WRIST,
    Limb.RIGHT_HAND: KeypointName.RIGHT_WRIST,
    Limb.LEFT_FOOT: KeypointName.LEFT_ANKLE,
    Limb.RIGHT_FOOT: KeypointName.RIGHT_ANKLE,
    Limb.LEFT_KNEE: KeypointName.LEFT_KNEE,
    Limb.RIGHT_KNEE: KeypointName.RIGHT_KNEE,
}

_ZONE_KEYPOINTS: dict[str, tuple[KeypointName, ...]] = {
    "head": (KeypointName.NOSE,),
    "body": (
        KeypointName.LEFT_SHOULDER,
        KeypointName.RIGHT_SHOULDER,
        KeypointName.LEFT_HIP,
        KeypointName.RIGHT_HIP,
    ),
    "legs": (
        KeypointName.LEFT_KNEE,
        KeypointName.RIGHT_KNEE,
        KeypointName.LEFT_ANKLE,
        KeypointName.RIGHT_ANKLE,
    ),
}


class StrikeClassifierEngine(MetricsEngine):
    """Classifies strike candidates into the active sport's strike types."""

    def __init__(
        self,
        bus: EventBus,
        profile: SportProfile,
        calibration: Calibration,
        config: StrikeClassifierConfig,
    ) -> None:
        super().__init__(bus, profile, calibration)
        self._config = config
        self._buffers: dict[FighterId, deque[TrackedPose]] = defaultdict(
            lambda: deque(maxlen=_BUFFER_FRAMES)
        )
        self._stances: dict[FighterId, Stance] = {}
        bus.subscribe(SpeedPeakEvent, self._on_candidate)
        bus.subscribe(StanceSample, self._on_stance)
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop buffered poses and stance for a label now held by a different person.

        Buffered poses are keyed by frame timestamp, not by who's wearing
        the label — without clearing this, a strike thrown moments after
        the relabel would window over a mix of the departed fighter's
        tail-end poses and the new fighter's poses. The remembered stance
        must go too: ``_lead_side`` uses it to decide jab vs. cross, and a
        stale stance would mislabel the new fighter's hand strikes until
        their own first :class:`StanceSample` arrives.
        """
        self._buffers.pop(event.fighter_id, None)
        self._stances.pop(event.fighter_id, None)

    def process(self, tracked: TrackedPose) -> None:
        """Buffer poses so stroke windows are available when candidates fire."""
        self._buffers[tracked.fighter_id].append(tracked)

    def _on_stance(self, event: StanceSample) -> None:
        """Track each fighter's current stance (lead-hand context)."""
        self._stances[event.fighter_id] = event.stance

    def _on_candidate(self, event: SpeedPeakEvent) -> None:
        """Classify one speed-peak candidate and publish a StrikeEvent."""
        window = [
            p
            for p in self._buffers[event.fighter_id]
            if event.start_s <= p.timestamp_s <= event.end_s
        ]
        if len(window) < 2:
            return

        if event.limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            strike_type, confidence = self._classify_hand(event, window)
        elif event.limb in (Limb.LEFT_KNEE, Limb.RIGHT_KNEE):
            strike_type, confidence = StrikeType.KNEE, 0.75
        else:
            strike_type, confidence = self._classify_kick(event, window)

        if not self._profile.allows(strike_type) or confidence < self._config.min_confidence:
            strike_type = StrikeType.UNKNOWN

        self._bus.publish(
            StrikeEvent(
                timestamp_s=event.end_s,
                fighter_id=event.fighter_id,
                strike_type=strike_type,
                limb=event.limb,
                speed=event.peak_speed,
                unit=event.unit,
                landed=self._detect_landed(event, window),
            )
        )

    # -- hand strikes ------------------------------------------------------

    def _classify_hand(
        self, event: SpeedPeakEvent, window: list[TrackedPose]
    ) -> tuple[StrikeType, float]:
        """Classify a hand candidate — see module docstring for the features."""
        left = event.limb == Limb.LEFT_HAND
        wrist_name = KeypointName.LEFT_WRIST if left else KeypointName.RIGHT_WRIST
        elbow_name = KeypointName.LEFT_ELBOW if left else KeypointName.RIGHT_ELBOW
        shoulder_name = KeypointName.LEFT_SHOULDER if left else KeypointName.RIGHT_SHOULDER

        path = self._track_px(window, wrist_name)
        if len(path) < 2:
            return StrikeType.UNKNOWN, 0.0

        extension_pose = self._extension_pose(window, wrist_name, shoulder_name)
        elbow_angle = self._joint_angle(
            extension_pose, shoulder_name, elbow_name, wrist_name
        )
        if elbow_angle is None:
            # Elbow/shoulder never both visible at the extension frame — no
            # reliable geometry to classify from. Missing data must not be
            # read as "arm was straight" (which is what a 180° default would
            # imply and confidently misclassify as a jab/cross).
            return StrikeType.UNKNOWN, 0.0

        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        straight = geometry.distance(path[0], path[-1])
        curve_ratio = geometry.path_length(path) / straight if straight > 0 else 1.0

        rises = -dy > self._config.uppercut_vertical_ratio * abs(dx)
        if rises and elbow_angle < self._config.straight_elbow_min_deg:
            return StrikeType.UPPERCUT, 0.75
        if elbow_angle >= self._config.straight_elbow_min_deg and (
            curve_ratio <= self._config.path_curve_ratio
        ):
            lead = self._lead_side(event.fighter_id)
            is_lead_hand = (lead == "left") == left
            return (StrikeType.JAB if is_lead_hand else StrikeType.CROSS), 0.9
        if elbow_angle <= self._config.hook_elbow_max_deg or (
            curve_ratio > self._config.path_curve_ratio
        ):
            return StrikeType.HOOK, 0.75
        return StrikeType.UNKNOWN, 0.5

    def _lead_side(self, fighter_id: FighterId) -> str:
        """'left' or 'right' — which hand is the lead, from stance context.

        Orthodox is the default when no stance has been observed yet.
        """
        stance = self._stances.get(fighter_id, Stance.ORTHODOX)
        return "right" if stance == Stance.SOUTHPAW else "left"

    # -- kicks -------------------------------------------------------------

    def _classify_kick(
        self, event: SpeedPeakEvent, window: list[TrackedPose]
    ) -> tuple[StrikeType, float]:
        """Classify a foot candidate by apex height, curvature, and rotation."""
        left = event.limb == Limb.LEFT_FOOT
        ankle_name = KeypointName.LEFT_ANKLE if left else KeypointName.RIGHT_ANKLE

        path = self._track_px(window, ankle_name)
        if len(path) < 2:
            return StrikeType.UNKNOWN, 0.0
        apex_index = min(range(len(path)), key=lambda i: path[i][1])
        apex_pose = window[apex_index].pose
        apex_y = path[apex_index][1]

        hip_y = self._mean_y(apex_pose, KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP)
        shoulder_y = self._mean_y(
            apex_pose, KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER
        )
        if hip_y is None or shoulder_y is None:
            return StrikeType.UNKNOWN, 0.0

        if apex_y < shoulder_y:
            return StrikeType.ROUNDHOUSE_HIGH, 0.7
        if apex_y >= hip_y:
            return StrikeType.ROUNDHOUSE_LOW, 0.7

        # Mid-height: separate side / round / front kicks.
        rotation = self._torso_rotation_deg(window)
        if rotation > self._config.side_kick_rotation_deg:
            return StrikeType.SIDE_KICK, 0.65
        straight = geometry.distance(path[0], path[-1])
        curve_ratio = geometry.path_length(path) / straight if straight > 0 else 1.0
        if curve_ratio > self._config.path_curve_ratio:
            return StrikeType.ROUNDHOUSE_MID, 0.7
        return StrikeType.FRONT_KICK, 0.7

    # -- landed detection --------------------------------------------------

    def _detect_landed(self, event: SpeedPeakEvent, window: list[TrackedPose]) -> bool | None:
        """Endpoint-vs-target-zone proximity check; None without an opponent."""
        opponent = self._latest_opponent_pose(event.fighter_id, event.end_s)
        if opponent is None:
            return None
        strike_kp = window[-1].pose.get(_LIMB_KEYPOINT[event.limb])
        if strike_kp is None:
            return None
        strike_px = self._calibration.to_pixels(strike_kp.x, strike_kp.y)

        threshold = (
            self._config.landed_max_distance_m
            if self._calibration.is_calibrated
            else self._config.landed_max_distance_px
        )
        for zone in self._profile.target_zones:
            for name in _ZONE_KEYPOINTS.get(zone.name, ()):
                target = opponent.get(name)
                if target is None:
                    continue
                target_px = self._calibration.to_pixels(target.x, target.y)
                if self._calibration.scale_length(
                    geometry.distance(strike_px, target_px)
                ) <= threshold:
                    return True
        return False

    def _latest_opponent_pose(self, fighter_id: FighterId, at_s: float) -> Pose | None:
        """Freshest other-fighter pose near ``at_s``, if any."""
        best: TrackedPose | None = None
        for other_id, buffer in self._buffers.items():
            if other_id == fighter_id or not buffer:
                continue
            candidate = buffer[-1]
            if abs(at_s - candidate.timestamp_s) <= self._config.opponent_max_age_s and (
                best is None or candidate.timestamp_s > best.timestamp_s
            ):
                best = candidate
        return best.pose if best else None

    # -- shared helpers ----------------------------------------------------

    def _track_px(self, window: list[TrackedPose], name: KeypointName) -> list[geometry.Point]:
        """Pixel-space trajectory of one keypoint over the window."""
        points = []
        for tracked in window:
            kp = tracked.pose.get(name)
            if kp is not None:
                points.append(self._calibration.to_pixels(kp.x, kp.y))
        return points

    def _extension_pose(
        self, window: list[TrackedPose], wrist: KeypointName, shoulder: KeypointName
    ) -> Pose:
        """The pose where the wrist is farthest from its shoulder."""

        def reach(tracked: TrackedPose) -> float:
            w, s = tracked.pose.get(wrist), tracked.pose.get(shoulder)
            if w is None or s is None:
                return -1.0
            return geometry.distance(
                self._calibration.to_pixels(w.x, w.y), self._calibration.to_pixels(s.x, s.y)
            )

        return max(window, key=reach).pose

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

    def _mean_y(self, pose: Pose, a: KeypointName, b: KeypointName) -> float | None:
        """Mean pixel y of two keypoints, or None if either is missing."""
        ka, kb = pose.get(a), pose.get(b)
        if ka is None or kb is None:
            return None
        return (
            self._calibration.to_pixels(ka.x, ka.y)[1]
            + self._calibration.to_pixels(kb.x, kb.y)[1]
        ) / 2

    def _torso_rotation_deg(self, window: list[TrackedPose]) -> float:
        """Absolute shoulder-line rotation over the window, in degrees."""
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
        return geometry.angle_delta(angles[0], angles[-1])
