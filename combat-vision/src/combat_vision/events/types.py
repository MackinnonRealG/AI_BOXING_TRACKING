"""Core data contracts shared by every pipeline stage.

Design rules:

* Frames and poses flow *down* the pipeline as immutable dataclasses.
* Metrics engines emit :class:`Event` subclasses *onto the bus* — they never
  call each other directly.
* Coordinates in a :class:`Pose` are **normalized** to ``[0, 1]`` relative to
  the frame, so contracts stay resolution-independent. Engines that need
  physical units go through :class:`combat_vision.calibration.Calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

FighterId: TypeAlias = str
"""Stable fighter label assigned by the tracker (``"A"`` or ``"B"``)."""


class KeypointName(StrEnum):
    """Canonical keypoint vocabulary.

    Every :class:`~combat_vision.pose.base.PoseBackend` maps its native
    landmark set (MediaPipe's 33, COCO's 17, ...) onto these names, so
    downstream code never depends on a specific model.
    """

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


SKELETON_EDGES: tuple[tuple[KeypointName, KeypointName], ...] = (
    (KeypointName.NOSE, KeypointName.LEFT_EYE),
    (KeypointName.NOSE, KeypointName.RIGHT_EYE),
    (KeypointName.LEFT_EYE, KeypointName.LEFT_EAR),
    (KeypointName.RIGHT_EYE, KeypointName.RIGHT_EAR),
    (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW),
    (KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW),
    (KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE),
    (KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE),
    (KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
    (KeypointName.LEFT_ANKLE, KeypointName.LEFT_HEEL),
    (KeypointName.LEFT_HEEL, KeypointName.LEFT_FOOT_INDEX),
    (KeypointName.RIGHT_ANKLE, KeypointName.RIGHT_HEEL),
    (KeypointName.RIGHT_HEEL, KeypointName.RIGHT_FOOT_INDEX),
)
"""Bone connectivity used by the overlay renderer."""


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A single body landmark in normalized image coordinates."""

    x: float
    """Horizontal position, 0 (left edge) .. 1 (right edge)."""
    y: float
    """Vertical position, 0 (top edge) .. 1 (bottom edge)."""
    z: float | None = None
    """Backend-relative depth, if the backend provides one (smaller = closer)."""
    visibility: float = 1.0
    """Backend confidence that the landmark is visible, 0..1."""


@dataclass(frozen=True, slots=True)
class Pose:
    """A full-body pose for one person in one frame."""

    keypoints: dict[KeypointName, Keypoint]

    def get(self, name: KeypointName) -> Keypoint | None:
        """Return the keypoint, or ``None`` if the backend did not report it."""
        return self.keypoints.get(name)

    def centroid(self) -> tuple[float, float]:
        """Mean (x, y) over available keypoints — used for track association."""
        xs = [k.x for k in self.keypoints.values()]
        ys = [k.y for k in self.keypoints.values()]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in normalized coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def center(self) -> tuple[float, float]:
        """Return the box center (x, y)."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


@dataclass(frozen=True, slots=True)
class PersonDetection:
    """One detected person before identity assignment."""

    pose: Pose
    bbox: BBox
    score: float
    """Overall detection confidence, 0..1."""


@dataclass(frozen=True, slots=True)
class TrackedPose:
    """A pose bound to a stable fighter identity — the engines' input unit."""

    fighter_id: FighterId
    pose: Pose
    timestamp_s: float
    """Seconds since the session started (monotonic within a source)."""
    camera_id: str = "cam0"


class SpeedUnit(StrEnum):
    """Unit of a speed measurement, depending on calibration state."""

    METERS_PER_SECOND = "m/s"
    PIXELS_PER_SECOND = "px/s"


class Limb(StrEnum):
    """Which limb produced a strike or candidate."""

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"


class StrikeType(StrEnum):
    """All strike classes across supported sports.

    The active subset for a session comes from the
    :class:`~combat_vision.sports.base.SportProfile`.
    """

    JAB = "jab"
    CROSS = "cross"
    HOOK = "hook"
    UPPERCUT = "uppercut"
    FRONT_KICK = "front_kick"
    ROUNDHOUSE_LOW = "roundhouse_low"
    ROUNDHOUSE_MID = "roundhouse_mid"
    ROUNDHOUSE_HIGH = "roundhouse_high"
    SIDE_KICK = "side_kick"
    KNEE = "knee"
    UNKNOWN = "unknown"


class Stance(StrEnum):
    """Fighter stance classification."""

    ORTHODOX = "orthodox"
    SOUTHPAW = "southpaw"
    SQUARE = "square"


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for everything published on the event bus."""

    timestamp_s: float
    fighter_id: FighterId


@dataclass(frozen=True, slots=True)
class SpeedPeakEvent(Event):
    """A punch *candidate*: a hand-speed stroke that peaked above threshold.

    Emitted by the speed engine; the strike classifier consumes these (plus
    pose context) to produce classified :class:`StrikeEvent` s.
    """

    limb: Limb
    peak_speed: float
    unit: SpeedUnit
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class StrikeEvent(Event):
    """A classified strike with its measured/estimated attributes."""

    strike_type: StrikeType
    limb: Limb
    speed: float
    unit: SpeedUnit
    power_score: float | None = None
    """Estimated 0..100 power score. An *estimate*, never measured force."""
    landed: bool | None = None
    """Whether the strike connected, if determinable. None = unknown."""


@dataclass(frozen=True, slots=True)
class PowerEstimateEvent(Event):
    """Estimated power score for one strike candidate.

    Joined to the matching :class:`SpeedPeakEvent` / :class:`StrikeEvent` by
    ``(fighter_id, limb, end_s)`` — all three originate from the same stroke.
    """

    limb: Limb
    score: float
    """0..100 *estimated* power — kinematic estimate, never measured force."""
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class StanceSample(Event):
    """The debounced current stance of a fighter (published on change)."""

    stance: Stance


@dataclass(frozen=True, slots=True)
class FootworkSample(Event):
    """Periodic footwork state: stance width and inferred weight shift."""

    stance_width: float
    unit: SpeedUnit
    weight_shift: float
    """-1 = weight fully over the left foot, +1 fully over the right, 0 centered."""


@dataclass(frozen=True, slots=True)
class StepEvent(Event):
    """One footstep: a foot lifted and re-planted somewhere else."""

    foot: Limb
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    displacement: float
    unit: SpeedUnit


@dataclass(frozen=True, slots=True)
class StanceSwitchEvent(Event):
    """A fighter switched stance (e.g. orthodox → southpaw)."""

    from_stance: Stance
    to_stance: Stance


@dataclass(frozen=True, slots=True)
class CleanTechniqueEvent(Event):
    """A hand strike that passed a technique check cleanly — the positive
    counterpart to :class:`RotationFaultEvent` / :class:`LegDriveFaultEvent`.

    ``check`` identifies which check passed: ``"hip_rotation"`` (from
    :mod:`engines.rotation`) or ``"leg_drive"`` (from
    :mod:`engines.knee_bend`). Only published when the check was actually
    evaluable — see the matching fault event's docstring for the gating
    that applies to both the good and bad outcome alike (e.g. a jab's low
    shoulder rotation is never judged either way).
    """

    check: str
    limb: Limb


@dataclass(frozen=True, slots=True)
class RotationFaultEvent(Event):
    """A hand strike whose shoulders turned but whose hips didn't follow.

    Only published for strokes where the shoulders rotated enough for the
    comparison to mean anything — a jab's naturally low rotation is not
    treated as a fault. Reads shoulder/hip *line angle* in the image plane,
    like :class:`PowerEstimateEvent`'s rotation component; a fighter facing
    the camera square-on reads as less rotation than they actually produced.
    """

    limb: Limb
    shoulder_rotation_deg: float
    hip_rotation_deg: float


@dataclass(frozen=True, slots=True)
class DepthPostureSample(Event):
    """Approximate, unitless depth-based posture reading.

    Built from MediaPipe's ``z`` landmark coordinate — a rough, uncalibrated,
    single-view depth estimate, noisier and less trustworthy than the (x, y)
    geometry every other engine relies on. Positive ``*_elbow_flare`` means
    that elbow sits closer to the camera than the shoulder/hip reference
    (pushed forward, out of the guard); positive ``torso_lean`` means the
    shoulders sit closer to the camera than the hips (leaning forward into
    range). Any field is None when the needed keypoints (including z)
    weren't available. Treat these as directional hints, not verified
    biomechanics — this is explicitly *not* published as a fault.
    """

    left_elbow_flare: float | None
    right_elbow_flare: float | None
    torso_lean: float | None


@dataclass(frozen=True, slots=True)
class HeadPostureSample(Event):
    """Periodic head-roll measurement — how level the head is vs. the shoulders.

    ``tilt_deg`` is the *unsigned* angular gap between the eye line and the
    shoulder line: 0 means the head is level with the shoulders, larger
    means more tilt. Deliberately not published as a fault: head tilt alone
    can't distinguish sloppy head position from a deliberate slip, so v1
    only exposes the measurement rather than judging it.
    """

    tilt_deg: float


@dataclass(frozen=True, slots=True)
class KneeBendStateEvent(Event):
    """Debounced continuous knee posture: both knees locked-straight or bent.

    ``locked`` is True only when *both* knees are near-straight — a stronger,
    less occlusion/false-positive-prone signal than judging either leg alone.
    """

    locked: bool


@dataclass(frozen=True, slots=True)
class LegDriveFaultEvent(Event):
    """A hand strike thrown with both knees already locked at the stroke's start.

    ``knee_angle_deg`` is the more bent (smaller) of the two knee angles, in
    degrees — 180° is fully locked, so even this one was already near-straight.
    """

    limb: Limb
    knee_angle_deg: float


@dataclass(frozen=True, slots=True)
class GuardStateEvent(Event):
    """One hand's guard-up/guard-down state, published on debounced change.

    ``guard_up`` reflects height only (hand at or above chin level, within
    tolerance) — a hand held wide at head height still reads as "up" in v1.
    """

    hand: Limb
    guard_up: bool


@dataclass(frozen=True, slots=True)
class ComboEvent(Event):
    """A sequence of strikes chained within the combination time window."""

    sequence: tuple[StrikeType, ...]
    strike_timestamps: tuple[float, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class FighterRelabeledEvent(Event):
    """``fighter_id`` now refers to a *different* physical person.

    Published by the pipeline when a tracker reports a recycled label (see
    :meth:`~combat_vision.tracking.base.Tracker.consume_relabeled`). Any sink
    holding per-label state that assumes one continuous person — accumulated
    counts, filters, heat maps — must drop that state on receipt, or it will
    attribute the departed fighter's history to whoever now holds the label.

    Broadcast on the bus rather than polled: ``consume_relabeled`` clears on
    read, so exactly one caller can drain it. The pipeline is that caller and
    fans the result out to everyone else.
    """


@dataclass(frozen=True, slots=True)
class DistanceSample(Event):
    """Inter-fighter distance at an instant. ``fighter_id`` is fighter A."""

    other_fighter_id: FighterId
    distance: float
    unit: SpeedUnit
