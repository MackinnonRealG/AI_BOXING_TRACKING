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
class ComboEvent(Event):
    """A sequence of strikes chained within the combination time window."""

    sequence: tuple[StrikeType, ...]
    strike_timestamps: tuple[float, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class DistanceSample(Event):
    """Inter-fighter distance at an instant. ``fighter_id`` is fighter A."""

    other_fighter_id: FighterId
    distance: float
    unit: SpeedUnit
