"""Typed event definitions and the in-process event bus.

Everything the pipeline communicates — poses, strikes, steps, stance
switches, combinations — is a frozen dataclass defined in
:mod:`combat_vision.events.types` and routed through
:class:`combat_vision.events.bus.EventBus`.
"""

from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    BBox,
    ComboEvent,
    DistanceSample,
    Event,
    FootworkSample,
    Keypoint,
    KeypointName,
    Limb,
    PersonDetection,
    Pose,
    PowerEstimateEvent,
    SpeedPeakEvent,
    SpeedUnit,
    Stance,
    StanceSample,
    StanceSwitchEvent,
    StepEvent,
    StrikeEvent,
    StrikeType,
    TrackedPose,
)

__all__ = [
    "BBox",
    "ComboEvent",
    "DistanceSample",
    "Event",
    "EventBus",
    "FootworkSample",
    "Keypoint",
    "KeypointName",
    "Limb",
    "PersonDetection",
    "Pose",
    "PowerEstimateEvent",
    "SpeedPeakEvent",
    "SpeedUnit",
    "Stance",
    "StanceSample",
    "StanceSwitchEvent",
    "StepEvent",
    "StrikeEvent",
    "StrikeType",
    "TrackedPose",
]
