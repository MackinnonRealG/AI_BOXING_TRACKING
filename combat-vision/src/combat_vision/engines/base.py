"""The MetricsEngine interface.

Engines are pure consumers/producers around the event bus:

* input — :meth:`process` is called once per tracked pose, in timestamp
  order; engines needing strike events subscribe to the bus instead.
* output — engines ``publish`` typed events; they never call each other.

This makes every engine unit-testable by replaying a recorded pose sequence
into :meth:`process` with no camera, model, or UI attached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from combat_vision.calibration import Calibration
from combat_vision.events.bus import EventBus
from combat_vision.events.types import TrackedPose
from combat_vision.sports.base import SportProfile


class MetricsEngine(ABC):
    """Base class for all metrics engines."""

    def __init__(self, bus: EventBus, profile: SportProfile, calibration: Calibration) -> None:
        self._bus = bus
        self._profile = profile
        self._calibration = calibration

    @abstractmethod
    def process(self, tracked: TrackedPose) -> None:
        """Consume one tracked, smoothed pose. Publish events as they occur."""

    def finish(self) -> None:  # noqa: B027 — optional hook, deliberately not abstract
        """Flush any pending state at end of stream (review mode calls this)."""
