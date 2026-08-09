"""The PoseBackend interface.

Swapping MediaPipe for YOLOv8-pose (or anything else) means implementing this
one class. Everything downstream consumes canonical
:class:`~combat_vision.events.types.PersonDetection` objects and never sees
backend-specific landmark indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from combat_vision.capture.base import TimestampedFrame
from combat_vision.events.types import PersonDetection


class PoseBackend(ABC):
    """Detects people and estimates their poses in a frame."""

    @abstractmethod
    def detect(self, frame: TimestampedFrame) -> list[PersonDetection]:
        """Return all detected persons with canonical keypoints.

        Coordinates must be normalized to ``[0, 1]`` relative to the frame.
        Order is unspecified — identity assignment is the tracker's job.
        """

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""
