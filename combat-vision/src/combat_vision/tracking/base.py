"""The Tracker protocol every tracking implementation satisfies."""

from __future__ import annotations

from typing import Protocol

from combat_vision.events.types import PersonDetection, TrackedPose


class Tracker(Protocol):
    """Assigns persistent fighter identities to per-frame detections."""

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Match detections to identities and return labelled poses."""
        ...
