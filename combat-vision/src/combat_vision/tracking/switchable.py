"""A tracker wrapper that can hot-swap implementations at runtime.

Powers the live on/off toggle for the supervision (ByteTrack) tracker: the
overlay's ``t`` key calls :meth:`toggle`, flipping between the primary and
fallback tracker without restarting the pipeline. Switching resets track
state, so identities may relabel on the frames right after a toggle.
"""

from __future__ import annotations

import logging

from combat_vision.events.types import PersonDetection, TrackedPose
from combat_vision.tracking.base import Tracker

logger = logging.getLogger(__name__)


class SwitchableTracker:
    """Delegates to one of two trackers, switchable while running."""

    def __init__(self, primary: Tracker, fallback: Tracker, use_primary: bool) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_primary = use_primary

    @property
    def active_name(self) -> str:
        """Name of the tracker currently in use."""
        active = self._primary if self._use_primary else self._fallback
        return getattr(active, "name", type(active).__name__)

    def toggle(self) -> str:
        """Switch tracker implementations; returns the new active name."""
        self._use_primary = not self._use_primary
        logger.info("tracker switched to %s", self.active_name)
        return self.active_name

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Delegate to the active tracker."""
        active = self._primary if self._use_primary else self._fallback
        return active.update(detections, timestamp_s, camera_id)
