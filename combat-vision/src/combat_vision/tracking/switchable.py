"""A tracker wrapper that can hot-swap implementations at runtime.

Powers the live on/off toggle for the supervision (ByteTrack) tracker: the
overlay's ``t`` key calls :meth:`toggle`, flipping between the primary and
fallback tracker without restarting the pipeline. Switching resets track
state, so identities may relabel on the frames right after a toggle — which
:meth:`SwitchableTracker.toggle` reports through ``consume_relabeled`` so
callers holding per-label state can reset it.
"""

from __future__ import annotations

import logging

from combat_vision.events.types import FighterId, PersonDetection, TrackedPose
from combat_vision.tracking.base import Tracker

logger = logging.getLogger(__name__)


class SwitchableTracker:
    """Delegates to one of two trackers, switchable while running."""

    def __init__(self, primary: Tracker, fallback: Tracker, use_primary: bool) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_primary = use_primary
        self._live_labels: set[FighterId] = set()
        self._pending_relabeled: set[FighterId] = set()

    @property
    def _active(self) -> Tracker:
        """The tracker currently receiving frames."""
        return self._primary if self._use_primary else self._fallback

    @property
    def active_name(self) -> str:
        """Name of the tracker currently in use."""
        return getattr(self._active, "name", type(self._active).__name__)

    def toggle(self) -> str:
        """Switch tracker implementations; returns the new active name.

        The switch is *itself* a relabel of every live label. Each tracker
        builds its label->person mapping independently, so the person holding
        "A" under the outgoing tracker is not necessarily the one holding "A"
        under the incoming one — and the incoming tracker has no way to know a
        handover happened, so it will report nothing. Without recording the
        switch here, per-label state (notably ``PoseSmoother``'s One-Euro
        filters) would carry the departed fighter's position/velocity into the
        new one's first frames and manufacture a phantom speed spike.

        Both trackers' own pending sets are drained into ours at the same
        time: the outgoing tracker's would otherwise be stranded until it next
        becomes active, surfacing a stale relabel at the wrong moment.
        """
        self._use_primary = not self._use_primary
        self._pending_relabeled |= self._live_labels
        self._pending_relabeled |= self._primary.consume_relabeled()
        self._pending_relabeled |= self._fallback.consume_relabeled()
        logger.info("tracker switched to %s", self.active_name)
        return self.active_name

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Delegate to the active tracker, remembering which labels are in play."""
        tracked = self._active.update(detections, timestamp_s, camera_id)
        self._live_labels.update(pose.fighter_id for pose in tracked)
        return tracked

    def consume_relabeled(self) -> frozenset[FighterId]:
        """Relabels from the active tracker, plus any caused by a toggle."""
        labels = frozenset(self._pending_relabeled | self._active.consume_relabeled())
        self._pending_relabeled.clear()
        return labels
