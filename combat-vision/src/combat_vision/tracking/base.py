"""The Tracker protocol every tracking implementation satisfies."""

from __future__ import annotations

from typing import Protocol

from combat_vision.events.types import FighterId, PersonDetection, TrackedPose


class Tracker(Protocol):
    """Assigns persistent fighter identities to per-frame detections."""

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Match detections to identities and return labelled poses."""
        ...

    def consume_relabeled(self) -> frozenset[FighterId]:
        """Labels reassigned to a different underlying track since the last
        call to this method; clears the set on return.

        A relabel means the physical person now holding a label is *not* the
        one who held it moments ago (e.g. ByteTrack recycling "A" onto a
        fresh track id). Callers with per-label state that assumes temporal
        continuity — e.g. :class:`~combat_vision.filtering.smoother.PoseSmoother` —
        must reset that state for every label returned here.
        """
        ...
