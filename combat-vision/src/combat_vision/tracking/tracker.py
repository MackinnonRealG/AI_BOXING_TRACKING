"""Greedy centroid tracker with occlusion tolerance.

Design: detections are matched to existing tracks by nearest centroid inside
a gate (``max_match_distance``). A track that misses detections — e.g. during
a clinch or a crossing — survives for ``max_missed_frames`` before retiring,
so identities persist through brief occlusion. The first ``max_fighters``
tracks get the stable labels "A", "B", ... which are *never* reassigned
within a session.

Known limitation (documented, deliberate for v1): after a long full
occlusion, identities can swap. The fix is appearance-based re-ID
(jersey/skin histograms) — see the README roadmap.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, field

from combat_vision.events.types import FighterId, PersonDetection, TrackedPose


@dataclass
class _Track:
    """Internal mutable state for one tracked person."""

    fighter_id: FighterId
    centroid: tuple[float, float]
    missed_frames: int = 0
    last_detection: PersonDetection | None = field(default=None, repr=False)


class FighterTracker:
    """Assigns persistent fighter identities to per-frame detections."""

    def __init__(
        self,
        max_match_distance: float,
        max_missed_frames: int,
        max_fighters: int,
    ) -> None:
        self._max_match_distance = max_match_distance
        self._max_missed_frames = max_missed_frames
        self._max_fighters = max_fighters
        self._tracks: list[_Track] = []
        self._labels_used = 0

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Match detections to tracks and return identity-stamped poses."""
        unmatched = list(detections)
        results: list[TrackedPose] = []

        # Greedy nearest-neighbour: process tracks in label order so fighter A
        # wins contested detections deterministically.
        for track in self._tracks:
            best: PersonDetection | None = None
            best_dist = self._max_match_distance
            for det in unmatched:
                dist = _distance(track.centroid, det.pose.centroid())
                if dist < best_dist:
                    best, best_dist = det, dist
            if best is None:
                track.missed_frames += 1
                continue
            unmatched.remove(best)
            track.centroid = best.pose.centroid()
            track.missed_frames = 0
            track.last_detection = best
            results.append(
                TrackedPose(
                    fighter_id=track.fighter_id,
                    pose=best.pose,
                    timestamp_s=timestamp_s,
                    camera_id=camera_id,
                )
            )

        # Retire tracks lost for too long.
        self._tracks = [t for t in self._tracks if t.missed_frames <= self._max_missed_frames]

        # Spawn new tracks for leftover detections, up to the fighter cap.
        for det in unmatched:
            if self._labels_used >= self._max_fighters:
                break
            fighter_id = string.ascii_uppercase[self._labels_used]
            self._labels_used += 1
            self._tracks.append(_Track(fighter_id=fighter_id, centroid=det.pose.centroid()))
            results.append(
                TrackedPose(
                    fighter_id=fighter_id,
                    pose=det.pose,
                    timestamp_s=timestamp_s,
                    camera_id=camera_id,
                )
            )
        return results


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in normalized image coordinates."""
    return math.hypot(a[0] - b[0], a[1] - b[1])
