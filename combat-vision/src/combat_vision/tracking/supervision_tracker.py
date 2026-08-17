"""ByteTrack-based tracker built on the ``supervision`` library.

ByteTrack (Zhang et al., 2022) associates detections across frames using
both high- and low-confidence boxes with a Kalman-filtered motion model —
markedly more robust through clinches and crossings than the greedy
centroid tracker. This class adapts it to Combat Vision's contracts:
detections in, stable Fighter A/B labels out.

ByteTrack's internal track ids grow forever (a re-acquired fighter gets a
fresh id), so a small label pool maps track ids onto the stable labels:
the first ``max_fighters`` ids get "A", "B", ...; when a labelled id has
not been seen for ``max_missed_frames`` frames its label may be recycled
to a new id (e.g. the fighter re-entering the frame).

When more than one label is eligible for recycling at once (both fighters
were lost, then both reappear), picking the first eligible slot regardless
of who is actually who is a coin flip — motion alone can't tell two people
apart after a long gap. Each slot remembers the last
:mod:`~combat_vision.pose.appearance` descriptor seen under it, and
recycling prefers whichever eligible slot's remembered appearance is
closest to the reappearing detection's, falling back to slot order only
when no descriptor is available on either side.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from combat_vision.events.types import FighterId, PersonDetection, TrackedPose
from combat_vision.pose import appearance

if TYPE_CHECKING:
    import supervision as sv


@dataclass
class _LabelSlot:
    """One stable fighter label and the ByteTrack id currently holding it."""

    label: FighterId
    track_id: int | None = None
    last_seen_frame: int = -1
    last_appearance: tuple[float, ...] | None = None


class SupervisionTracker:
    """Fighter tracking via supervision's ByteTrack."""

    name = "supervision"

    def __init__(
        self,
        frame_width_px: int,
        frame_height_px: int,
        max_fighters: int,
        max_missed_frames: int,
        frame_rate: int,
    ) -> None:
        import supervision as sv  # local import keeps startup cheap when unused

        self._sv = sv
        self._tracker = sv.ByteTrack(
            lost_track_buffer=max_missed_frames,
            frame_rate=frame_rate,
            # A track must survive a few frames before earning a fighter
            # label — stops startup flicker from claiming "A".
            minimum_consecutive_frames=3,
        )
        self._frame_size = (frame_width_px, frame_height_px)
        self._max_missed_frames = max_missed_frames
        self._slots = [
            _LabelSlot(label=string.ascii_uppercase[i]) for i in range(max_fighters)
        ]
        self._frame_index = 0
        self._relabeled: set[FighterId] = set()

    def update(
        self, detections: list[PersonDetection], timestamp_s: float, camera_id: str
    ) -> list[TrackedPose]:
        """Run ByteTrack on this frame's detections and label the results."""
        self._frame_index += 1
        sv_detections = self._to_sv(detections)
        tracked = self._tracker.update_with_detections(sv_detections)

        results: list[TrackedPose] = []
        if tracked.tracker_id is None:
            return results
        for detection_index, track_id in zip(
            tracked.data.get("detection_index", range(len(tracked))),
            tracked.tracker_id,
            strict=True,
        ):
            det = detections[int(detection_index)]
            label = self._label_for(int(track_id), det.appearance)
            if label is None:
                continue
            results.append(
                TrackedPose(
                    fighter_id=label,
                    pose=det.pose,
                    timestamp_s=timestamp_s,
                    camera_id=camera_id,
                )
            )
        results.sort(key=lambda t: t.fighter_id)
        return results

    def _to_sv(self, detections: list[PersonDetection]) -> sv.Detections:
        """Convert canonical detections to a supervision Detections batch."""
        sv = self._sv
        if not detections:
            return sv.Detections.empty()
        w, h = self._frame_size
        xyxy = np.array(
            [
                [d.bbox.x_min * w, d.bbox.y_min * h, d.bbox.x_max * w, d.bbox.y_max * h]
                for d in detections
            ],
            dtype=np.float32,
        )
        result = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([d.score for d in detections], dtype=np.float32),
            class_id=np.zeros(len(detections), dtype=int),
        )
        # Remember which input detection each box came from: ByteTrack may
        # drop or reorder boxes, and the pose must follow its detection.
        result.data["detection_index"] = np.arange(len(detections))
        return result

    def _label_for(
        self, track_id: int, det_appearance: tuple[float, ...] | None
    ) -> FighterId | None:
        """Stable label for a ByteTrack id, recycling long-lost labels."""
        for slot in self._slots:  # already labelled?
            if slot.track_id == track_id:
                slot.last_seen_frame = self._frame_index
                if det_appearance is not None:
                    slot.last_appearance = det_appearance
                return slot.label
        for slot in self._slots:  # a label never used yet?
            if slot.track_id is None:
                slot.track_id = track_id
                slot.last_seen_frame = self._frame_index
                slot.last_appearance = det_appearance
                return slot.label

        recyclable = [
            slot
            for slot in self._slots
            if self._frame_index - slot.last_seen_frame > self._max_missed_frames
        ]
        if not recyclable:
            return None  # more people than fighter slots — ignore extras
        slot = self._best_match(recyclable, det_appearance)
        slot.track_id = track_id
        slot.last_seen_frame = self._frame_index
        slot.last_appearance = det_appearance
        # This label now points at a *different* physical person — any
        # per-label state elsewhere (e.g. smoothing filters) is stale and
        # must be reset.
        self._relabeled.add(slot.label)
        return slot.label

    def _best_match(
        self, candidates: list[_LabelSlot], det_appearance: tuple[float, ...] | None
    ) -> _LabelSlot:
        """The recyclable slot whose remembered appearance is closest to
        ``det_appearance``, or the first candidate if no descriptors are
        available to compare (falls back to the old slot-order behavior)."""
        scored = [
            (dist, slot)
            for slot in candidates
            if (dist := appearance.distance(slot.last_appearance, det_appearance)) is not None
        ]
        if not scored:
            return candidates[0]
        return min(scored, key=lambda pair: pair[0])[1]

    def consume_relabeled(self) -> frozenset[FighterId]:
        """Labels recycled onto a different track id since the last call."""
        labels = frozenset(self._relabeled)
        self._relabeled.clear()
        return labels
