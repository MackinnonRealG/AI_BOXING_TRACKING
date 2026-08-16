"""pipeline.py tests: a tracker relabel must reach every consumer.

``Tracker.consume_relabeled`` clears on read, so exactly one caller can drain
it. The pipeline is that caller; these tests pin that it both resets the
smoother *and* broadcasts, so sinks holding per-label state (the overlay) are
not silently left stale.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from combat_vision.capture.base import TimestampedFrame
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    FighterId,
    FighterRelabeledEvent,
    Keypoint,
    KeypointName,
    Pose,
    TrackedPose,
)
from combat_vision.pipeline import Pipeline


def _frame(index: int) -> TimestampedFrame:
    return TimestampedFrame(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp_s=index / 30,
        frame_index=index,
        camera_id="cam0",
    )


def _tracked(fighter_id: str, t: float) -> TrackedPose:
    return TrackedPose(
        fighter_id=fighter_id,
        pose=Pose(keypoints={KeypointName.NOSE: Keypoint(x=0.5, y=0.5)}),
        timestamp_s=t,
        camera_id="cam0",
    )


class _Source:
    def __init__(self, count: int) -> None:
        self._count = count
        self.closed = False

    def frames(self) -> Iterator[TimestampedFrame]:
        for i in range(self._count):
            yield _frame(i)

    def close(self) -> None:
        self.closed = True


class _PoseBackend:
    def detect(self, frame: TimestampedFrame) -> list:
        return []

    def close(self) -> None:
        pass


class _Tracker:
    """Reports a relabel of "A" on exactly one frame."""

    def __init__(self, relabel_on_frame: int) -> None:
        self._relabel_on = relabel_on_frame
        self._frame = -1
        self.consume_calls = 0

    def update(self, detections: list, timestamp_s: float, camera_id: str) -> list[TrackedPose]:
        self._frame += 1
        return [_tracked("A", timestamp_s)]

    def consume_relabeled(self) -> frozenset[FighterId]:
        self.consume_calls += 1
        return frozenset({"A"}) if self._frame == self._relabel_on else frozenset()


class _Smoother:
    def __init__(self) -> None:
        self.reset_calls: list[str] = []

    def smooth(self, tracked: TrackedPose) -> TrackedPose:
        return tracked

    def reset(self, fighter_id: FighterId) -> None:
        self.reset_calls.append(fighter_id)


def _run(relabel_on_frame: int, frames: int = 3) -> tuple[_Smoother, list]:
    bus = EventBus()
    seen: list[FighterRelabeledEvent] = []
    bus.subscribe(FighterRelabeledEvent, seen.append)
    smoother = _Smoother()
    Pipeline(
        source=_Source(frames),
        pose_backend=_PoseBackend(),
        tracker=_Tracker(relabel_on_frame),
        smoother=smoother,
        engines=[],
        bus=bus,
    ).run()
    return smoother, seen


def test_relabel_is_broadcast_on_the_bus() -> None:
    """Sinks with per-label state learn about a relabel they cannot poll for."""
    _, seen = _run(relabel_on_frame=1)

    assert len(seen) == 1
    assert seen[0].fighter_id == "A"


def test_relabel_still_resets_the_smoother() -> None:
    """Broadcasting must not replace the existing direct smoother reset."""
    smoother, _ = _run(relabel_on_frame=1)

    assert smoother.reset_calls == ["A"]


def test_no_relabel_means_no_event() -> None:
    """Steady tracking publishes nothing and resets nothing."""
    smoother, seen = _run(relabel_on_frame=-99)

    assert seen == []
    assert smoother.reset_calls == []
