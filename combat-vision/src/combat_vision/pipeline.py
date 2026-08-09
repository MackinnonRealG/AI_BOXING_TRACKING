"""The single processing pipeline shared by live and review modes.

    capture → pose → tracking → smoothing → engines → bus → sinks

Mode only changes the *source* (webcam vs file) and the *sink* (overlay
window vs report builder). Everything in between is identical, so review
results are directly comparable to live results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from combat_vision.capture.base import CameraSource, TimestampedFrame
from combat_vision.engines.base import MetricsEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import TrackedPose
from combat_vision.filtering.smoother import PoseSmoother
from combat_vision.pose.base import PoseBackend
from combat_vision.tracking.base import Tracker

logger = logging.getLogger(__name__)

FrameSink = Callable[[TimestampedFrame, list[TrackedPose]], bool]
"""Called once per processed frame; return False to stop the pipeline."""


class Pipeline:
    """Wires the stages together and drives them frame by frame."""

    def __init__(
        self,
        source: CameraSource,
        pose_backend: PoseBackend,
        tracker: Tracker,
        smoother: PoseSmoother,
        engines: Sequence[MetricsEngine],
        bus: EventBus,
        frame_sink: FrameSink | None = None,
    ) -> None:
        self._source = source
        self._pose_backend = pose_backend
        self._tracker = tracker
        self._smoother = smoother
        self._engines = engines
        self._bus = bus
        self._frame_sink = frame_sink

    @property
    def tracker(self) -> Tracker:
        """The tracker in use (a SwitchableTracker in the default wiring)."""
        return self._tracker

    def set_frame_sink(self, sink: FrameSink) -> None:
        """Attach/replace the per-frame sink (used when the sink — e.g. the
        overlay — needs the pipeline's bus and must be built afterwards)."""
        self._frame_sink = sink

    def run(self) -> float:
        """Process the stream to completion. Returns stream duration (s)."""
        last_timestamp = 0.0
        frames = 0
        wall_start = time.monotonic()
        try:
            for frame in self._source.frames():
                last_timestamp = frame.timestamp_s
                frames += 1

                detections = self._pose_backend.detect(frame)
                tracked = self._tracker.update(detections, frame.timestamp_s, frame.camera_id)
                smoothed = [self._smoother.smooth(t) for t in tracked]

                for pose in smoothed:
                    for engine in self._engines:
                        engine.process(pose)

                if self._frame_sink is not None and not self._frame_sink(frame, smoothed):
                    break
        finally:
            for engine in self._engines:
                engine.finish()
            self._pose_backend.close()
            self._source.close()

        wall = time.monotonic() - wall_start
        if wall > 0 and frames:
            logger.info("processed %d frames at %.1f FPS", frames, frames / wall)
        return last_timestamp
