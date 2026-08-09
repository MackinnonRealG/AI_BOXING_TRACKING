"""The CameraSource interface and its frame contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TimestampedFrame:
    """One captured frame with provenance.

    ``image`` is BGR uint8 (OpenCV convention). This is the *only* contract in
    the system that carries a raw array; it exists solely between capture and
    the pose backend / renderer.
    """

    image: np.ndarray
    timestamp_s: float
    """Seconds since the source started. Video files use PTS; live sources
    use a monotonic clock, so timestamps are comparable within one session."""
    frame_index: int
    camera_id: str = "cam0"


class CameraSource(ABC):
    """A stream of timestamped frames.

    Implementations: :class:`~combat_vision.capture.webcam.WebcamSource`,
    :class:`~combat_vision.capture.video_file.VideoFileSource`,
    :class:`~combat_vision.capture.rtsp.RtspSource`.

    Multi-camera note: the pipeline is structured so N sources can run side by
    side, each tagging frames with its ``camera_id``. Cross-camera calibration
    and fusion are a future module — see
    :func:`combat_vision.calibration.multi_camera_fusion_stub`.
    """

    @abstractmethod
    def frames(self) -> Iterator[TimestampedFrame]:
        """Yield frames until the stream ends or :meth:`close` is called."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture device."""

    def __enter__(self) -> CameraSource:
        """Support ``with source:`` usage."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the device on context exit."""
        self.close()
