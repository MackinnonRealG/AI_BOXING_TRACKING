"""Recorded-footage capture for review mode."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from combat_vision.capture.base import CameraSource, TimestampedFrame


class VideoFileSource(CameraSource):
    """Frames from a video file, timestamped by presentation time.

    Using PTS (not wall clock) means review mode measures the *recorded*
    motion faithfully regardless of how fast we can process it.
    """

    def __init__(self, path: str | Path, camera_id: str = "file0") -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file {self._path}")
        self._camera_id = camera_id

    @property
    def fps(self) -> float:
        """Native frame rate of the file."""
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def frames(self) -> Iterator[TimestampedFrame]:
        """Yield every frame with its presentation timestamp."""
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = pts_ms / 1000.0 if pts_ms > 0 else index / self.fps
            yield TimestampedFrame(
                image=image,
                timestamp_s=timestamp_s,
                frame_index=index,
                camera_id=self._camera_id,
            )
            index += 1

    def close(self) -> None:
        """Release the file handle."""
        self._cap.release()
