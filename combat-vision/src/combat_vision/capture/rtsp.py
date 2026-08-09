"""RTSP / IP-camera capture."""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from combat_vision.capture.base import CameraSource, TimestampedFrame


class RtspSource(CameraSource):
    """Frames from an RTSP or HTTP stream URL.

    Note: network streams can stall; the iterator simply ends on read failure.
    Reconnect/backoff policy is a deliberate future addition (see README
    roadmap) so v1 behavior stays predictable.
    """

    def __init__(self, url: str, camera_id: str = "rtsp0") -> None:
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open stream {url}")
        self._camera_id = camera_id

    def frames(self) -> Iterator[TimestampedFrame]:
        """Yield frames stamped with a monotonic clock relative to start."""
        start = time.monotonic()
        index = 0
        while self._cap.isOpened():
            ok, image = self._cap.read()
            if not ok:
                break
            yield TimestampedFrame(
                image=image,
                timestamp_s=time.monotonic() - start,
                frame_index=index,
                camera_id=self._camera_id,
            )
            index += 1

    def close(self) -> None:
        """Release the stream."""
        self._cap.release()
