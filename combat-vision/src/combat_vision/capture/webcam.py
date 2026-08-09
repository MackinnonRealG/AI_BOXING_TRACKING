"""Live webcam capture."""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from combat_vision.capture.base import CameraSource, TimestampedFrame


class WebcamSource(CameraSource):
    """Frames from a local camera by OpenCV device index."""

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        camera_id: str = "cam0",
    ) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
        """Release the camera device."""
        self._cap.release()
