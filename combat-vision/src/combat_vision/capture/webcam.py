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
        if width <= 0 or height <= 0:
            # Caught here, at construction, rather than left to surface as an
            # opaque cv2.error deep in the frame loop the first time a
            # captured frame needs downscaling to a zero/negative size.
            raise ValueError(f"capture width/height must be positive, got {width}x{height}")
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._size = (width, height)
        self._camera_id = camera_id

    def frames(self) -> Iterator[TimestampedFrame]:
        """Yield frames stamped with a monotonic clock relative to start.

        Some cameras (notably macOS FaceTime cameras) ignore the requested
        capture size and deliver native-resolution frames; those are scaled
        down here so pose inference always runs at the configured size —
        processing 1080p instead of 720p roughly halves the frame rate.
        """
        start = time.monotonic()
        index = 0
        while self._cap.isOpened():
            ok, image = self._cap.read()
            if not ok:
                break
            if image.shape[1] > self._size[0]:
                scale = self._size[0] / image.shape[1]  # fit width, keep aspect
                new_size = (self._size[0], round(image.shape[0] * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
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
