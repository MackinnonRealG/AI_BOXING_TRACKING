"""WebcamSource construction tests — no real camera hardware required.

cv2.VideoCapture is mocked out entirely: these tests only exercise the
validation and wiring in __init__, not actual device I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from combat_vision.capture.webcam import WebcamSource


def _opened_capture() -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = True
    return cap


@pytest.mark.parametrize("width,height", [(0, 720), (1280, 0), (-1, 720), (1280, -1)])
def test_nonpositive_size_is_rejected_before_opening_the_device(width: int, height: int) -> None:
    """A misconfigured zero/negative size fails loudly at construction.

    Previously this passed silently through and produced a hard cv2.error
    deep inside the frame loop the first time a captured frame needed
    downscaling to a zero-width/height target.
    """
    with patch("cv2.VideoCapture") as video_capture:
        with pytest.raises(ValueError, match="positive"):
            WebcamSource(index=0, width=width, height=height)
        video_capture.assert_not_called()  # must fail before even touching the device


def test_positive_size_opens_normally() -> None:
    with patch("cv2.VideoCapture", return_value=_opened_capture()) as video_capture:
        source = WebcamSource(index=0, width=1280, height=720)
        video_capture.assert_called_once_with(0)
        source.close()
