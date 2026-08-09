"""Camera abstraction: webcam, video file, and RTSP sources behind one interface."""

from combat_vision.capture.base import CameraSource, TimestampedFrame
from combat_vision.capture.rtsp import RtspSource
from combat_vision.capture.video_file import VideoFileSource
from combat_vision.capture.webcam import WebcamSource

__all__ = ["CameraSource", "RtspSource", "TimestampedFrame", "VideoFileSource", "WebcamSource"]
