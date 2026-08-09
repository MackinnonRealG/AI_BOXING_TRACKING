"""Pose estimation behind a pluggable backend interface."""

from combat_vision.pose.base import PoseBackend
from combat_vision.pose.mediapipe_backend import MediaPipePoseBackend
from combat_vision.pose.yolo_backend import YoloV8PoseBackend

__all__ = ["MediaPipePoseBackend", "PoseBackend", "YoloV8PoseBackend"]
