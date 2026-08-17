"""Pixel→metre calibration with graceful uncalibrated fallback."""

from combat_vision.calibration.calibrator import Calibration, multi_camera_fusion_stub
from combat_vision.calibration.triangulation import CameraParams, project, triangulate

__all__ = [
    "Calibration",
    "CameraParams",
    "multi_camera_fusion_stub",
    "project",
    "triangulate",
]
