"""MediaPipe PoseLandmarker backend — the default estimator.

Uses the modern MediaPipe *tasks* API (mediapipe >= 1.0), which natively
detects multiple people (``num_poses``), so both fighters come from a single
inference pass. The landmarker's ``.task`` model file is downloaded to a
local cache on first use.
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import cv2

from combat_vision.capture.base import TimestampedFrame
from combat_vision.events.types import BBox, Keypoint, KeypointName, PersonDetection, Pose
from combat_vision.pose.base import PoseBackend

logger = logging.getLogger(__name__)

# MediaPipe landmark index -> canonical name (subset we consume).
_MP_TO_CANONICAL: dict[int, KeypointName] = {
    0: KeypointName.NOSE,
    2: KeypointName.LEFT_EYE,
    5: KeypointName.RIGHT_EYE,
    7: KeypointName.LEFT_EAR,
    8: KeypointName.RIGHT_EAR,
    11: KeypointName.LEFT_SHOULDER,
    12: KeypointName.RIGHT_SHOULDER,
    13: KeypointName.LEFT_ELBOW,
    14: KeypointName.RIGHT_ELBOW,
    15: KeypointName.LEFT_WRIST,
    16: KeypointName.RIGHT_WRIST,
    23: KeypointName.LEFT_HIP,
    24: KeypointName.RIGHT_HIP,
    25: KeypointName.LEFT_KNEE,
    26: KeypointName.RIGHT_KNEE,
    27: KeypointName.LEFT_ANKLE,
    28: KeypointName.RIGHT_ANKLE,
    29: KeypointName.LEFT_HEEL,
    30: KeypointName.RIGHT_HEEL,
    31: KeypointName.LEFT_FOOT_INDEX,
    32: KeypointName.RIGHT_FOOT_INDEX,
}

_MODEL_URLS: dict[str, str] = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}


def _model_cache_dir() -> Path:
    """Local cache for downloaded ``.task`` model files."""
    return Path.home() / ".cache" / "combat_vision"


def ensure_model(variant: str) -> Path:
    """Return the local path of the landmarker model, downloading if absent.

    Downloads to a uniquely-named ``.part`` sibling and renames it into place
    only after a full, successful transfer. A network failure/interruption
    mid-download otherwise leaves a truncated file sitting at the final path;
    since the only freshness check is ``path.exists()``, every future run
    would silently hand that corrupt file to PoseLandmarker instead of
    retrying. The temp name is unique per call (not just per variant) so two
    concurrent invocations downloading the same variant never share one
    `.part` file and race each other's rename/cleanup.
    """
    if variant not in _MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}; choose from {sorted(_MODEL_URLS)}")
    path = _model_cache_dir() / f"pose_landmarker_{variant}.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".part"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        logger.info("downloading pose model %s -> %s", variant, path)
        try:
            urllib.request.urlretrieve(_MODEL_URLS[variant], tmp_path)  # noqa: S310 — fixed https host
            tmp_path.replace(path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return path


class MediaPipePoseBackend(PoseBackend):
    """Multi-person pose estimation via MediaPipe PoseLandmarker."""

    def __init__(
        self,
        model_variant: str,
        num_poses: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        # Local imports: keep the package importable without mediapipe installed.
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_variant))),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker: Any = PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame: TimestampedFrame) -> list[PersonDetection]:
        """Run the landmarker on one frame; one detection per visible person."""
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        timestamp_ms = max(int(frame.timestamp_s * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        detections: list[PersonDetection] = []
        for landmarks in result.pose_landmarks:
            keypoints: dict[KeypointName, Keypoint] = {}
            for index, name in _MP_TO_CANONICAL.items():
                lm = landmarks[index]
                visibility = lm.visibility if lm.visibility is not None else 1.0
                keypoints[name] = Keypoint(x=lm.x, y=lm.y, z=lm.z, visibility=visibility)
            xs = [k.x for k in keypoints.values()]
            ys = [k.y for k in keypoints.values()]
            detections.append(
                PersonDetection(
                    pose=Pose(keypoints=keypoints),
                    bbox=BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys)),
                    score=sum(k.visibility for k in keypoints.values()) / len(keypoints),
                )
            )
        return detections

    def close(self) -> None:
        """Release the landmarker."""
        self._landmarker.close()
