"""YOLOv8-pose backend.

Natively multi-person, GPU-friendly — the intended production backend for
crowded frames. Emits 17 COCO keypoints (no heels/foot-index), so
foot-dependent engines degrade gracefully to ankle-only data, which the
canonical contract already allows. COCO does include eyes/ears, unlike the
old canonical set — they're mapped below alongside everything else COCO has.

Requires the optional dependency: ``pip install "combat-vision[yolo]"``
(pulls in ``ultralytics`` and PyTorch).
"""

from __future__ import annotations

from typing import Any

from combat_vision.capture.base import TimestampedFrame
from combat_vision.events.types import BBox, Keypoint, KeypointName, PersonDetection, Pose
from combat_vision.pose.base import PoseBackend

# COCO keypoint index -> canonical name.
_COCO_TO_CANONICAL: dict[int, KeypointName] = {
    0: KeypointName.NOSE,
    1: KeypointName.LEFT_EYE,
    2: KeypointName.RIGHT_EYE,
    3: KeypointName.LEFT_EAR,
    4: KeypointName.RIGHT_EAR,
    5: KeypointName.LEFT_SHOULDER,
    6: KeypointName.RIGHT_SHOULDER,
    7: KeypointName.LEFT_ELBOW,
    8: KeypointName.RIGHT_ELBOW,
    9: KeypointName.LEFT_WRIST,
    10: KeypointName.RIGHT_WRIST,
    11: KeypointName.LEFT_HIP,
    12: KeypointName.RIGHT_HIP,
    13: KeypointName.LEFT_KNEE,
    14: KeypointName.RIGHT_KNEE,
    15: KeypointName.LEFT_ANKLE,
    16: KeypointName.RIGHT_ANKLE,
}


class YoloV8PoseBackend(PoseBackend):
    """Multi-person pose estimation via ultralytics YOLOv8-pose."""

    def __init__(self, model_path: str = "yolov8n-pose.pt", max_people: int = 2) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover — depends on optional extra
            raise ImportError(
                "YOLOv8 backend needs the optional extra: pip install 'combat-vision[yolo]'"
            ) from exc
        self._model: Any = YOLO(model_path)
        self._max_people = max_people

    def detect(self, frame: TimestampedFrame) -> list[PersonDetection]:
        """Run YOLOv8-pose; one detection per person, best-scored first."""
        result = self._model.predict(frame.image, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            return []

        height, width = frame.image.shape[:2]
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = (
            result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        )
        box_scores = result.boxes.conf.cpu().numpy()

        order = box_scores.argsort()[::-1][: self._max_people]
        detections: list[PersonDetection] = []
        for person in order:
            canonical: dict[KeypointName, Keypoint] = {}
            for index, name in _COCO_TO_CANONICAL.items():
                x_px, y_px = keypoints_xy[person][index]
                if x_px == 0 and y_px == 0:  # ultralytics marks missing points as (0, 0)
                    continue
                confidence = (
                    float(keypoints_conf[person][index]) if keypoints_conf is not None else 1.0
                )
                canonical[name] = Keypoint(
                    x=float(x_px) / width, y=float(y_px) / height, visibility=confidence
                )
            if not canonical:
                continue
            xs = [k.x for k in canonical.values()]
            ys = [k.y for k in canonical.values()]
            detections.append(
                PersonDetection(
                    pose=Pose(keypoints=canonical),
                    bbox=BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys)),
                    score=float(box_scores[person]),
                )
            )
        return detections

    def close(self) -> None:
        """Nothing to release — ultralytics manages its own resources."""
