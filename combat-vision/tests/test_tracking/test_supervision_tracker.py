"""Tests for the supervision (ByteTrack) tracker and the runtime toggle."""

from __future__ import annotations

from combat_vision.events.types import BBox, Keypoint, KeypointName, PersonDetection, Pose
from combat_vision.tracking import FighterTracker, SupervisionTracker, SwitchableTracker

_FRAME = (1280, 720)
_FPS = 30


def _detection(cx: float, cy: float = 0.5, size: float = 0.12) -> PersonDetection:
    """A person detection centered at (cx, cy) in normalized coordinates."""
    pose = Pose(
        keypoints={
            KeypointName.NOSE: Keypoint(x=cx, y=cy - size),
            KeypointName.LEFT_HIP: Keypoint(x=cx + 0.02, y=cy),
            KeypointName.RIGHT_HIP: Keypoint(x=cx - 0.02, y=cy),
        }
    )
    return PersonDetection(
        pose=pose,
        bbox=BBox(x_min=cx - size, y_min=cy - 2 * size, x_max=cx + size, y_max=cy + 2 * size),
        score=0.9,
    )


def _make_supervision_tracker() -> SupervisionTracker:
    return SupervisionTracker(
        frame_width_px=_FRAME[0],
        frame_height_px=_FRAME[1],
        max_fighters=2,
        max_missed_frames=15,
        frame_rate=_FPS,
    )


def test_two_fighters_get_stable_labels() -> None:
    """Two people drifting toward each other keep their A/B labels."""
    tracker = _make_supervision_tracker()
    labels_by_side: list[tuple[str, str]] = []
    for i in range(_FPS):
        t = i / _FPS
        left = _detection(0.25 + 0.003 * i)
        right = _detection(0.75 - 0.003 * i)
        tracked = tracker.update([left, right], t, "cam0")
        if len(tracked) == 2:
            by_x = sorted(tracked, key=lambda p: p.pose.centroid()[0])
            labels_by_side.append((by_x[0].fighter_id, by_x[1].fighter_id))

    assert len(labels_by_side) >= _FPS - 5  # ByteTrack activates within a few frames
    assert len(set(labels_by_side)) == 1  # the same person keeps the same label
    assert set(labels_by_side[0]) == {"A", "B"}


def test_identity_survives_brief_occlusion() -> None:
    """A fighter vanishing for a few frames keeps their label on return."""
    tracker = _make_supervision_tracker()
    label_before = label_after = None
    for i in range(60):
        t = i / _FPS
        detections = [_detection(0.30)]
        if not 20 <= i < 26:  # fighter B disappears for 6 frames
            detections.append(_detection(0.70))
        tracked = tracker.update(detections, t, "cam0")
        for pose in tracked:
            if pose.pose.centroid()[0] > 0.5:
                if i < 20:
                    label_before = pose.fighter_id
                elif i >= 26:
                    label_after = pose.fighter_id

    assert label_before is not None and label_after is not None
    assert label_before == label_after


def test_empty_frames_are_handled() -> None:
    """Frames with no detections must not crash and return nothing."""
    tracker = _make_supervision_tracker()
    for i in range(10):
        assert tracker.update([], i / _FPS, "cam0") == []


def test_switchable_tracker_toggles_and_keeps_working() -> None:
    """The on/off switch flips backends and both keep producing output."""
    switchable = SwitchableTracker(
        primary=_make_supervision_tracker(),
        fallback=FighterTracker(max_match_distance=0.25, max_missed_frames=15, max_fighters=2),
        use_primary=True,
    )
    assert switchable.active_name == "supervision"

    for i in range(10):  # warm up ByteTrack past activation
        supervision_out = switchable.update([_detection(0.4)], i / _FPS, "cam0")
    assert len(supervision_out) == 1

    assert switchable.toggle() == "centroid"
    centroid_out = switchable.update([_detection(0.4)], 11 / _FPS, "cam0")
    assert len(centroid_out) == 1

    assert switchable.toggle() == "supervision"
