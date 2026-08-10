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


def test_recycled_label_is_reported_once() -> None:
    """A label recycled onto a new physical person is reported exactly once.

    Fighter B disappears for longer than ``max_missed_frames``, so ByteTrack
    hands the reappearing detection a brand-new internal track id and the
    label pool recycles "B" onto it — that transition must surface through
    ``consume_relabeled`` so per-label state (e.g. smoothing filters) can be
    reset by the caller.
    """
    tracker = _make_supervision_tracker()  # max_missed_frames=15
    recycled_at: list[int] = []
    for i in range(80):
        t = i / _FPS
        detections = [_detection(0.30)]
        if i < 20 or i >= 45:  # B absent for 25 frames > max_missed_frames
            detections.append(_detection(0.70))
        tracker.update(detections, t, "cam0")
        relabeled = tracker.consume_relabeled()
        if relabeled:
            recycled_at.append(i)
            assert relabeled == frozenset({"B"})

    assert len(recycled_at) == 1, "expected exactly one recycle event, not repeated reports"


def test_no_recycle_reported_without_a_gap() -> None:
    """Two continuously-tracked fighters must never report a recycled label."""
    tracker = _make_supervision_tracker()
    for i in range(_FPS):
        t = i / _FPS
        left = _detection(0.25 + 0.003 * i)
        right = _detection(0.75 - 0.003 * i)
        tracker.update([left, right], t, "cam0")
        assert tracker.consume_relabeled() == frozenset()


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


def _make_switchable(use_primary: bool = True) -> SwitchableTracker:
    return SwitchableTracker(
        primary=_make_supervision_tracker(),
        fallback=FighterTracker(max_match_distance=0.25, max_missed_frames=15, max_fighters=2),
        use_primary=use_primary,
    )


def test_toggle_reports_live_labels_as_relabeled() -> None:
    """Switching backends must report every live label as relabeled.

    Each tracker assigns labels independently, so after a switch "A" may be a
    different physical person — but the incoming tracker cannot know a handover
    happened and reports nothing on its own. If the switch is not reported, the
    smoother keeps the departed fighter's filter state and the new person's
    first frames produce a phantom speed spike.
    """
    switchable = _make_switchable()
    for i in range(10):  # warm up ByteTrack so labels are actually in play
        switchable.update([_detection(0.3), _detection(0.7)], i / _FPS, "cam0")
    switchable.consume_relabeled()  # drain warm-up churn

    switchable.toggle()

    assert switchable.consume_relabeled() == frozenset({"A", "B"})


def test_toggle_relabels_are_reported_once() -> None:
    """The toggle-induced relabel set is cleared after it is consumed."""
    switchable = _make_switchable()
    for i in range(10):
        switchable.update([_detection(0.4)], i / _FPS, "cam0")
    switchable.consume_relabeled()

    switchable.toggle()
    assert switchable.consume_relabeled() != frozenset()
    assert switchable.consume_relabeled() == frozenset()


def test_no_toggle_means_no_relabel() -> None:
    """Steady tracking through the wrapper reports nothing, as before."""
    switchable = _make_switchable()
    for i in range(_FPS):
        left = _detection(0.25 + 0.003 * i)
        right = _detection(0.75 - 0.003 * i)
        switchable.update([left, right], i / _FPS, "cam0")
        assert switchable.consume_relabeled() == frozenset()
