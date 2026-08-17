"""Adversarial/degraded-condition tests — pinning behavior the happy-path
fixtures never exercise.

Every other test file in this suite feeds engines complete, fully-visible,
single-fighter pose data — exactly the condition real footage will *not*
reliably provide (occlusion, low-confidence detections, two people crossing
paths). These tests don't assert an "ideal" response to degraded input;
they document what the system *actually* does today, so a first real-camera
test has a known baseline to compare against, and so future changes don't
silently regress this behavior. Where the documented behavior is a real
limitation rather than a deliberate design choice, the test says so.
"""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.engines.guard import GuardEngine
from combat_vision.engines.speed import SpeedEngine
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    BBox,
    GuardStateEvent,
    Keypoint,
    KeypointName,
    Limb,
    PersonDetection,
    Pose,
    SpeedPeakEvent,
    TrackedPose,
)
from combat_vision.sports import get_profile
from combat_vision.tracking import SupervisionTracker
from combat_vision.utils.config import GuardConfig, SpeedEngineConfig
from tests.conftest import calibration_from_meta

_FPS = 60


# -- speed engine: strokes lost or corrupted by detection dropout ----------


def test_a_punch_entirely_inside_a_dropout_window_is_silently_missed(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """A real, documented limitation, not a bug: if pose tracking loses the
    wrist for the whole duration of a punch (a clinch, a crossing limb, a
    motion-blurred frame the pose model can't resolve), there is no data to
    detect a stroke from at all. The punch is not misclassified — it simply
    never happened as far as the pipeline is concerned.

    This matters for the checklist accompanying this test: a real sparring
    session is exactly where this happens most (clinches, exchanges at
    close range), so a fighter should not assume a session's punch count is
    a complete count.
    """
    poses, meta = jab_sequence
    # Drop every frame across the punch's extension/hold/retract span
    # (~t=0.45 to ~t=1.15), keeping only the idle frames before and after.
    surviving = [p for p in poses if p.timestamp_s < 0.45 or p.timestamp_s > 1.15]
    assert len(surviving) < len(poses)  # sanity: we actually removed the punch

    bus = EventBus()
    events: list[SpeedPeakEvent] = []
    bus.subscribe(SpeedPeakEvent, events.append)
    engine = SpeedEngine(
        bus, get_profile("boxing"), calibration_from_meta(meta), SpeedEngineConfig()
    )
    for pose in surviving:
        engine.process(pose)
    engine.finish()

    assert events == []


def test_a_brief_mid_stroke_gap_does_not_manufacture_a_speed_spike(
    jab_sequence: tuple[list[TrackedPose], dict],
) -> None:
    """A short dropout *during* a stroke (not the whole thing) must not
    average out into a bogus, wildly inflated speed reading — the observed
    peak should stay in the neighborhood of the fixture's true 6.0 m/s peak,
    not spike to some multiple of it because a gap made one frame-to-frame
    distance look larger than it is.
    """
    poses, meta = jab_sequence
    # Remove ~0.1s (6 frames) right around the peak, leaving the rest of the
    # stroke intact.
    surviving = [p for p in poses if not (0.70 <= p.timestamp_s <= 0.80)]
    assert len(surviving) < len(poses)

    bus = EventBus()
    events: list[SpeedPeakEvent] = []
    bus.subscribe(SpeedPeakEvent, events.append)
    engine = SpeedEngine(
        bus, get_profile("boxing"), calibration_from_meta(meta), SpeedEngineConfig()
    )
    for pose in surviving:
        engine.process(pose)
    engine.finish()

    assert len(events) == 1
    # True peak is 6.0 m/s; a gap-induced artifact would blow well past this.
    assert events[0].peak_speed < 15.0


# -- guard engine: low-confidence keypoints are currently trusted fully ----


def _guard_pose(wrist_y: float, visibility: float) -> Pose:
    return Pose(
        keypoints={
            KeypointName.NOSE: Keypoint(x=0.50, y=0.20, visibility=visibility),
            KeypointName.LEFT_SHOULDER: Keypoint(x=0.55, y=0.35, visibility=visibility),
            KeypointName.RIGHT_SHOULDER: Keypoint(x=0.45, y=0.35, visibility=visibility),
            KeypointName.LEFT_WRIST: Keypoint(x=0.55, y=wrist_y, visibility=visibility),
            KeypointName.RIGHT_WRIST: Keypoint(x=0.45, y=0.25, visibility=visibility),
        }
    )


def test_low_visibility_keypoints_are_currently_trusted_the_same_as_high_confidence() -> None:
    """Documents a real gap, not a design choice: no metrics engine reads
    the ``visibility`` field at all (only the overlay does, for what to
    draw). A guard-drop detected from a keypoint MediaPipe is only 15%
    confident about triggers exactly the same live cue as one it is 99%
    confident about.

    This is intentionally left as-is pending real-footage evidence of
    whether it actually causes false positives in practice (per the
    council's validate-before-optimizing guidance) — this test exists so
    that if/when visibility-weighting is added, its behavior change shows
    up here instead of silently.
    """
    bus = EventBus()
    events: list[GuardStateEvent] = []
    bus.subscribe(GuardStateEvent, events.append)
    engine = GuardEngine(bus, get_profile("boxing"), Calibration(None, 1280, 720), GuardConfig())

    # Left wrist held below the guard line throughout, at very low confidence.
    for i in range(2 * _FPS):
        pose = _guard_pose(wrist_y=0.45, visibility=0.15)
        engine.process(TrackedPose(fighter_id="A", pose=pose, timestamp_s=i / _FPS))

    drops = [e for e in events if e.hand == Limb.LEFT_HAND and not e.guard_up]
    assert len(drops) == 1  # fires identically to a high-confidence reading


# -- tracker: flickering detections and near-clinch crossings --------------


def _detection(cx: float, cy: float = 0.5, size: float = 0.12) -> PersonDetection:
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


def test_flickering_detection_within_tolerance_does_not_relabel() -> None:
    """A fighter whose detection drops in and out every other frame (a
    realistic symptom of a marginal pose-confidence threshold, not a clean
    disappearance) must not be treated as repeatedly lost and reacquired,
    as long as the flicker stays within max_missed_frames.
    """
    tracker = SupervisionTracker(
        frame_width_px=1280, frame_height_px=720, max_fighters=2, max_missed_frames=15,
        frame_rate=_FPS,
    )
    label_before = label_after = None
    for i in range(60):
        t = i / _FPS
        detections = [_detection(0.30)]
        # Fighter B flickers: present on even frames only, from frame 20-40.
        if not (20 <= i < 40 and i % 2 == 1):
            detections.append(_detection(0.70))
        tracked = tracker.update(detections, t, "cam0")
        for pose in tracked:
            if pose.pose.centroid()[0] > 0.5:
                if i < 20:
                    label_before = pose.fighter_id
                elif i >= 40:
                    label_after = pose.fighter_id

    assert label_before is not None and label_after is not None
    assert label_before == label_after
    assert tracker.consume_relabeled() == frozenset()


def test_close_bbox_overlap_without_full_occlusion_keeps_labels_stable() -> None:
    """Two fighters whose bounding boxes overlap heavily (a clinch, a close
    exchange) but who both remain at least partially detected must not swap
    labels — this is exactly the moment identity stability matters most for
    per-fighter fault attribution.
    """
    tracker = SupervisionTracker(
        frame_width_px=1280, frame_height_px=720, max_fighters=2, max_missed_frames=15,
        frame_rate=_FPS,
    )
    labels_by_side: list[tuple[str, str]] = []
    for i in range(90):
        t = i / _FPS
        # Both fighters converge to near-overlapping centers, then separate.
        progress = min(i / 45, 1.0) if i < 45 else max(1.0 - (i - 45) / 45, 0.0)
        offset = 0.25 * (1.0 - progress) + 0.02 * progress
        left = _detection(0.5 - offset)
        right = _detection(0.5 + offset)
        tracked = tracker.update([left, right], t, "cam0")
        if len(tracked) == 2:
            by_x = sorted(tracked, key=lambda p: p.pose.centroid()[0])
            labels_by_side.append((by_x[0].fighter_id, by_x[1].fighter_id))

    assert len(labels_by_side) > 0
    assert len(set(labels_by_side)) == 1  # same left/right label pairing throughout
