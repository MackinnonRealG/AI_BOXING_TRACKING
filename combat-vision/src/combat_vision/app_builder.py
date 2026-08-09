"""Composition root: builds a fully wired pipeline from configuration.

Lives apart from :mod:`combat_vision.app` (the CLI) so review mode, tests,
and the future web server can construct identical pipelines.
"""

from __future__ import annotations

from combat_vision.calibration import Calibration
from combat_vision.capture.base import CameraSource
from combat_vision.engines.base import MetricsEngine
from combat_vision.engines.combination import CombinationEngine
from combat_vision.engines.distance import DistanceEngine
from combat_vision.engines.footwork import FootworkEngine
from combat_vision.engines.power import PowerEngine
from combat_vision.engines.speed import SpeedEngine
from combat_vision.engines.stance import StanceEngine
from combat_vision.engines.strike_classifier import StrikeClassifierEngine
from combat_vision.events.bus import EventBus
from combat_vision.filtering.smoother import PoseSmoother
from combat_vision.pipeline import FrameSink, Pipeline
from combat_vision.pose.base import PoseBackend
from combat_vision.pose.mediapipe_backend import MediaPipePoseBackend
from combat_vision.pose.yolo_backend import YoloV8PoseBackend
from combat_vision.sports import get_profile
from combat_vision.tracking import FighterTracker, SupervisionTracker, SwitchableTracker
from combat_vision.utils.config import AppConfig


def build_pose_backend(config: AppConfig) -> PoseBackend:
    """Instantiate the configured pose backend."""
    if config.pose.backend == "mediapipe":
        return MediaPipePoseBackend(
            model_variant=config.pose.model_variant,
            num_poses=config.tracking.max_fighters,
            min_detection_confidence=config.pose.min_detection_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
        )
    if config.pose.backend == "yolov8":
        return YoloV8PoseBackend()
    raise ValueError(f"unknown pose backend {config.pose.backend!r}")


def build_pipeline(
    source: CameraSource,
    sport: str,
    config: AppConfig,
    frame_sink: FrameSink | None,
) -> tuple[Pipeline, EventBus]:
    """Wire capture→pose→tracking→smoothing→engines→bus for either mode."""
    profile = get_profile(sport)
    bus = EventBus()
    calibration = Calibration.from_reference(
        reference_length_m=config.calibration.reference_length_m,
        reference_length_px=config.calibration.reference_length_px,
        frame_width_px=config.capture.frame_width,
        frame_height_px=config.capture.frame_height,
    )

    # Order matters: engines that buffer poses (stance, power, classifier)
    # must process each frame BEFORE the speed engine, so their buffers
    # already contain the current pose when a SpeedPeakEvent fires mid-frame.
    engines: list[MetricsEngine] = [
        StanceEngine(bus, profile, calibration, config.engines.stance),
        PowerEngine(bus, profile, calibration, config.engines.power),
        StrikeClassifierEngine(bus, profile, calibration, config.engines.strike_classifier),
        FootworkEngine(bus, profile, calibration, config.engines.footwork),
        DistanceEngine(bus, profile, calibration, config.engines.distance),
        SpeedEngine(bus, profile, calibration, config.engines.speed),
        CombinationEngine(bus, profile, calibration, config.engines.combination),
    ]

    # Both trackers are always built; SwitchableTracker lets the live UI
    # (the 't' key) flip between them without restarting the pipeline.
    tracker = SwitchableTracker(
        primary=SupervisionTracker(
            frame_width_px=config.capture.frame_width,
            frame_height_px=config.capture.frame_height,
            max_fighters=config.tracking.max_fighters,
            max_missed_frames=config.tracking.max_missed_frames,
            frame_rate=config.capture.target_fps,
        ),
        fallback=FighterTracker(
            max_match_distance=config.tracking.max_match_distance,
            max_missed_frames=config.tracking.max_missed_frames,
            max_fighters=config.tracking.max_fighters,
        ),
        use_primary=config.tracking.backend == "supervision",
    )
    smoother = PoseSmoother(
        min_cutoff=config.filtering.min_cutoff,
        beta=config.filtering.beta,
        d_cutoff=config.filtering.d_cutoff,
    )
    pipeline = Pipeline(
        source=source,
        pose_backend=build_pose_backend(config),
        tracker=tracker,
        smoother=smoother,
        engines=engines,
        bus=bus,
        frame_sink=frame_sink,
    )
    return pipeline, bus
