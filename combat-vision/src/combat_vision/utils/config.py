"""Typed configuration loaded from YAML.

Pydantic validates every tunable at startup, so a typo'd threshold fails
loudly at launch instead of silently mis-measuring a session.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class CaptureConfig(BaseModel):
    """Camera/frame settings."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30


class PoseConfig(BaseModel):
    """Pose backend selection and model parameters."""

    backend: str = "mediapipe"
    model_variant: str = "lite"  # lite | full | heavy (speed vs accuracy)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class TrackingConfig(BaseModel):
    """Identity-assignment tunables."""

    backend: str = "supervision"  # supervision (ByteTrack) | centroid; 't' toggles live
    max_match_distance: float = 0.25
    max_missed_frames: int = 15
    max_fighters: int = 2


class FilteringConfig(BaseModel):
    """One-Euro filter parameters."""

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


class CalibrationConfig(BaseModel):
    """Known reference for pixel→metre scaling (both null = uncalibrated)."""

    reference_length_m: float | None = None
    reference_length_px: float | None = None


class SpeedEngineConfig(BaseModel):
    """Speed engine thresholds; *_pxps values are the uncalibrated fallbacks."""

    smoothing_window: int = 5
    start_speed_mps: float = 3.0
    peak_min_speed_mps: float = 4.5
    start_speed_pxps: float = 900
    peak_min_speed_pxps: float = 1400
    end_speed_ratio: float = 0.4
    min_event_interval_s: float = 0.20
    max_event_duration_s: float = 0.60


class PowerEngineConfig(BaseModel):
    """Weights and normalization ceilings for the estimated power score."""

    speed_weight: float = 0.5
    extension_weight: float = 0.25
    rotation_weight: float = 0.25
    speed_ceiling_mps: float = 12.0     # hand speed that maps to component = 1.0
    speed_ceiling_pxps: float = 4000.0  # uncalibrated fallback of the same ceiling
    rotation_ceiling_dps: float = 600.0  # shoulder-line angular speed ceiling (deg/s)


class StrikeClassifierConfig(BaseModel):
    """Heuristic thresholds for strike classification."""

    min_confidence: float = 0.6
    straight_elbow_min_deg: float = 150.0  # elbow angle at extension for jab/cross
    hook_elbow_max_deg: float = 130.0      # bent-arm ceiling for hooks
    uppercut_vertical_ratio: float = 1.2   # |dy| must exceed ratio * |dx|
    path_curve_ratio: float = 1.25         # path length / straight distance above -> arc
    side_kick_rotation_deg: float = 45.0   # torso rotation separating side from front kick
    landed_max_distance_m: float = 0.35    # strike endpoint within this of a target = landed
    landed_max_distance_px: float = 120.0  # uncalibrated fallback
    opponent_max_age_s: float = 0.5        # opponent pose freshness needed for landed calls


class FootworkConfig(BaseModel):
    """Step detection, sampling, and heat-map binning."""

    step_min_displacement_m: float = 0.08
    step_min_displacement_px: float = 40.0
    plant_speed_mps: float = 0.6           # foot slower than this counts as planted
    plant_speed_pxps: float = 200.0
    plant_frames: int = 3                  # consecutive slow frames to confirm a plant
    sample_interval_s: float = 0.2         # FootworkSample decimation
    heatmap_bins: tuple[int, int] = (48, 27)


class StanceConfig(BaseModel):
    """Stance classification and switch debouncing."""

    switch_debounce_s: float = 0.5
    square_width_ratio: float = 0.5  # ankle x-separation under ratio*shoulder width = square


class RotationEngineConfig(BaseModel):
    """Hip-shoulder rotation ("power source") fault detection thresholds."""

    min_shoulder_rotation_deg: float = 15.0  # below this, shoulder turn is too small to judge
    min_hip_ratio: float = 0.5  # hip rotation below this fraction of shoulder rotation = fault


class KneeBendConfig(BaseModel):
    """Locked-knee posture and no-leg-drive fault thresholds."""

    locked_angle_deg: float = 165.0  # knee angle at/above this counts as "locked"
    lock_debounce_s: float = 1.0     # both knees must stay locked this long to flag posture
    bend_debounce_s: float = 0.15    # recovery debounce, mirrors guard.py's pattern


class DepthPostureConfig(BaseModel):
    """Approximate elbow-flare / torso-lean sampling (unitless MediaPipe z)."""

    sample_interval_s: float = 0.2  # DepthPostureSample decimation, mirrors footwork


class HeadPostureConfig(BaseModel):
    """Head-roll measurement sampling."""

    sample_interval_s: float = 0.2  # HeadPostureSample decimation, mirrors footwork


class GuardConfig(BaseModel):
    """Guard-height fault detection: how close each hand must stay to the chin."""

    chin_ratio: float = 0.35        # chin position between nose (0) and shoulder line (1)
    drop_margin_ratio: float = 0.25  # extra tolerance below the chin line still read as "up"
    drop_debounce_s: float = 0.6     # hand must be below the line this long before flagging
    recover_debounce_s: float = 0.15  # hand must be above the line this long before clearing


class DistanceEngineConfig(BaseModel):
    """Inter-fighter distance sampling."""

    sample_interval_s: float = 0.2
    max_pose_age_s: float = 0.15  # both fighters must be seen within this window


class CombinationConfig(BaseModel):
    """Combo chaining rules."""

    max_gap_s: float = 1.0
    min_length: int = 2


class EnginesConfig(BaseModel):
    """All metrics-engine tunables."""

    speed: SpeedEngineConfig = SpeedEngineConfig()
    power: PowerEngineConfig = PowerEngineConfig()
    strike_classifier: StrikeClassifierConfig = StrikeClassifierConfig()
    footwork: FootworkConfig = FootworkConfig()
    stance: StanceConfig = StanceConfig()
    guard: GuardConfig = GuardConfig()
    rotation: RotationEngineConfig = RotationEngineConfig()
    knee_bend: KneeBendConfig = KneeBendConfig()
    head_posture: HeadPostureConfig = HeadPostureConfig()
    depth_posture: DepthPostureConfig = DepthPostureConfig()
    distance: DistanceEngineConfig = DistanceEngineConfig()
    combination: CombinationConfig = CombinationConfig()


class UiConfig(BaseModel):
    """Overlay rendering settings."""

    draw_skeleton: bool = True
    heatmap_alpha: float = 0.35
    window_name: str = "Combat Vision"
    mirror: bool = True                 # mirror view — natural for solo training
    min_keypoint_visibility: float = 0.5  # hide keypoints the model can't see
    toast_duration_s: float = 0.8       # how long strike pop-ups stay on screen


class StorageConfig(BaseModel):
    """Persistence settings."""

    database_url: str = "sqlite:///combat_vision.db"


class AppMetaConfig(BaseModel):
    """Top-level app settings."""

    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Root configuration object passed through the whole system."""

    app: AppMetaConfig = AppMetaConfig()
    capture: CaptureConfig = CaptureConfig()
    pose: PoseConfig = PoseConfig()
    tracking: TrackingConfig = TrackingConfig()
    filtering: FilteringConfig = FilteringConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    engines: EnginesConfig = EnginesConfig()
    ui: UiConfig = UiConfig()
    storage: StorageConfig = StorageConfig()


def default_config_path() -> Path:
    """Repo-relative default config location."""
    # utils/config.py -> utils -> combat_vision -> src -> repo root
    return Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration; missing file falls back to defaults."""
    candidate = Path(path) if path is not None else default_config_path()
    if candidate.exists():
        with candidate.open() as fh:
            raw = yaml.safe_load(fh) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()
