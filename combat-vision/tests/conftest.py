"""Shared test helpers: load JSON pose fixtures into pipeline contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from combat_vision.calibration import Calibration
from combat_vision.events.types import Keypoint, KeypointName, Pose, TrackedPose

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_sequence(name: str, fighter_id: str = "A") -> tuple[list[TrackedPose], dict]:
    """Load a fixture into TrackedPose objects plus its metadata dict."""
    data = json.loads((FIXTURES_DIR / name).read_text())
    poses = []
    for frame in data["frames"]:
        keypoints = {
            KeypointName(kp_name): Keypoint(x=xy[0], y=xy[1])
            for kp_name, xy in frame["keypoints"].items()
        }
        poses.append(
            TrackedPose(
                fighter_id=fighter_id,
                pose=Pose(keypoints=keypoints),
                timestamp_s=frame["t"],
            )
        )
    return poses, data


def calibration_from_meta(meta: dict, calibrated: bool = True) -> Calibration:
    """Build a Calibration matching a fixture's recording geometry."""
    return Calibration(
        metres_per_pixel=meta["metres_per_pixel"] if calibrated else None,
        frame_width_px=meta["frame_width"],
        frame_height_px=meta["frame_height"],
    )


@pytest.fixture()
def jab_sequence() -> tuple[list[TrackedPose], dict]:
    """A synthetic left jab with true peak speed = meta['true_peak_speed_mps']."""
    return load_sequence("jab_sequence.json")


@pytest.fixture()
def idle_sequence() -> tuple[list[TrackedPose], dict]:
    """Two seconds of stillness with realistic estimator jitter."""
    return load_sequence("idle_sequence.json")
