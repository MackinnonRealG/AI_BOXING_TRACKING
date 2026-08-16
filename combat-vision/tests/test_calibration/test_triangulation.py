"""Triangulation math tests: synthetic two-camera geometry, no real cameras needed."""

from __future__ import annotations

import numpy as np
import pytest

from combat_vision.calibration.triangulation import CameraParams, project, triangulate


def _camera(
    translation: tuple[float, float, float], rotation: np.ndarray | None = None
) -> CameraParams:
    return CameraParams(
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        rotation=np.eye(3) if rotation is None else rotation,
        translation=np.array(translation),
    )


def test_project_matches_pinhole_geometry_on_the_optical_axis() -> None:
    """A point straight ahead on the optical axis projects to the principal point."""
    cam = _camera(translation=(0.0, 0.0, 0.0))
    x, y = project(cam, (0.0, 0.0, 3.0))
    assert x == pytest.approx(cam.cx)
    assert y == pytest.approx(cam.cy)


def test_triangulate_recovers_a_known_point_from_a_parallel_stereo_rig() -> None:
    cam_a = _camera(translation=(0.0, 0.0, 0.0))
    cam_b = _camera(translation=(-0.5, 0.0, 0.0))  # camera B sits 0.5m to the +x side of A

    true_point = (0.2, -0.1, 5.0)
    pixel_a = project(cam_a, true_point)
    pixel_b = project(cam_b, true_point)

    recovered = triangulate(cam_a, pixel_a, cam_b, pixel_b)
    assert recovered == pytest.approx(true_point, abs=1e-6)


def test_triangulate_handles_a_toed_in_second_camera() -> None:
    """A rotated second camera (not a pure parallel rig) still triangulates correctly."""
    cam_a = _camera(translation=(0.0, 0.0, 0.0))
    angle = np.radians(15.0)
    rotation_b = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    camera_b_world_position = np.array([1.0, 0.0, 0.0])
    translation_b = -rotation_b @ camera_b_world_position
    cam_b = _camera(translation=tuple(translation_b), rotation=rotation_b)

    true_point = (0.3, 0.2, 6.0)
    pixel_a = project(cam_a, true_point)
    pixel_b = project(cam_b, true_point)

    recovered = triangulate(cam_a, pixel_a, cam_b, pixel_b)
    assert recovered == pytest.approx(true_point, abs=1e-6)


def test_triangulate_is_robust_to_small_pixel_noise() -> None:
    """Perturbed (non-intersecting-ray) observations still recover a close point."""
    cam_a = _camera(translation=(0.0, 0.0, 0.0))
    cam_b = _camera(translation=(-0.5, 0.0, 0.0))
    true_point = (0.0, 0.0, 5.0)
    x_a, y_a = project(cam_a, true_point)
    x_b, y_b = project(cam_b, true_point)

    recovered = triangulate(cam_a, (x_a + 0.5, y_a - 0.5), cam_b, (x_b - 0.5, y_b + 0.5))
    # Depth error from sub-pixel noise scales with (depth^2 / baseline), so a
    # 0.5px jitter at 5m depth over a 0.5m baseline is expected to move the
    # recovered depth by several centimeters — this tolerance reflects that,
    # not a loosened correctness bar.
    assert recovered == pytest.approx(true_point, abs=0.1)


def test_near_parallel_rays_raise_instead_of_returning_a_wild_coordinate() -> None:
    """As depth grows, a parallel stereo rig's rays approach true parallelism.

    Before the fix, only an exact ``w == 0`` raised; anything merely tiny
    (e.g. w ~ 1e-10, which this depth reliably produces — see the module
    docstring's tolerance) would silently divide out to a huge, meaningless
    coordinate instead of raising. The homogeneous w shrinks continuously
    with depth rather than snapping to exactly zero, so a real near-parallel
    configuration is what actually exercises the tolerance, not two
    literally-identical cameras (whose rank-deficient null space doesn't
    reliably land on the "at infinity" direction at all).
    """
    cam_a = _camera(translation=(0.0, 0.0, 0.0))
    cam_b = _camera(translation=(-0.5, 0.0, 0.0))

    far_point = (0.0, 0.0, 1e13)
    pixel_a = project(cam_a, far_point)
    pixel_b = project(cam_b, far_point)

    with pytest.raises(ValueError, match="parallel"):
        triangulate(cam_a, pixel_a, cam_b, pixel_b)


def test_point_behind_the_camera_raises() -> None:
    cam = _camera(translation=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="behind or on the camera plane"):
        project(cam, (0.0, 0.0, -1.0))
