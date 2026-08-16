"""Two-camera triangulation math for future multi-camera fusion.

Standalone pinhole-camera geometry: given two calibrated cameras' intrinsic
and extrinsic parameters and one 2D pixel observation of the same physical
point from each, recover its 3D position via direct linear transform (DLT)
triangulation.

This module is the math half of the roadmap item described in
:func:`combat_vision.calibration.calibrator.multi_camera_fusion_stub` — it
does **not** wire into live capture. Turning this into working multi-camera
fusion still needs, and does not have: per-camera intrinsic calibration from
real checkerboard captures, extrinsic calibration from shared scene
references, and synchronized dual-camera frame capture in the pipeline.
None of that can be built or validated without physical cameras in hand, so
it stays out of scope here. What *is* here is fully self-contained and
tested against synthetic camera geometry — the arithmetic a future
integration would call once those pieces exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CameraParams:
    """A calibrated camera's intrinsics and pose (standard pinhole model).

    ``rotation`` and ``translation`` map a *world*-space point into that
    camera's own coordinate frame: ``p_cam = rotation @ p_world + translation``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3,

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """The 3x3 camera intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def projection_matrix(self) -> np.ndarray:
        """The 3x4 camera projection matrix P = K [R | t]."""
        extrinsic = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return self.intrinsic_matrix @ extrinsic


def project(params: CameraParams, point_3d: Point3D) -> Point2D:
    """Project a world-space 3D point into this camera's pixel coordinates.

    Used to build synthetic two-view fixtures for testing
    :func:`triangulate` — a real pipeline would never call this the other
    way around (it observes pixels and wants the 3D point back).
    """
    world = np.array([*point_3d, 1.0])
    projected = params.projection_matrix @ world
    if projected[2] <= 0:
        raise ValueError("point is behind or on the camera plane")
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def triangulate(
    params_a: CameraParams, point_a: Point2D, params_b: CameraParams, point_b: Point2D
) -> Point3D:
    """Recover a 3D point from its 2D observation in two calibrated cameras.

    Direct linear transform (DLT): each 2D observation contributes two
    linear constraints on the unknown world point (its projection ray must
    pass through the observed pixel); stacking both cameras' constraints
    into one 4x4 system and taking the least-squares solution (via SVD)
    gives the point that best satisfies both rays simultaneously — the
    standard closed-form triangulation, robust to the two rays not
    perfectly intersecting (they generally won't, under any pixel noise).
    """
    p_a, p_b = params_a.projection_matrix, params_b.projection_matrix
    x_a, y_a = point_a
    x_b, y_b = point_b

    design = np.array(
        [
            x_a * p_a[2] - p_a[0],
            y_a * p_a[2] - p_a[1],
            x_b * p_b[2] - p_b[0],
            y_b * p_b[2] - p_b[1],
        ]
    )
    _, _, vt = np.linalg.svd(design)
    homogeneous = vt[-1]
    if homogeneous[3] == 0:
        raise ValueError("triangulated point is at infinity — rays are parallel")
    world = homogeneous[:3] / homogeneous[3]
    return (float(world[0]), float(world[1]), float(world[2]))
