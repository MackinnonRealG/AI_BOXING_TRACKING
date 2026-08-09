"""Small 2D geometry helpers shared by the metrics engines.

All functions operate on pixel-space points ``(x, y)`` with y pointing down
(image convention).
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    """Midpoint of the segment ab."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle_at(vertex: Point, a: Point, b: Point) -> float:
    """Interior angle in degrees at ``vertex`` formed by rays to ``a`` and ``b``.

    Returns 180.0 for a fully straight joint, smaller values for flexed ones.
    Degenerate (zero-length) rays return 180.0 so missing-resolution frames
    read as "straight" rather than raising.
    """
    v1 = (a[0] - vertex[0], a[1] - vertex[1])
    v2 = (b[0] - vertex[0], b[1] - vertex[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def line_angle(a: Point, b: Point) -> float:
    """Angle of the line a→b in degrees, in ``[-180, 180]``."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def angle_delta(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff)


def path_length(points: list[Point]) -> float:
    """Total polyline length through ``points``."""
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))
