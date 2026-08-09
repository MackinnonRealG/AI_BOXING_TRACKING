"""Scene calibration: converting pixel measurements to metres.

v1 uses a single scalar scale derived from one known reference length (ring
rope span, mat edge, or a printed marker). That is accurate enough for speed
comparison as long as motion is roughly parallel to the image plane. Full
homography / multi-camera triangulation are roadmap items.
"""

from __future__ import annotations

from dataclasses import dataclass

from combat_vision.events.types import SpeedUnit


@dataclass(slots=True)
class Calibration:
    """Holds the pixel→metre scale for one camera, if known.

    All engines accept a ``Calibration`` and use :meth:`scale_speed` /
    :meth:`scale_length` so calibrated and uncalibrated sessions flow through
    identical code paths — only the unit of the output changes.

    Mutable on purpose: the live UI's calibration mode calls
    :meth:`set_scale` mid-session, and every engine holding this object
    starts producing metric output from that moment on.
    """

    metres_per_pixel: float | None
    frame_width_px: int
    frame_height_px: int

    @classmethod
    def from_reference(
        cls,
        reference_length_m: float | None,
        reference_length_px: float | None,
        frame_width_px: int,
        frame_height_px: int,
    ) -> Calibration:
        """Build from a known real-world reference, or uncalibrated if absent."""
        scale = None
        if reference_length_m and reference_length_px:
            scale = reference_length_m / reference_length_px
        return cls(
            metres_per_pixel=scale,
            frame_width_px=frame_width_px,
            frame_height_px=frame_height_px,
        )

    def set_scale(self, metres_per_pixel: float | None) -> None:
        """Set (or clear) the pixel→metre scale at runtime."""
        self.metres_per_pixel = metres_per_pixel

    @property
    def is_calibrated(self) -> bool:
        """True when metric output is available."""
        return self.metres_per_pixel is not None

    @property
    def unit(self) -> SpeedUnit:
        """The unit measurements will be expressed in."""
        return (
            SpeedUnit.METERS_PER_SECOND if self.is_calibrated else SpeedUnit.PIXELS_PER_SECOND
        )

    def to_pixels(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        """Convert normalized image coordinates to pixel coordinates."""
        return (x_norm * self.frame_width_px, y_norm * self.frame_height_px)

    def scale_length(self, length_px: float) -> float:
        """Pixels → metres if calibrated, otherwise pass through pixels."""
        if self.metres_per_pixel is None:
            return length_px
        return length_px * self.metres_per_pixel

    def scale_speed(self, speed_px_per_s: float) -> float:
        """px/s → m/s if calibrated, otherwise pass through px/s."""
        return self.scale_length(speed_px_per_s)


def multi_camera_fusion_stub() -> None:
    """Placeholder for future multi-camera calibration and fusion.

    TODO:
        * Per-camera intrinsics (checkerboard) + extrinsics from shared
          ring-corner references.
        * Triangulate keypoints visible from 2+ cameras into 3D, falling back
          to single-view 2D per fighter otherwise.
        * Feed fused 3D poses through the same TrackedPose contract with
          true metric coordinates (calibration then becomes identity).
    """
    raise NotImplementedError("multi-camera fusion is a roadmap module")
