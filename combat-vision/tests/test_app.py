"""CLI helper tests — no camera or terminal required."""

from __future__ import annotations

import argparse

from combat_vision.app import _resolve_camera_index


def _args(camera: int | None) -> argparse.Namespace:
    return argparse.Namespace(camera=camera)


def test_explicit_camera_zero_is_not_treated_as_falsy() -> None:
    """--camera 0 must resolve to 0, not silently fall back to the default.

    A prior bug used ``args.camera or default_index``, which treats 0 as
    falsy and substitutes the default — wrong both for which device opens
    and for the source label recorded with the session.
    """
    assert _resolve_camera_index(_args(0), default_index=3) == 0


def test_missing_camera_falls_back_to_default() -> None:
    """No --camera flag -> the configured default index."""
    assert _resolve_camera_index(_args(None), default_index=3) == 3


def test_explicit_nonzero_camera_is_used() -> None:
    """--camera 2 overrides the default."""
    assert _resolve_camera_index(_args(2), default_index=3) == 2
