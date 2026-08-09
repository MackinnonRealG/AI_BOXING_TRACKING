"""Combat Vision CLI entry point.

Usage:
    combat-vision live --sport boxing [--camera 0]
    combat-vision review path/to/sparring.mp4 --sport kickboxing [--output report.json]
"""

from __future__ import annotations

import argparse
import logging
import sys

from combat_vision.utils.config import load_config
from combat_vision.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_SPORTS = ("boxing", "kickboxing")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="combat-vision", description=__doc__)
    parser.add_argument("--config", default=None, help="path to a config YAML")
    sub = parser.add_subparsers(dest="mode", required=True)

    live = sub.add_parser("live", help="real-time camera analysis with overlay")
    live.add_argument("--sport", choices=_SPORTS, help="sport profile (prompted if omitted)")
    live.add_argument("--camera", type=int, default=None, help="webcam index")
    live.add_argument("--rtsp", default=None, help="RTSP/IP stream URL instead of a webcam")

    review = sub.add_parser("review", help="analyse recorded footage into a report")
    review.add_argument("video", help="path to the video file")
    review.add_argument("--sport", choices=_SPORTS, help="sport profile (prompted if omitted)")
    review.add_argument("--output", default=None, help="write the JSON report here")
    review.add_argument("--no-store", action="store_true", help="skip persisting the session")
    return parser


def _select_sport(current: str | None) -> str:
    """Interactive sport selection when --sport was not given."""
    if current is not None:
        return current
    print("Select sport:")
    for i, sport in enumerate(_SPORTS, start=1):
        print(f"  {i}. {sport}")
    choice = input("> ").strip()
    if choice in {"1", "2"}:
        return _SPORTS[int(choice) - 1]
    if choice.lower() in _SPORTS:
        return choice.lower()
    print(f"unrecognised choice {choice!r}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    """Parse arguments and launch the requested mode."""
    args = _parser().parse_args()
    config = load_config(args.config)
    configure_logging(config.app.log_level)
    sport = _select_sport(args.sport)

    if args.mode == "live":
        _run_live(args, sport, config)
    else:
        _run_review(args, sport, config)


def _run_live(args: argparse.Namespace, sport: str, config: object) -> None:
    """Live mode: camera → pipeline → overlay window."""
    from combat_vision.app_builder import build_pipeline
    from combat_vision.capture.rtsp import RtspSource
    from combat_vision.capture.webcam import WebcamSource
    from combat_vision.tracking import SwitchableTracker
    from combat_vision.ui.overlay import LiveOverlay
    from combat_vision.utils.config import AppConfig

    assert isinstance(config, AppConfig)
    source = (
        RtspSource(args.rtsp)
        if args.rtsp
        else WebcamSource(
            index=args.camera if args.camera is not None else config.capture.camera_index,
            width=config.capture.frame_width,
            height=config.capture.frame_height,
        )
    )
    # Build the pipeline first, then bind the overlay to its bus.
    pipeline, bus = build_pipeline(source=source, sport=sport, config=config, frame_sink=None)
    tracker = pipeline.tracker
    overlay = LiveOverlay(
        bus, config.ui, tracker=tracker if isinstance(tracker, SwitchableTracker) else None
    )
    pipeline.set_frame_sink(overlay.render)
    logger.info(
        "live mode started sport=%s (q quits, h heatmap, t tracker toggle)", sport
    )
    try:
        pipeline.run()
    finally:
        overlay.close()


def _run_review(args: argparse.Namespace, sport: str, config: object) -> None:
    """Review mode: video file → pipeline → report (stdout + JSON + storage)."""
    from combat_vision.review.processor import review_video
    from combat_vision.storage.repository import SessionRepository
    from combat_vision.utils.config import AppConfig

    assert isinstance(config, AppConfig)
    repo = None if args.no_store else SessionRepository(config.storage.database_url)
    report = review_video(
        video_path=args.video,
        sport=sport,
        config=config,
        output_json=args.output,
        repo=repo,
    )
    print(report.to_text())


if __name__ == "__main__":
    main()
