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
    live.add_argument("--no-store", action="store_true", help="skip persisting the session")
    live.add_argument("--name-a", default=None, help="fighter A's name (shown on screen, saved)")
    live.add_argument("--name-b", default=None, help="fighter B's name (shown on screen, saved)")

    review = sub.add_parser("review", help="analyse recorded footage into a report")
    review.add_argument("video", help="path to the video file")
    review.add_argument("--sport", choices=_SPORTS, help="sport profile (prompted if omitted)")
    review.add_argument("--output", default=None, help="write the JSON report here")
    review.add_argument("--no-store", action="store_true", help="skip persisting the session")
    review.add_argument("--name-a", default=None, help="fighter A's name (used in report, saved)")
    review.add_argument("--name-b", default=None, help="fighter B's name (used in report, saved)")
    return parser


def _fighter_names(args: argparse.Namespace) -> dict[str, str]:
    """Label -> display-name mapping from CLI flags; interactive fallback.

    With no flags given, live mode asks once at startup (enter to skip).
    """
    names: dict[str, str] = {}
    if args.name_a:
        names["A"] = args.name_a
    if args.name_b:
        names["B"] = args.name_b
    if not names and args.mode == "live" and sys.stdin.isatty():
        entered = input("Fighter A's name (enter to skip): ").strip()
        if entered:
            names["A"] = entered
        entered = input("Fighter B's name (enter to skip): ").strip()
        if entered:
            names["B"] = entered
    return names


def _resolve_camera_index(args: argparse.Namespace, default_index: int) -> int:
    """The webcam index to open: ``--camera`` if given, else ``default_index``.

    Must use ``is not None`` — ``args.camera or default_index`` would treat
    an explicit ``--camera 0`` as falsy and silently substitute the default,
    which is wrong both for opening the device and for the source label
    recorded with the session.
    """
    return args.camera if args.camera is not None else default_index


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
    """Live mode: camera → pipeline → HUD; session saved + reported on quit."""
    from combat_vision.analytics.reports import build_session_report
    from combat_vision.app_builder import build_pipeline
    from combat_vision.capture.rtsp import RtspSource
    from combat_vision.capture.webcam import WebcamSource
    from combat_vision.storage.repository import SessionRepository
    from combat_vision.tracking import SwitchableTracker
    from combat_vision.ui.overlay import LiveOverlay
    from combat_vision.utils.config import AppConfig

    assert isinstance(config, AppConfig)
    names = _fighter_names(args)
    camera_index = _resolve_camera_index(args, config.capture.camera_index)
    source_name = args.rtsp if args.rtsp else f"webcam:{camera_index}"
    source = (
        RtspSource(args.rtsp)
        if args.rtsp
        else WebcamSource(
            index=camera_index,
            width=config.capture.frame_width,
            height=config.capture.frame_height,
        )
    )
    # Build the pipeline first, then bind the HUD to its bus and calibration.
    pipeline, bus, calibration = build_pipeline(
        source=source, sport=sport, config=config, frame_sink=None
    )
    bus.start_recording()
    tracker = pipeline.tracker
    overlay = LiveOverlay(
        bus,
        config.ui,
        tracker=tracker if isinstance(tracker, SwitchableTracker) else None,
        calibration=calibration,
        sport=sport,
        names=names,
    )
    pipeline.set_frame_sink(overlay.render)
    logger.info("live mode started sport=%s — controls are shown on screen", sport)
    duration_s = 0.0
    try:
        duration_s = pipeline.run()
    finally:
        overlay.close()

    report = build_session_report(
        sport=sport, source=source_name, duration_s=duration_s, events=bus.history, names=names
    )

    if not args.no_store and bus.history:
        from combat_vision.analytics.baseline import personal_best_notes

        repo = SessionRepository(config.storage.database_url)
        session_id = repo.create_session(
            sport=sport, mode="live", source=source_name, calibrated=calibration.is_calibrated
        )
        repo.save_events(session_id, bus.history)
        for label in sorted({e.fighter_id for e in bus.history}):
            fighter_db_id = repo.get_or_create_fighter(names.get(label, f"Fighter {label}"))
            repo.link_fighter(session_id, fighter_db_id, label)
            report.coaching_notes.extend(personal_best_notes(repo, fighter_db_id, session_id))
        for number, start_s, end_s in overlay.rounds:
            repo.create_round(session_id, number=number, start_s=start_s, end_s=end_s)
        repo.finish_session(session_id, duration_s)
        logger.info(
            "session %d saved: %d events, %d rounds — label round outcomes to "
            "unlock pattern recognition",
            session_id,
            len(bus.history),
            len(overlay.rounds),
        )

    print(report.to_text())


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
        names=_fighter_names(args),
    )
    print(report.to_text())


if __name__ == "__main__":
    main()
