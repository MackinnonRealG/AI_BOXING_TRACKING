"""Review mode: run the identical pipeline over a file, emit a report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from combat_vision.analytics.reports import SessionReport, build_session_report
from combat_vision.app_builder import build_pipeline
from combat_vision.capture.video_file import VideoFileSource
from combat_vision.storage.repository import SessionRepository
from combat_vision.utils.config import AppConfig

logger = logging.getLogger(__name__)


def review_video(
    video_path: str | Path,
    sport: str,
    config: AppConfig,
    output_json: str | Path | None = None,
    repo: SessionRepository | None = None,
) -> SessionReport:
    """Analyse a recorded video and return (and optionally persist) a report.

    Uses the same :class:`~combat_vision.pipeline.Pipeline` as live mode —
    only the source (file) and sink (report instead of overlay) differ.
    """
    source = VideoFileSource(video_path)
    pipeline, bus = build_pipeline(source=source, sport=sport, config=config, frame_sink=None)
    bus.start_recording()

    duration_s = pipeline.run()
    report = build_session_report(
        sport=sport, source=str(video_path), duration_s=duration_s, events=bus.history
    )

    if repo is not None:
        session_id = repo.create_session(
            sport=sport, mode="review", source=str(video_path), calibrated=False
        )
        repo.save_events(session_id, bus.history)
        repo.finish_session(session_id, duration_s)
        logger.info("session %d persisted with %d events", session_id, len(bus.history))

    if output_json is not None:
        Path(output_json).write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("report written to %s", output_json)
    return report
