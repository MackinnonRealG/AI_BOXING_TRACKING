"""Structured logging setup (key=value single-line records)."""

from __future__ import annotations

import logging
import sys


class _KeyValueFormatter(logging.Formatter):
    """Renders records as ``ts=... level=... logger=... msg=...``."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}"
            f" level={record.levelname.lower()}"
            f" logger={record.name}"
            f' msg="{record.getMessage()}"'
        )
        if record.exc_info:
            base += f" exc={self.formatException(record.exc_info)!r}"
        return base


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler with structured formatting."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
