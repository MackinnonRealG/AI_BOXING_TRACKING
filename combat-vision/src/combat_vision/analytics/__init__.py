"""Analytics layer: reads *storage only* — works with no camera attached."""

from combat_vision.analytics.reports import SessionReport, build_session_report
from combat_vision.analytics.trends import FighterTrends, compute_trends

__all__ = ["FighterTrends", "SessionReport", "build_session_report", "compute_trends"]
