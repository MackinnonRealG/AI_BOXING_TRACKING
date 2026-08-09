"""Pattern recognition: mining win-correlated behavior from stored sessions.

Round outcomes are user-labelled (``Round.outcome`` holds the winning
fighter's label, e.g. ``"A"``, or ``"draw"``); this module never infers who
won. It reads *storage only* — no camera required.

v1 algorithm — median-split association mining:

1. Build a feature vector per labelled round for the fighter of interest
   (output rate, avg/max hand speed, combo count, stance switches, avg
   estimated power, avg opponent distance).
2. For each feature, split rounds at the median and compare win rates of
   the high half vs the low half.
3. A gap of at least ``_MIN_WIN_RATE_GAP`` with enough supporting rounds
   becomes a human-readable :class:`Pattern`, ranked by gap × support.

Roadmap: replace with gradient-boosted trees + SHAP once ~100 labelled
rounds exist; the interface stays identical.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from combat_vision.storage.models import Round
from combat_vision.storage.repository import SessionRepository

_MIN_WIN_RATE_GAP = 0.15

_FEATURE_LABELS: dict[str, str] = {
    "punches_per_min": "punch output (candidates/min)",
    "avg_peak_speed": "average hand speed",
    "max_peak_speed": "best hand speed",
    "combo_count": "combinations thrown",
    "stance_switches": "stance switches",
    "avg_power": "average estimated power",
    "avg_distance": "average distance to opponent",
}


@dataclass(frozen=True, slots=True)
class Pattern:
    """A discovered behavior→outcome correlation."""

    description: str
    """e.g. 'rounds with high punch output were won 78% vs 40% of the time'."""
    support: int
    """Number of labelled rounds the pattern was computed from."""
    win_rate: float
    """Win rate of the favorable half of the split."""
    confidence: float
    """Win-rate gap between the two halves, 0..1 (association strength)."""


class PatternRecognizer:
    """Mines stored sessions for patterns that correlate with winning rounds."""

    def __init__(self, repo: SessionRepository) -> None:
        self._repo = repo

    def discover(self, fighter_label: str, min_support: int = 5) -> list[Pattern]:
        """Return win-correlated patterns for ``fighter_label``.

        Needs at least ``min_support`` labelled rounds on *each* side of a
        split; returns an empty list when there is not enough labelled data.
        """
        rows = self._collect(fighter_label)
        if len(rows) < 2 * min_support:
            return []

        patterns: list[Pattern] = []
        for feature, label in _FEATURE_LABELS.items():
            values: list[tuple[float, bool]] = []
            for features, won in rows:
                value = features.get(feature)
                if value is not None:
                    values.append((value, won))
            if len(values) < 2 * min_support:
                continue
            median = statistics.median(v for v, _ in values)
            high = [won for v, won in values if v > median]
            low = [won for v, won in values if v <= median]
            if len(high) < min_support or len(low) < min_support:
                continue
            high_rate = sum(high) / len(high)
            low_rate = sum(low) / len(low)
            gap = high_rate - low_rate
            if abs(gap) < _MIN_WIN_RATE_GAP:
                continue
            favorable = "high" if gap > 0 else "low"
            win_rate = high_rate if gap > 0 else low_rate
            other_rate = low_rate if gap > 0 else high_rate
            patterns.append(
                Pattern(
                    description=(
                        f"Fighter {fighter_label}: rounds with {favorable} {label} were won "
                        f"{win_rate:.0%} of the time vs {other_rate:.0%} otherwise."
                    ),
                    support=len(values),
                    win_rate=win_rate,
                    confidence=abs(gap),
                )
            )
        patterns.sort(key=lambda p: p.confidence * p.support, reverse=True)
        return patterns

    def _collect(self, fighter_label: str) -> list[tuple[dict[str, float | None], bool]]:
        """(features, won) per labelled round for the fighter."""
        rows: list[tuple[dict[str, float | None], bool]] = []
        for session in self._repo.list_sessions():
            events = self._repo.events_for_session(session.id)
            for rnd in self._repo.rounds_for_session(session.id):
                if rnd.outcome is None or rnd.outcome == "draw" or rnd.end_s is None:
                    continue
                rows.append(
                    (
                        _round_features(fighter_label, rnd, events),
                        rnd.outcome == fighter_label,
                    )
                )
        return rows


def _round_features(
    fighter_label: str, rnd: Round, events: list
) -> dict[str, float | None]:
    """Aggregate one round's events into the feature vector."""
    assert rnd.end_s is not None
    in_round = [
        e
        for e in events
        if rnd.start_s <= e.timestamp_s <= rnd.end_s and e.fighter_label == fighter_label
    ]
    by_type: dict[str, list] = {}
    for e in in_round:
        by_type.setdefault(e.event_type, []).append(e)

    speeds = [e.payload["peak_speed"] for e in by_type.get("SpeedPeakEvent", [])]
    powers = [e.payload["score"] for e in by_type.get("PowerEstimateEvent", [])]
    distances = [e.payload["distance"] for e in by_type.get("DistanceSample", [])]
    minutes = max((rnd.end_s - rnd.start_s) / 60.0, 1e-6)
    return {
        "punches_per_min": len(speeds) / minutes,
        "avg_peak_speed": statistics.fmean(speeds) if speeds else None,
        "max_peak_speed": max(speeds) if speeds else None,
        "combo_count": float(len(by_type.get("ComboEvent", []))),
        "stance_switches": float(len(by_type.get("StanceSwitchEvent", []))),
        "avg_power": statistics.fmean(powers) if powers else None,
        "avg_distance": statistics.fmean(distances) if distances else None,
    }
