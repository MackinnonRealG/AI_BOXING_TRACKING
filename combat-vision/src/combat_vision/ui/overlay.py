"""OpenCV live view: skeletons, fighter labels, stat panel, foot heat map.

Render state is fed exclusively by the event bus and the per-frame pose list,
so replacing this class with a websocket publisher (see
:mod:`combat_vision.ui.web`) requires no pipeline changes.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import cv2
import numpy as np

from combat_vision.capture.base import TimestampedFrame
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    SKELETON_EDGES,
    KeypointName,
    PowerEstimateEvent,
    SpeedPeakEvent,
    StanceSample,
    StrikeEvent,
    TrackedPose,
)
from combat_vision.tracking import SwitchableTracker
from combat_vision.utils.config import UiConfig

_FIGHTER_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (80, 200, 80),   # BGR green
    "B": (80, 120, 255),  # BGR orange-red
}
_PANEL_COLOR = (30, 30, 30)
_TEXT_COLOR = (240, 240, 240)
_HEATMAP_BINS = (64, 36)
_FEET = (KeypointName.LEFT_ANKLE, KeypointName.RIGHT_ANKLE)


class LiveOverlay:
    """Draws the live view and owns per-session display state.

    Keys: ``h`` toggles the foot-placement heat map, ``t`` toggles the
    tracker backend (supervision/ByteTrack ↔ centroid), ``q`` quits.
    """

    def __init__(
        self, bus: EventBus, config: UiConfig, tracker: SwitchableTracker | None = None
    ) -> None:
        self._config = config
        self._tracker = tracker
        self._show_heatmap = False
        self._last_speed: dict[str, SpeedPeakEvent] = {}
        self._last_strike: dict[str, StrikeEvent] = {}
        self._last_power: dict[str, PowerEstimateEvent] = {}
        self._candidate_counts: Counter[str] = Counter()
        self._stances: dict[str, str] = {}
        self._foot_heat: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((_HEATMAP_BINS[1], _HEATMAP_BINS[0]), dtype=np.float32)
        )
        bus.subscribe(SpeedPeakEvent, self._on_speed_peak)
        bus.subscribe(StrikeEvent, self._on_strike)
        bus.subscribe(PowerEstimateEvent, self._on_power)
        bus.subscribe(StanceSample, self._on_stance)

    def _on_speed_peak(self, event: SpeedPeakEvent) -> None:
        self._last_speed[event.fighter_id] = event
        self._candidate_counts[event.fighter_id] += 1

    def _on_strike(self, event: StrikeEvent) -> None:
        self._last_strike[event.fighter_id] = event

    def _on_power(self, event: PowerEstimateEvent) -> None:
        self._last_power[event.fighter_id] = event

    def _on_stance(self, event: StanceSample) -> None:
        self._stances[event.fighter_id] = event.stance.value

    def render(self, frame: TimestampedFrame, poses: list[TrackedPose]) -> bool:
        """Draw one frame; return False when the user quits."""
        image = frame.image
        h, w = image.shape[:2]

        for tracked in poses:
            self._accumulate_feet(tracked)
        if self._show_heatmap:
            image = self._blend_heatmap(image)

        if self._config.draw_skeleton:
            for tracked in poses:
                self._draw_skeleton(image, tracked, w, h)
        self._draw_stat_panel(image)

        cv2.imshow(self._config.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("h"):
            self._show_heatmap = not self._show_heatmap
        if key == ord("t") and self._tracker is not None:
            self._tracker.toggle()
        return True

    def close(self) -> None:
        """Destroy the display window."""
        cv2.destroyAllWindows()

    def _draw_skeleton(self, image: np.ndarray, tracked: TrackedPose, w: int, h: int) -> None:
        color = _FIGHTER_COLORS.get(tracked.fighter_id, (200, 200, 200))
        pts: dict[KeypointName, tuple[int, int]] = {}
        for name, kp in tracked.pose.keypoints.items():
            pts[name] = (int(kp.x * w), int(kp.y * h))
        for a, b in SKELETON_EDGES:
            if a in pts and b in pts:
                cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for point in pts.values():
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
        nose = pts.get(KeypointName.NOSE)
        if nose is not None:
            cv2.putText(
                image,
                f"Fighter {tracked.fighter_id}",
                (nose[0] - 40, nose[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_stat_panel(self, image: np.ndarray) -> None:
        header = "[h] heatmap  [q] quit"
        if self._tracker is not None:
            header = f"[h] heatmap  [t] tracker: {self._tracker.active_name}  [q] quit"
        lines = [header]
        for fighter_id in sorted(set(self._candidate_counts) | set(self._last_speed)):
            last = self._last_speed.get(fighter_id)
            speed_text = f"{last.peak_speed:.1f} {last.unit}" if last else "—"
            stance = self._stances.get(fighter_id, "?")
            strike = self._last_strike.get(fighter_id)
            strike_text = strike.strike_type.value if strike else "—"
            power = self._last_power.get(fighter_id)
            power_text = f"~{power.score:.0f}" if power else "—"  # estimated, hence ~
            lines.append(
                f"{fighter_id}: {strike_text} {speed_text}  est.pwr {power_text}  "
                f"count {self._candidate_counts[fighter_id]}  stance {stance}"
            )
        pad, line_h = 8, 22
        panel_h = pad * 2 + line_h * len(lines)
        overlay = image[0:panel_h, 0:420]
        overlay[:] = (overlay * 0.35 + np.array(_PANEL_COLOR) * 0.65).astype(np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (pad, pad + line_h * (i + 1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                _TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _accumulate_feet(self, tracked: TrackedPose) -> None:
        heat = self._foot_heat[tracked.fighter_id]
        for foot in _FEET:
            kp = tracked.pose.get(foot)
            if kp is None:
                continue
            bx = min(int(kp.x * _HEATMAP_BINS[0]), _HEATMAP_BINS[0] - 1)
            by = min(int(kp.y * _HEATMAP_BINS[1]), _HEATMAP_BINS[1] - 1)
            heat[by, bx] += 1.0

    def _blend_heatmap(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        combined = np.zeros((_HEATMAP_BINS[1], _HEATMAP_BINS[0]), dtype=np.float32)
        for heat in self._foot_heat.values():
            combined += heat
        if combined.max() <= 0:
            return image
        normalized = (combined / combined.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(
            cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET
        )
        alpha = self._config.heatmap_alpha
        return cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)
