"""OpenCV live HUD: fighter cards, guidance, rounds, calibration, toasts.

Everything on screen is driven by the event bus plus the per-frame pose
list, so a future web frontend can replace this class without touching the
pipeline.

Keys (all shown on screen):
    ``r`` start/end round     ``c`` click-two-points calibration
    ``h`` foot heat map       ``m`` mirror view
    ``f`` fullscreen          ``t`` tracker backend toggle
    ``s`` toggle boxing/kickboxing — switches which strikes/faults every
          engine monitors immediately, no restart
    ``d`` start/stop a guided drill for fighter A (cycles through combos)
    ``q`` quit (the session is saved and reported by the caller)
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from combat_vision.calibration import Calibration
from combat_vision.capture.base import TimestampedFrame
from combat_vision.drills import Drill, drills_for_profile
from combat_vision.events.bus import EventBus
from combat_vision.events.types import (
    SKELETON_EDGES,
    BalanceFaultEvent,
    ElbowStateEvent,
    FighterRelabeledEvent,
    GuardStateEvent,
    KeypointName,
    KneeBendStateEvent,
    LegDriveFaultEvent,
    Limb,
    PowerEstimateEvent,
    RotationFaultEvent,
    SpeedPeakEvent,
    StanceSample,
    StepEvent,
    StrikeEvent,
    TrackedPose,
)
from combat_vision.sports.switchable import SwitchableSportProfile
from combat_vision.tracking import SwitchableTracker
from combat_vision.ui.drill_coach import DrillCoach
from combat_vision.utils import geometry
from combat_vision.utils.config import UiConfig

_FIGHTER_COLORS: dict[str, tuple[int, int, int]] = {
    "A": (80, 200, 80),   # BGR green
    "B": (80, 120, 255),  # BGR orange
}
_NEUTRAL = (200, 200, 200)
_TEXT = (240, 240, 240)
_ACCENT = (90, 210, 255)   # yellow-ish guidance
_PANEL = (25, 25, 25)
_HEATMAP_BINS = (64, 36)
_FEET = (KeypointName.LEFT_ANKLE, KeypointName.RIGHT_ANKLE)
_LIMBS_FOR_GUIDANCE = (
    KeypointName.LEFT_WRIST,
    KeypointName.RIGHT_WRIST,
    KeypointName.LEFT_ANKLE,
    KeypointName.RIGHT_ANKLE,
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE_H = 24
_FIGHTER_STALE_S = 1.0  # hide a fighter card this long after they leave frame
_FAULT_NOTE_DURATION_S = 2.0  # how long a per-rep fault cue (e.g. rotation) stays on screen


@dataclass
class _FighterHud:
    """Everything the HUD shows for one fighter."""

    last_speed: SpeedPeakEvent | None = None
    last_strike: StrikeEvent | None = None
    last_power: PowerEstimateEvent | None = None
    strike_counts: Counter = field(default_factory=Counter)
    candidates: int = 0
    steps: int = 0
    stance: str = "?"
    guard_up: dict[Limb, bool] = field(default_factory=dict)
    elbow_tucked: dict[Limb, bool] = field(default_factory=dict)
    knees_locked: bool | None = None
    fault_notes: list[tuple[str, float]] = field(default_factory=list)
    last_seen_s: float = -1.0e9


class LiveOverlay:
    """Draws the live view and owns per-session display and round state."""

    def __init__(
        self,
        bus: EventBus,
        config: UiConfig,
        tracker: SwitchableTracker | None = None,
        calibration: Calibration | None = None,
        sport_profile: SwitchableSportProfile | None = None,
        names: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._tracker = tracker
        self._calibration = calibration
        self._sport_profile = sport_profile
        self._names = names or {}

        self._show_heatmap = False
        self._mirror = config.mirror
        self._fullscreen = False
        self._window_ready = False

        self._fighters: dict[str, _FighterHud] = defaultdict(_FighterHud)
        self._foot_heat: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((_HEATMAP_BINS[1], _HEATMAP_BINS[0]), dtype=np.float32)
        )
        self._toasts: list[tuple[StrikeEvent, float]] = []
        self._render_times: deque[float] = deque(maxlen=30)

        # Rounds: (number, start_s, end_s|None); the caller persists them.
        self.rounds: list[tuple[int, float, float | None]] = []
        self._round_open = False
        self._now_s = 0.0

        # Calibration mode state machine: off -> picking -> typing -> off.
        self._cal_mode = "off"
        self._cal_points: list[tuple[int, int]] = []
        self._cal_buffer = ""

        self._drills: tuple[Drill, ...] = ()
        self._refresh_drills()
        self._drill_index = 0
        self._drill_coach = DrillCoach()

        bus.subscribe(SpeedPeakEvent, self._on_speed_peak)
        bus.subscribe(StrikeEvent, self._on_strike)
        bus.subscribe(PowerEstimateEvent, self._on_power)
        bus.subscribe(StanceSample, self._on_stance)
        bus.subscribe(StepEvent, self._on_step)
        bus.subscribe(GuardStateEvent, self._on_guard)
        bus.subscribe(ElbowStateEvent, self._on_elbow)
        bus.subscribe(RotationFaultEvent, self._on_rotation_fault)
        bus.subscribe(KneeBendStateEvent, self._on_knee_bend)
        bus.subscribe(LegDriveFaultEvent, self._on_leg_drive_fault)
        bus.subscribe(BalanceFaultEvent, self._on_balance_fault)
        bus.subscribe(FighterRelabeledEvent, self._on_relabeled)

    def _display_name(self, fighter_id: str) -> str:
        """The fighter's given name, or 'Fighter <label>' when unnamed."""
        return self._names.get(fighter_id, f"Fighter {fighter_id}")

    # -- event intake ------------------------------------------------------

    def _on_speed_peak(self, event: SpeedPeakEvent) -> None:
        hud = self._fighters[event.fighter_id]
        hud.last_speed = event
        hud.candidates += 1

    def _on_strike(self, event: StrikeEvent) -> None:
        hud = self._fighters[event.fighter_id]
        hud.last_strike = event
        hud.strike_counts[event.strike_type.value] += 1
        self._toasts.append((event, time.monotonic()))
        self._drill_coach.on_strike(event)

    def _on_power(self, event: PowerEstimateEvent) -> None:
        self._fighters[event.fighter_id].last_power = event

    def _on_stance(self, event: StanceSample) -> None:
        self._fighters[event.fighter_id].stance = event.stance.value

    def _on_step(self, event: StepEvent) -> None:
        self._fighters[event.fighter_id].steps += 1

    def _on_guard(self, event: GuardStateEvent) -> None:
        self._fighters[event.fighter_id].guard_up[event.hand] = event.guard_up

    def _on_elbow(self, event: ElbowStateEvent) -> None:
        self._fighters[event.fighter_id].elbow_tucked[event.elbow] = event.tucked

    def _on_rotation_fault(self, event: RotationFaultEvent) -> None:
        self._add_fault_note(
            event.fighter_id,
            f"that {event.limb.value.replace('_', ' ')} was all arm — turn your hips!",
        )

    def _on_knee_bend(self, event: KneeBendStateEvent) -> None:
        self._fighters[event.fighter_id].knees_locked = event.locked

    def _on_leg_drive_fault(self, event: LegDriveFaultEvent) -> None:
        self._add_fault_note(
            event.fighter_id,
            f"that {event.limb.value.replace('_', ' ')} had no leg drive — "
            "bend your knees and push!",
        )

    def _on_balance_fault(self, event: BalanceFaultEvent) -> None:
        self._add_fault_note(
            event.fighter_id,
            f"that {event.limb.value.replace('_', ' ')} wobbled on your base leg — "
            "stay grounded on the standing foot!",
        )

    def _add_fault_note(self, fighter_id: str, message: str) -> None:
        """Queue a transient per-rep coaching cue, replacing any prior one."""
        self._fighters[fighter_id].fault_notes = [(message, time.monotonic())]

    def _on_relabeled(self, event: FighterRelabeledEvent) -> None:
        """Drop everything accumulated under a label now held by someone else.

        Counts, stance, last strike/power and the foot heat map are all derived
        from the *previous* person's frames; carrying them over would credit
        their punches, steps and footwork to whoever just walked into the slot.
        Pending toasts are dropped for the same reason — they would otherwise
        pop the departed fighter's last strike over the new fighter's head.

        The entries are deleted rather than zeroed: both containers are
        ``defaultdict`` s, so the next event or rendered frame recreates them
        clean, and a fighter who never returns stops occupying memory.

        ``self._names`` is deliberately left alone. It is operator-supplied
        configuration for the *label* ("slot A is Dana"), not state derived
        from the video, so this code is not the right place to decide it has
        become wrong.
        """
        self._fighters.pop(event.fighter_id, None)
        self._foot_heat.pop(event.fighter_id, None)
        self._toasts = [t for t in self._toasts if t[0].fighter_id != event.fighter_id]
        if self._drill_coach.fighter_id == event.fighter_id:
            self._drill_coach.stop()  # the drill was targeting whoever just left

    # -- main entry --------------------------------------------------------

    def render(self, frame: TimestampedFrame, poses: list[TrackedPose]) -> bool:
        """Draw one frame; return False when the user quits."""
        self._now_s = frame.timestamp_s
        self._render_times.append(time.monotonic())
        self._drill_coach.tick(frame.timestamp_s)
        image = cv2.flip(frame.image, 1) if self._mirror else frame.image
        h, w = image.shape[:2]
        self._ensure_window(w, h)

        for tracked in poses:
            self._fighters[tracked.fighter_id].last_seen_s = frame.timestamp_s
            self._accumulate_feet(tracked)
        if self._show_heatmap:
            image = self._blend_heatmap(image)

        if self._config.draw_skeleton:
            for tracked in poses:
                self._draw_skeleton(image, tracked, w, h)

        self._draw_toasts(image, poses, w, h)
        self._draw_status_strip(image, w)
        self._draw_guidance(image, poses, w)
        self._draw_fighter_cards(image, w, h)
        self._draw_calibration(image)

        cv2.imshow(self._config.window_name, image)
        return self._handle_key(cv2.waitKey(1) & 0xFF, w)

    def close(self) -> None:
        """Close any open round and destroy the display window."""
        if self._round_open and self.rounds:
            number, start_s, _ = self.rounds[-1]
            self.rounds[-1] = (number, start_s, self._now_s)
            self._round_open = False
        cv2.destroyAllWindows()

    # -- keys & window -----------------------------------------------------

    def _ensure_window(self, w: int, h: int) -> None:
        if self._window_ready:
            return
        cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._config.window_name, w, h)
        cv2.setMouseCallback(self._config.window_name, self._on_mouse)
        self._window_ready = True

    def _handle_key(self, key: int, frame_width: int) -> bool:
        if self._cal_mode == "typing":
            return self._handle_calibration_key(key, frame_width)
        if key == ord("q"):
            return False
        if key == ord("h"):
            self._show_heatmap = not self._show_heatmap
        elif key == ord("t") and self._tracker is not None:
            self._tracker.toggle()
        elif key == ord("m"):
            self._mirror = not self._mirror
        elif key == ord("f"):
            self._fullscreen = not self._fullscreen
            cv2.setWindowProperty(
                self._config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self._fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord("r"):
            self._toggle_round()
        elif key == ord("c") and self._calibration is not None:
            self._cal_mode = "picking"
            self._cal_points = []
            self._cal_buffer = ""
        elif key == ord("d"):
            self._toggle_drill()
        elif key == ord("s") and self._sport_profile is not None:
            self._toggle_sport()
        elif key == 27 and self._cal_mode != "off":  # esc cancels calibration
            self._cal_mode = "off"
        return True

    def _toggle_sport(self) -> None:
        """Flip between boxing and kickboxing, live.

        Every engine holds the same :class:`SwitchableSportProfile` instance
        and reads its properties at use-time, so this takes effect starting
        on the next frame — no pipeline rebuild. Kickboxing is a strict
        superset of boxing's strike types here, so this is one toggle
        between two modes, not two independent switches: there's no
        meaningful state where "both on" or "both off" means anything.
        """
        assert self._sport_profile is not None
        other = "kickboxing" if self._sport_profile.name == "boxing" else "boxing"
        self._sport_profile.switch(other)
        self._refresh_drills()
        self._drill_coach.stop()  # the old drill list may not exist in the new sport

    def _refresh_drills(self) -> None:
        """Recompute the drill list for whichever sport is now active."""
        self._drills = drills_for_profile(self._sport_profile) if self._sport_profile else ()

    def _toggle_drill(self) -> None:
        """Stop the running drill, or start the next one in rotation.

        Targets whichever single fighter is currently in frame — solo
        training shouldn't silently target label "A" if you happen to be
        tracked as "B" today. With both (or neither) fighter present, falls
        back to "A".
        """
        if self._drill_coach.active:
            self._drill_coach.stop()
            return
        if not self._drills:
            return
        visible = [
            fid
            for fid, hud in self._fighters.items()
            if self._now_s - hud.last_seen_s <= _FIGHTER_STALE_S
        ]
        target = visible[0] if len(visible) == 1 else "A"
        drill = self._drills[self._drill_index % len(self._drills)]
        self._drill_index += 1
        self._drill_coach.start(drill, fighter_id=target, now_s=self._now_s)

    def _toggle_round(self) -> None:
        if self._round_open:
            number, start_s, _ = self.rounds[-1]
            self.rounds[-1] = (number, start_s, self._now_s)
            self._round_open = False
        else:
            self.rounds.append((len(self.rounds) + 1, self._now_s, None))
            self._round_open = True

    # -- calibration mode --------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self._cal_mode == "picking":
            self._cal_points.append((x, y))
            if len(self._cal_points) == 2:
                self._cal_mode = "typing"

    def _handle_calibration_key(self, key: int, frame_width: int) -> bool:
        """Digit entry for the real-world length of the clicked segment."""
        if key in (13, 10):  # enter commits
            self._commit_calibration()
        elif key == 27:  # esc cancels
            self._cal_mode = "off"
        elif key in (8, 127):  # backspace
            self._cal_buffer = self._cal_buffer[:-1]
        elif key != 255 and (chr(key).isdigit() or chr(key) == "."):
            self._cal_buffer += chr(key)
        return True

    def _commit_calibration(self) -> None:
        assert self._calibration is not None
        try:
            metres = float(self._cal_buffer)
        except ValueError:
            self._cal_mode = "off"
            return
        # Mirroring preserves distances, so display coords are fine here.
        px = geometry.distance(
            (float(self._cal_points[0][0]), float(self._cal_points[0][1])),
            (float(self._cal_points[1][0]), float(self._cal_points[1][1])),
        )
        if metres > 0 and px > 0:
            self._calibration.set_scale(metres / px)
        self._cal_mode = "off"

    def _draw_calibration(self, image: np.ndarray) -> None:
        if self._cal_mode == "off":
            return
        for point in self._cal_points:
            cv2.drawMarker(image, point, _ACCENT, cv2.MARKER_CROSS, 18, 2)
        if len(self._cal_points) == 2:
            cv2.line(image, self._cal_points[0], self._cal_points[1], _ACCENT, 1, cv2.LINE_AA)
        if self._cal_mode == "picking":
            prompt = "CALIBRATION: click two points a known distance apart"
        else:
            prompt = (
                f"CALIBRATION: type distance in metres: {self._cal_buffer}_"
                "  (enter=ok, esc=cancel)"
            )
        h = image.shape[0]
        self._text_block(image, [prompt], 10, h - 40, color=_ACCENT)

    # -- HUD sections ------------------------------------------------------

    def _draw_status_strip(self, image: np.ndarray, w: int) -> None:
        fps = self._fps()
        if self._round_open:
            number, start_s, _ = self.rounds[-1]
            round_text = f"R{number} {_clock(self._now_s - start_s)} LIVE"
        elif self.rounds:
            round_text = f"{len(self.rounds)} rounds done — [r] next"
        else:
            round_text = "[r] start round"
        if self._calibration is not None and self._calibration.is_calibrated:
            units = "units: m/s"
        else:
            units = "units: px/s — [c] calibrate"
        tracker = f"tracker: {self._tracker.active_name}" if self._tracker else ""
        sport = self._sport_profile.name.upper() if self._sport_profile else "?"
        line1 = (
            f"{sport}  |  {_clock(self._now_s)}  |  {round_text}  |  "
            f"{fps:.0f} FPS  |  {tracker}  |  {units}"
        )
        line2 = (
            "[r] round  [c] calibrate  [h] heatmap  [m] mirror  [f] fullscreen  "
            "[t] tracker  [s] sport  [d] drill  [q] quit+save"
        )
        self._text_block(image, [line1, line2], 10, 8, width=w - 20)

    def _draw_guidance(self, image: np.ndarray, poses: list[TrackedPose], w: int) -> None:
        message = self._guidance(poses)
        if message:
            self._text_block(image, [message], 10, 8 + 2 * _LINE_H + 14, color=_ACCENT)

    def _guidance(self, poses: list[TrackedPose]) -> str | None:
        if not poses:
            return "No fighter detected — stand back so your full body is in view"
        for tracked in poses:
            vis = [
                kp.visibility
                for name in _LIMBS_FOR_GUIDANCE
                if (kp := tracked.pose.get(name)) is not None
            ]
            if not vis or sum(vis) / len(vis) < self._config.min_keypoint_visibility:
                return (
                    f"Fighter {tracked.fighter_id}: hands/feet not visible — "
                    "step back until your whole body is in frame"
                )
        drill_prompt = self._drill_coach.prompt(self._now_s)
        if drill_prompt is not None:
            return drill_prompt
        for tracked in poses:
            note = self._fresh_fault_note(tracked.fighter_id)
            if note is not None:
                return note
        for tracked in poses:
            dropped = self._dropped_hand(tracked.fighter_id)
            if dropped is not None:
                return (
                    f"Fighter {tracked.fighter_id}: {dropped.value.replace('_', ' ')} "
                    "dropping — keep your guard up!"
                )
        for tracked in poses:
            flared = self._flared_elbow(tracked.fighter_id)
            if flared is not None:
                return (
                    f"Fighter {tracked.fighter_id}: {flared.value.replace('_', ' ')} elbow "
                    "flared out — tuck it back to your ribs"
                )
        for tracked in poses:
            if self._fighters[tracked.fighter_id].knees_locked:
                return (
                    f"Fighter {tracked.fighter_id}: knees locked out — "
                    "stay loose, bend your knees"
                )
        return None

    def _dropped_hand(self, fighter_id: str) -> Limb | None:
        """The first hand currently flagged guard-down for this fighter, if any."""
        guard = self._fighters[fighter_id].guard_up
        for hand in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            if guard.get(hand) is False:
                return hand
        return None

    def _flared_elbow(self, fighter_id: str) -> Limb | None:
        """The first arm currently flagged elbow-flared for this fighter, if any."""
        elbows = self._fighters[fighter_id].elbow_tucked
        for arm in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
            if elbows.get(arm) is False:
                return arm
        return None

    def _fresh_fault_note(self, fighter_id: str) -> str | None:
        """The most recent per-rep fault cue for this fighter, or None once it expires."""
        notes = self._fighters[fighter_id].fault_notes
        if not notes:
            return None
        message, seen_at = notes[-1]
        if time.monotonic() - seen_at > _FAULT_NOTE_DURATION_S:
            return None
        return f"Fighter {fighter_id}: {message}"

    def _draw_fighter_cards(self, image: np.ndarray, w: int, h: int) -> None:
        active = sorted(
            fid
            for fid, hud in self._fighters.items()
            if self._now_s - hud.last_seen_s <= _FIGHTER_STALE_S
        )
        card_w = 360
        for fighter_id in active:
            hud = self._fighters[fighter_id]
            color = _FIGHTER_COLORS.get(fighter_id, _NEUTRAL)
            x = 10 if fighter_id == "A" or len(active) == 1 else w - card_w - 10
            y = h - 6 * _LINE_H - 26

            if hud.last_strike is not None:
                strike = hud.last_strike
                power = f"  pwr ~{hud.last_power.score:.0f}" if hud.last_power else ""
                last_line = (
                    f"last: {strike.strike_type.value.upper()} "
                    f"{_fmt_speed(strike.speed, strike.unit.value)}{power}"
                )
            elif hud.last_speed is not None:
                last_line = (
                    f"last: {_fmt_speed(hud.last_speed.peak_speed, hud.last_speed.unit.value)}"
                )
            else:
                last_line = "last: — throw a punch!"
            top = hud.strike_counts.most_common(3)
            counts_line = "  ".join(f"{name} x{count}" for name, count in top) or "no strikes yet"
            lines = [
                f"{self._display_name(fighter_id).upper()} ({fighter_id})   "
                f"stance: {hud.stance}",
                f"{self._guard_text(hud)}   {self._elbow_text(hud)}   {self._knee_text(hud)}",
                last_line,
                f"punches: {hud.candidates}   steps: {hud.steps}",
                counts_line,
            ]
            self._text_block(image, lines, x, y, width=card_w, title_color=color)

    def _guard_text(self, hud: _FighterHud) -> str:
        """Compact per-hand guard indicator, e.g. 'guard: L^ Rv'."""
        parts = [
            f"{label}{'^' if hud.guard_up[hand] else 'v'}"
            for hand, label in ((Limb.LEFT_HAND, "L"), (Limb.RIGHT_HAND, "R"))
            if hand in hud.guard_up
        ]
        return "guard: " + " ".join(parts) if parts else "guard: --"

    def _elbow_text(self, hud: _FighterHud) -> str:
        """Compact per-arm elbow-tuck indicator, e.g. 'elbows: L< R>'."""
        parts = [
            f"{label}{'<' if hud.elbow_tucked[arm] else '>'}"
            for arm, label in ((Limb.LEFT_HAND, "L"), (Limb.RIGHT_HAND, "R"))
            if arm in hud.elbow_tucked
        ]
        return "elbows: " + " ".join(parts) if parts else "elbows: --"

    def _knee_text(self, hud: _FighterHud) -> str:
        """Compact knee-posture indicator: 'knees: bent', 'locked', or unknown."""
        if hud.knees_locked is None:
            return "knees: --"
        return "knees: locked" if hud.knees_locked else "knees: bent"

    def _draw_toasts(
        self, image: np.ndarray, poses: list[TrackedPose], w: int, h: int
    ) -> None:
        now = time.monotonic()
        self._toasts = [
            t for t in self._toasts if now - t[1] <= self._config.toast_duration_s
        ]
        nose_px: dict[str, tuple[int, int]] = {}
        for tracked in poses:
            nose = tracked.pose.get(KeypointName.NOSE)
            if nose is not None:
                nose_px[tracked.fighter_id] = self._to_px(nose.x, nose.y, w, h)
        for event, _ in self._toasts:
            text = f"{event.strike_type.value.upper()}  {_fmt_speed(event.speed, event.unit.value)}"
            anchor = nose_px.get(event.fighter_id, (w // 2, h // 3))
            position = (max(10, anchor[0] - 90), max(40, anchor[1] - 60))
            color = _FIGHTER_COLORS.get(event.fighter_id, _NEUTRAL)
            cv2.putText(image, text, position, _FONT, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, position, _FONT, 0.9, color, 2, cv2.LINE_AA)

    # -- skeleton & heat map ----------------------------------------------

    def _to_px(self, x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
        """Normalized pose coords -> display pixels, honoring mirror mode."""
        x = 1.0 - x_norm if self._mirror else x_norm
        return (int(x * w), int(y_norm * h))

    def _draw_skeleton(self, image: np.ndarray, tracked: TrackedPose, w: int, h: int) -> None:
        color = _FIGHTER_COLORS.get(tracked.fighter_id, _NEUTRAL)
        min_vis = self._config.min_keypoint_visibility
        pts: dict[KeypointName, tuple[int, int]] = {}
        for name, kp in tracked.pose.keypoints.items():
            if kp.visibility >= min_vis:  # draw only what the model truly sees
                pts[name] = self._to_px(kp.x, kp.y, w, h)
        for a, b in SKELETON_EDGES:
            if a in pts and b in pts:
                cv2.line(image, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for point in pts.values():
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
        nose = pts.get(KeypointName.NOSE)
        if nose is not None:
            cv2.putText(
                image,
                self._display_name(tracked.fighter_id),
                (nose[0] - 40, nose[1] - 30),
                _FONT,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    def _accumulate_feet(self, tracked: TrackedPose) -> None:
        heat = self._foot_heat[tracked.fighter_id]
        for foot in _FEET:
            kp = tracked.pose.get(foot)
            if kp is None or kp.visibility < self._config.min_keypoint_visibility:
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
        if self._mirror:
            combined = np.fliplr(combined)
        normalized = (combined / combined.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(
            cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET
        )
        alpha = self._config.heatmap_alpha
        return cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)

    # -- drawing utilities -------------------------------------------------

    def _text_block(
        self,
        image: np.ndarray,
        lines: list[str],
        x: int,
        y: int,
        width: int | None = None,
        color: tuple[int, int, int] = _TEXT,
        title_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Semi-transparent panel with one or more lines of text."""
        pad = 8
        block_w = width if width is not None else 8 + max(len(s) for s in lines) * 11
        block_h = pad * 2 + _LINE_H * len(lines)
        y2 = min(y + block_h, image.shape[0])
        x2 = min(x + block_w, image.shape[1])
        region = image[y:y2, x:x2]
        region[:] = (region * 0.35 + np.array(_PANEL) * 0.65).astype(np.uint8)
        for i, line in enumerate(lines):
            line_color = title_color if (i == 0 and title_color is not None) else color
            cv2.putText(
                image,
                line,
                (x + pad, y + pad + _LINE_H * (i + 1) - 8),
                _FONT,
                0.55,
                line_color,
                1,
                cv2.LINE_AA,
            )

    def _fps(self) -> float:
        if len(self._render_times) < 2:
            return 0.0
        span = self._render_times[-1] - self._render_times[0]
        return (len(self._render_times) - 1) / span if span > 0 else 0.0


def _clock(seconds: float) -> str:
    """mm:ss formatting."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _fmt_speed(value: float, unit: str) -> str:
    """Human-friendly speed: 1 decimal for m/s, whole numbers for px/s."""
    return f"{value:.1f} {unit}" if unit == "m/s" else f"{value:.0f} {unit}"
