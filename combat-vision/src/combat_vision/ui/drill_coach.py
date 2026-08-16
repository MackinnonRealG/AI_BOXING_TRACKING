"""Guided drill/practice mode: on-screen combo prompts with a countdown.

Turns live mode from a passive logger into something that prompts a rep and
grades it, using only what's already flowing through the bus — no new pose
analysis. A small state machine (idle -> countdown -> active -> result ->
idle) driven by two inputs: a wall-clock tick each rendered frame, and the
:class:`~combat_vision.events.types.StrikeEvent` stream the strike
classifier already publishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from combat_vision.drills import Drill
from combat_vision.events.types import FighterId, StrikeEvent


class _State(StrEnum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    ACTIVE = "active"
    RESULT = "result"


@dataclass
class DrillCoach:
    """Owns one active drill attempt at a time, for one fighter."""

    countdown_s: float = 3.0
    result_hold_s: float = 2.0

    _drill: Drill | None = field(default=None, init=False)
    _fighter_id: FighterId | None = field(default=None, init=False)
    _state: _State = field(default=_State.IDLE, init=False)
    _state_since_s: float = field(default=0.0, init=False)
    _progress: int = field(default=0, init=False)
    _result_ok: bool | None = field(default=None, init=False)

    @property
    def active(self) -> bool:
        """True whenever a drill is counting down, running, or showing a result."""
        return self._state != _State.IDLE

    @property
    def fighter_id(self) -> FighterId | None:
        """Which fighter the current (or most recent) drill targets, if any."""
        return self._fighter_id

    def start(self, drill: Drill, fighter_id: FighterId, now_s: float) -> None:
        """Begin a countdown for ``fighter_id`` to throw ``drill``."""
        self._drill = drill
        self._fighter_id = fighter_id
        self._state = _State.COUNTDOWN
        self._state_since_s = now_s
        self._progress = 0
        self._result_ok = None

    def stop(self) -> None:
        """Cancel and clear the current drill, if any."""
        self._drill = None
        self._fighter_id = None
        self._state = _State.IDLE
        self._progress = 0
        self._result_ok = None

    def tick(self, now_s: float) -> None:
        """Advance the countdown/result timers. Call once per rendered frame."""
        if self._state == _State.COUNTDOWN and now_s - self._state_since_s >= self.countdown_s:
            self._state = _State.ACTIVE
            self._state_since_s = now_s
        elif self._state == _State.RESULT and now_s - self._state_since_s >= self.result_hold_s:
            self._state = _State.IDLE

    def on_strike(self, event: StrikeEvent) -> None:
        """Score one strike against the expected sequence, if a drill is active."""
        if self._state != _State.ACTIVE or event.fighter_id != self._fighter_id:
            return
        assert self._drill is not None
        expected = self._drill.sequence[self._progress]
        if event.strike_type != expected:
            self._finish(ok=False, at_s=event.timestamp_s)
            return
        self._progress += 1
        if self._progress == len(self._drill.sequence):
            self._finish(ok=True, at_s=event.timestamp_s)

    def _finish(self, ok: bool, at_s: float) -> None:
        self._result_ok = ok
        self._state = _State.RESULT
        self._state_since_s = at_s

    def prompt(self, now_s: float) -> str | None:
        """The on-screen line for the current state, or None when idle."""
        if self._drill is None or self._state == _State.IDLE:
            return None
        label = self._drill.name.upper()

        if self._state == _State.COUNTDOWN:
            remaining = max(0, int(self.countdown_s - (now_s - self._state_since_s)) + 1)
            return f"DRILL {label} — ready in {remaining}..."

        if self._state == _State.ACTIVE:
            thrown = [t.value.upper() for t in self._drill.sequence[: self._progress]]
            next_strike = self._drill.sequence[self._progress].value.upper()
            thrown_text = "-".join([*thrown, f"[{next_strike}]"])
            return f"DRILL {label} — throw: {thrown_text}"

        # RESULT
        verdict = "CLEAN!" if self._result_ok else "broke the sequence — reset"
        return f"DRILL {label} — {verdict}"
