"""A minimal synchronous, in-process event bus.

Engines publish typed events; sinks (overlay UI, storage, report builders)
subscribe by event class. Synchronous dispatch keeps ordering deterministic,
which matters for engines that consume other engines' events (the combination
engine consumes :class:`~combat_vision.events.types.StrikeEvent`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from combat_vision.events.types import Event

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)
Handler = Callable[[Event], None]


class EventBus:
    """Type-keyed publish/subscribe hub.

    Handlers subscribed to a class also receive events of its subclasses is
    *not* supported deliberately — subscribe to concrete event types to keep
    routing explicit.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = defaultdict(list)
        self._history: list[Event] = []
        self._record = False

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Register ``handler`` for events of exactly ``event_type``."""
        self._handlers[event_type].append(handler)  # type: ignore[arg-type]

    def publish(self, event: Event) -> None:
        """Dispatch ``event`` synchronously to all matching handlers."""
        if self._record:
            self._history.append(event)
        for handler in self._handlers[type(event)]:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 — one bad sink must not kill the pipeline
                logger.exception("event handler failed for %r", event)

    def start_recording(self) -> None:
        """Keep every published event in memory (used by review mode)."""
        self._record = True

    @property
    def history(self) -> list[Event]:
        """Events recorded since :meth:`start_recording` was called."""
        return self._history
