"""ui/web.py tests: push_stats atomicity and stats_socket client cleanup.

No httpx/TestClient dependency needed — stats_socket is driven directly with
a minimal async double, since the behavior under test (the try/finally
cleanup, the atomic dict rebind) doesn't need a real ASGI connection.
"""

from __future__ import annotations

import asyncio

import pytest

from combat_vision.ui import web
from combat_vision.ui.web import WebSocketDisconnect, push_stats, stats_socket


class _FakeWebSocket:
    """Minimal async double for a Starlette WebSocket."""

    def __init__(self, messages: list[str] | Exception) -> None:
        self._messages = list(messages) if isinstance(messages, list) else messages
        self.sent: list[dict] = []

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if isinstance(self._messages, Exception):
            raise self._messages
        if not self._messages:
            raise WebSocketDisconnect
        return self._messages.pop(0)

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


@pytest.fixture(autouse=True)
def _reset_web_state():
    web._clients.clear()
    web._latest_stats = {"status": "no session running"}
    yield
    web._clients.clear()


def test_push_stats_rebinds_instead_of_mutating() -> None:
    """A reader holding the old snapshot must never see it change underfoot.

    push_stats used to do ``_latest_stats.clear(); _latest_stats.update(...)``
    — two separate mutations a concurrent reader could observe mid-way
    (empty, or partially updated). Rebinding to a fresh dict means any
    reference taken before the call stays exactly as it was.
    """
    old_ref = web._latest_stats
    push_stats({"punches": 5})

    assert old_ref == {"status": "no session running"}  # untouched by the update
    assert web._latest_stats == {"punches": 5}
    assert web._latest_stats is not old_ref


def test_client_removed_on_normal_disconnect() -> None:
    """A client that disconnects cleanly is removed from the roster."""
    ws = _FakeWebSocket(["ping"])
    asyncio.run(stats_socket(ws))
    assert ws not in web._clients


def test_client_removed_even_on_unexpected_exception() -> None:
    """Any failure — not just WebSocketDisconnect — must free the client slot.

    The old code only removed the client inside `except WebSocketDisconnect`,
    so any other exception left a dead reference in `_clients` forever.
    """
    ws = _FakeWebSocket(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        asyncio.run(stats_socket(ws))
    assert ws not in web._clients
