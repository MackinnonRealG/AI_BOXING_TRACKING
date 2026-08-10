"""FastAPI + websocket scaffold for the future web frontend.

Run with: ``uvicorn combat_vision.ui.web:app --reload``

The intended production flow: the live pipeline pushes stat snapshots into
:func:`push_stats`; every connected ``/stats`` websocket client receives
them. v1 ships the plumbing with an in-memory latest-snapshot store.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Combat Vision", version="0.1.0")

_latest_stats: dict[str, Any] = {"status": "no session running"}
_clients: list[WebSocket] = []


def push_stats(stats: dict[str, Any]) -> None:
    """Update the latest stats snapshot (called by a pipeline stats sink).

    Rebinds the module global to a fresh dict rather than mutating the
    existing one in place: this is meant to be called from the pipeline
    thread while ``stats_socket`` reads ``_latest_stats`` concurrently on the
    asyncio event loop, and a bare name rebind is a single atomic bytecode
    op — readers always see either the complete old snapshot or the complete
    new one, never a dict caught mid ``clear()``/``update()``.

    TODO: fan out to connected clients from the pipeline thread via an
    asyncio queue instead of relying on client polling.
    """
    global _latest_stats
    _latest_stats = dict(stats)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.websocket("/stats")
async def stats_socket(websocket: WebSocket) -> None:
    """Stream live stats to the web UI.

    Stub behavior: sends the latest snapshot whenever the client sends any
    message (client-driven polling over one socket).
    """
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json(_latest_stats)
    except WebSocketDisconnect:
        pass
    finally:
        # finally (not just the WebSocketDisconnect except) so any other
        # failure — a protocol error, a send on an already-closed socket,
        # anything — still frees the slot instead of leaking it forever.
        if websocket in _clients:
            _clients.remove(websocket)
