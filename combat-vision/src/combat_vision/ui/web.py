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

    TODO: fan out to connected clients from the pipeline thread via an
    asyncio queue instead of relying on client polling.
    """
    _latest_stats.clear()
    _latest_stats.update(stats)


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
        _clients.remove(websocket)
