from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket

_ws_clients: set[WebSocket] = set()


def register_ws(ws: WebSocket) -> None:
    _ws_clients.add(ws)


def unregister_ws(ws: WebSocket) -> None:
    _ws_clients.discard(ws)


async def broadcast(event: str, data: dict[str, Any]) -> None:
    message = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)
