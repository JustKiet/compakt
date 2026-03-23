"""WebSocket connection manager for live status updates."""

import json
import logging

from fastapi import WebSocket

LOGGER = logging.getLogger("backend.ws_manager")


class ConnectionManager:
    """Manages WebSocket connections keyed by blob_name."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, blob_name: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if blob_name not in self._connections:
            self._connections[blob_name] = set()
        self._connections[blob_name].add(websocket)
        LOGGER.info(
            "WebSocket connected for blob=%s (total=%d)",
            blob_name,
            len(self._connections[blob_name]),
        )

    def disconnect(self, blob_name: str, websocket: WebSocket) -> None:
        if blob_name in self._connections:
            self._connections[blob_name].discard(websocket)
            if not self._connections[blob_name]:
                del self._connections[blob_name]
        LOGGER.info("WebSocket disconnected for blob=%s", blob_name)

    async def broadcast_to_blob(self, blob_name: str, message: dict[str, str | None]) -> None:
        """Send a JSON message to all WebSocket clients subscribed to a blob_name."""
        connections = self._connections.get(blob_name, set())
        if not connections:
            LOGGER.debug("No WebSocket clients for blob=%s, skipping broadcast", blob_name)
            return

        payload = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                LOGGER.warning("Failed to send to WebSocket for blob=%s; marking dead", blob_name)
                dead.append(ws)

        for ws in dead:
            connections.discard(ws)
        if not connections and blob_name in self._connections:
            del self._connections[blob_name]


manager = ConnectionManager()
