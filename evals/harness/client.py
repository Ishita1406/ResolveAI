"""A /ws/chat client speaking the same JSON protocol as the real front end."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import websockets


@dataclass
class TurnResult:
    reply: str
    events: list[dict[str, Any]] = field(default_factory=list)
    done: dict[str, Any] | None = None  # the "done" frame; None if the turn ended in error

    @property
    def escalated(self) -> bool:
        return any(e["type"] == "escalated" for e in self.events)


class ChatClient:
    def __init__(self, url: str, *, turn_timeout_s: float) -> None:
        self.url = url
        self.turn_timeout_s = turn_timeout_s
        self.session_id: str | None = None
        self._ws = None

    async def __aenter__(self) -> "ChatClient":
        self._ws = await websockets.connect(self.url, max_size=2**22)
        first = json.loads(await self._ws.recv())
        if first.get("type") != "session":
            raise RuntimeError(f"expected a session frame, got {first}")
        self.session_id = first["session_id"]
        return self

    async def __aexit__(self, *exc) -> None:
        await self._ws.close()

    async def send(self, text: str) -> TurnResult:
        """Send one customer message and collect frames until the turn is done."""
        await self._ws.send(json.dumps({"type": "message", "text": text}))
        events: list[dict[str, Any]] = []
        async with asyncio.timeout(self.turn_timeout_s):
            while True:
                frame = json.loads(await self._ws.recv())
                events.append(frame)
                if frame["type"] == "done":
                    return TurnResult(reply=frame["reply"], events=events, done=frame)
                if frame["type"] == "error":  # e.g. agent unavailable: no done frame follows
                    return TurnResult(reply="", events=events)
