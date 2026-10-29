"""WebSocket chat: /ws/chat[?session_id=...]

Client -> server (JSON):
  {"type": "message", "text": "..."}   a customer message (cancels a reply in progress)
  {"type": "cancel"}                   stop the reply in progress (barge-in)
  {"type": "ping"}
Server -> client (JSON):
  {"type": "session", "session_id": "..."}  first frame
  agent events: token | tool | action | escalated | error | done
  {"type": "cancelled"}                after a reply is stopped
  {"type": "pong"}
"""

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await ws.send_json({"type": "error", "code": code, "message": message})


@router.websocket("/ws/chat")
async def chat(ws: WebSocket) -> None:
    await ws.accept()
    agent = getattr(ws.app.state, "agent", None)
    if agent is None:
        await _send_error(ws, "agent_unavailable", "The assistant isn't configured right now.")
        await ws.close(code=1013)  # try again later
        return

    session_id = ws.query_params.get("session_id")
    if session_id is None or await agent.store.load(session_id) is None:
        session_id = (await agent.start_session()).session_id
    await ws.send_json({"type": "session", "session_id": session_id})

    current: asyncio.Task | None = None

    async def reply(text: str) -> None:
        async for event in agent.respond(session_id, text):
            await ws.send_json(event.model_dump(mode="json"))

    async def stop_current() -> bool:
        nonlocal current
        if current is None or current.done():
            return False
        current.cancel()
        with suppress(asyncio.CancelledError):
            await current
        return True

    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame = json.loads(raw)
                kind = frame.get("type") if isinstance(frame, dict) else None
            except json.JSONDecodeError:
                kind = None
            if kind == "message":
                text = frame.get("text")
                if not isinstance(text, str) or not text.strip():
                    await _send_error(ws, "invalid_message", "Message text is required.")
                    continue
                if len(text) > MAX_MESSAGE_CHARS:
                    await _send_error(ws, "message_too_long", "That message is too long.")
                    continue
                if await stop_current():  # a new message interrupts the reply in progress
                    await ws.send_json({"type": "cancelled"})
                current = asyncio.create_task(reply(text.strip()))
            elif kind == "cancel":
                if await stop_current():
                    await ws.send_json({"type": "cancelled"})
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
            else:
                await _send_error(ws, "invalid_frame", "Unknown or malformed frame.")
    except WebSocketDisconnect:
        pass
    finally:
        await stop_current()
