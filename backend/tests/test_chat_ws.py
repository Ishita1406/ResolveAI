"""The /ws/chat protocol, with a fake agent (the real loop is tested in test_agent_loop)."""

import asyncio
import threading
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.agent.events import DoneEvent, TokenEvent
from app.agent.session import AgentSession, InMemorySessionStore
from app.api.chat import MAX_MESSAGE_CHARS
from app.main import create_app


class FakeAgent:
    def __init__(self, *, hang: bool = False):
        self.store = InMemorySessionStore()
        self.hang = hang
        self.cancelled = threading.Event()
        self.received: list[tuple[str, str]] = []

    async def start_session(self):
        now = datetime.now(UTC)
        session = AgentSession(created_at=now, updated_at=now)
        await self.store.save(session)
        return session

    async def respond(self, session_id, text):
        self.received.append((session_id, text))
        try:
            yield TokenEvent(text="Hello ")
            if self.hang:
                await asyncio.Event().wait()
            yield TokenEvent(text="there.")
            yield DoneEvent(reply="Hello there.", outcome="completed", iterations=1)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def _client(agent) -> TestClient:
    app = create_app()  # lifespan not run: no DB or Redis needed
    app.state.agent = agent
    return TestClient(app)


def _until_done(ws) -> list[dict]:
    frames = []
    while True:
        frames.append(ws.receive_json())
        if frames[-1]["type"] in ("done", "error"):
            return frames


def test_without_an_agent_the_socket_says_so():
    with _client(None).websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["code"] == "agent_unavailable"


def test_message_streams_tokens_then_done():
    agent = FakeAgent()
    with _client(agent).websocket_connect("/ws/chat") as ws:
        session = ws.receive_json()
        assert session["type"] == "session"
        ws.send_json({"type": "message", "text": "  hi  "})
        frames = _until_done(ws)
    assert [f["type"] for f in frames] == ["token", "token", "done"]
    assert "".join(f["text"] for f in frames if f["type"] == "token") == "Hello there."
    assert agent.received == [(session["session_id"], "hi")]


def test_resume_existing_session_or_start_a_new_one():
    agent = FakeAgent()
    client = _client(agent)
    with client.websocket_connect("/ws/chat") as ws:
        first = ws.receive_json()["session_id"]
    with client.websocket_connect(f"/ws/chat?session_id={first}") as ws:
        assert ws.receive_json()["session_id"] == first
    with client.websocket_connect("/ws/chat?session_id=unknown") as ws:
        assert ws.receive_json()["session_id"] not in (first, "unknown")


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        ("not json", "invalid_frame"),
        ({"type": "dance"}, "invalid_frame"),
        ({"type": "message", "text": "   "}, "invalid_message"),
        ({"type": "message", "text": 5}, "invalid_message"),
        ({"type": "message", "text": "x" * (MAX_MESSAGE_CHARS + 1)}, "message_too_long"),
    ],
)
def test_bad_frames_get_errors_and_the_socket_stays_open(frame, code):
    with _client(FakeAgent()).websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        if isinstance(frame, str):
            ws.send_text(frame)
        else:
            ws.send_json(frame)
        error = ws.receive_json()
        assert (error["type"], error["code"]) == ("error", code)
        assert error["message"]
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_cancel_stops_the_reply_in_progress():
    agent = FakeAgent(hang=True)
    with _client(agent).websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "message", "text": "hi"})
        assert ws.receive_json() == {"type": "token", "text": "Hello "}
        ws.send_json({"type": "cancel"})
        assert ws.receive_json() == {"type": "cancelled"}
    assert agent.cancelled.is_set()


def test_new_message_interrupts_the_previous_reply():
    agent = FakeAgent(hang=True)
    with _client(agent).websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"type": "message", "text": "first"})
        ws.receive_json()
        agent.hang = False
        ws.send_json({"type": "message", "text": "second"})
        assert ws.receive_json() == {"type": "cancelled"}
        frames = _until_done(ws)
    assert frames[-1]["type"] == "done"
    assert [t for _, t in agent.received] == ["first", "second"]
