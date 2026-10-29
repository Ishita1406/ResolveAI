"""Agent session state and where it lives (Redis in the app, memory in tests).

A session holds the tool-layer SessionState (verification, pending actions, call history)
plus the model-facing message history. It's serialized as JSON; both stores round-trip it
through JSON so tests exercise the same path as production.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import LockError

from app.tools.context import SessionState

Message = dict[str, Any]


class AgentSession(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    conversation_id: int | None = None
    channel: str = "text"
    created_at: datetime
    updated_at: datetime
    state: SessionState = Field(default_factory=SessionState)
    # Completed turns, oldest first; each is the messages to replay to the model.
    history: list[list[Message]] = Field(default_factory=list)
    # The order the conversation is about, for system-initiated handoffs.
    current_order_number: str | None = None
    escalation_reference: str | None = None


class SessionBusyError(Exception):
    """Another turn is already running for this session."""


class SessionStore(Protocol):
    async def load(self, session_id: str) -> AgentSession | None: ...
    async def save(self, session: AgentSession) -> None: ...
    def lock(self, session_id: str) -> Any: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def load(self, session_id: str) -> AgentSession | None:
        raw = self._data.get(session_id)
        return AgentSession.model_validate_json(raw) if raw else None

    async def save(self, session: AgentSession) -> None:
        self._data[session.session_id] = session.model_dump_json()

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        if lock.locked():
            raise SessionBusyError(session_id)
        async with lock:
            yield


class RedisSessionStore:
    def __init__(
        self,
        redis: Redis,
        *,
        ttl_s: int,
        lock_timeout_s: float,
        prefix: str = "resolveai:session:",
    ) -> None:
        self._redis = redis
        self._ttl_s = ttl_s
        self._lock_timeout_s = lock_timeout_s
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def load(self, session_id: str) -> AgentSession | None:
        raw = await self._redis.get(self._key(session_id))
        return AgentSession.model_validate_json(raw) if raw else None

    async def save(self, session: AgentSession) -> None:
        await self._redis.set(
            self._key(session.session_id), session.model_dump_json(), ex=self._ttl_s
        )

    @asynccontextmanager
    async def lock(self, session_id: str) -> AsyncIterator[None]:
        # Auto-expires so a crashed worker can't hold a session forever.
        lock = self._redis.lock(
            f"{self._key(session_id)}:lock", timeout=self._lock_timeout_s, blocking=False
        )
        if not await lock.acquire():
            raise SessionBusyError(session_id)
        try:
            yield
        finally:
            try:
                await lock.release()
            except LockError:
                pass  # expired while held; the turn timeout is shorter, so this is rare
