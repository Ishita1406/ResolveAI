"""A scripted stand-in for the LLM, and an Agent wired to the test database."""

import asyncio
import copy
import itertools
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm import Completion, LLMEvent, TextDelta, ToolCall
from app.agent.loop import Agent, AgentConfig
from app.agent.prompts import load_prompts
from app.agent.session import InMemorySessionStore
from app.db.seed import ANCHOR
from app.policy.config import get_policy

_ids = itertools.count(1)


@dataclass(frozen=True)
class Pause:
    seconds: float


def say(text: str) -> list:
    """A final reply, streamed word by word."""
    words = text.split(" ")
    deltas = [TextDelta(w if i == len(words) - 1 else w + " ") for i, w in enumerate(words)]
    return [*deltas, Completion(text=text)]


def call(name: str, *, before: str = "", **args: Any) -> list:
    """A tool call, optionally preceded by spoken text ("Let me check.")."""
    tool_call = ToolCall(id=f"call_{next(_ids)}", name=name, arguments=json.dumps(args))
    events: list = [TextDelta(before)] if before else []
    return [*events, Completion(text=before, tool_calls=[tool_call])]


class ScriptedLLM:
    """Plays one scripted step per stream() call. Steps may contain Pause or an exception."""

    def __init__(self, *steps: list, default: list | None = None) -> None:
        self.steps = list(steps)
        self.default = default
        self.calls: list[list[dict[str, Any]]] = []
        self.tools: list[dict[str, Any]] = []
        self.closed = 0
        self._called = asyncio.Event()

    async def stream(self, messages, tools, *, stop=None):
        self.calls.append(copy.deepcopy(messages))
        self.stop = stop
        self._called.set()
        self.tools = tools
        step = self.steps.pop(0) if self.steps else self.default
        assert step is not None, "LLM called more times than scripted"
        try:
            for event in step:
                if isinstance(event, Pause):
                    await asyncio.sleep(event.seconds)
                elif isinstance(event, BaseException):
                    raise event
                else:
                    yield event
        finally:
            self.closed += 1

    async def wait_for_calls(self, n: int) -> None:
        while len(self.calls) < n:
            self._called.clear()
            await self._called.wait()

    def last_messages(self) -> list[dict[str, Any]]:
        return self.calls[-1]


def shared_session_factory(db: AsyncSession):
    @asynccontextmanager
    async def factory():
        yield db  # the test owns the session (and its outer transaction)

    return factory


def make_agent(db: AsyncSession, llm: ScriptedLLM, **config: Any) -> Agent:
    return Agent(
        llm=llm,
        store=InMemorySessionStore(),
        db_factory=shared_session_factory(db),
        prompts=load_prompts("v2"),
        policy=get_policy(),
        config=AgentConfig(**config),
        clock=lambda: ANCHOR,
    )


async def run_turn(agent: Agent, session_id: str, text: str) -> list:
    return [event async for event in agent.respond(session_id, text)]


def tokens(events: list) -> str:
    return "".join(e.text for e in events if e.type == "token")


def of_type(events: list, kind: str) -> list:
    return [e for e in events if e.type == kind]


LLMStep = list[LLMEvent | Pause | BaseException]
