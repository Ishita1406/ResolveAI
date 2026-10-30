"""Structured JSON logs. Every line carries the conversation it belongs to.

The ids live in context variables, set where a conversation's work starts (the agent's turn,
the voice call, the chat socket). asyncio tasks copy the context when they're created, so
everything a turn starts logs with its ids, without passing them around. Lines logged
outside any conversation (startup, health checks) have null ids.

    {"ts": "...", "level": "WARNING", "logger": "app.agent.loop", "message": "...",
     "conversation_id": 42, "session_id": "...", "turn_id": "...", "trace_id": "...", ...}
"""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from opentelemetry import trace

conversation_id_var: ContextVar[int | None] = ContextVar("conversation_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
turn_id_var: ContextVar[str | None] = ContextVar("turn_id", default=None)

# LogRecord attributes that aren't caller-supplied `extra` fields.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_CONTEXT = ("conversation_id", "session_id", "turn_id", "trace_id", "span_id")


def bind(
    *, conversation_id: int | None = None, session_id: str | None = None, turn_id: str | None = None
) -> None:
    """Attach ids to everything logged from here on in this task (and tasks it starts)."""
    if conversation_id is not None:
        conversation_id_var.set(conversation_id)
    if session_id is not None:
        session_id_var.set(session_id)
    if turn_id is not None:
        turn_id_var.set(turn_id)


@contextmanager
def bound(**ids) -> Iterator[None]:
    """`bind` for a block only."""
    tokens = [
        (var, var.set(value))
        for var, value in (
            (conversation_id_var, ids.get("conversation_id")),
            (session_id_var, ids.get("session_id")),
            (turn_id_var, ids.get("turn_id")),
        )
        if value is not None
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class ContextFilter(logging.Filter):
    """Copies the context ids (and the current span's trace and span ids) onto the record.
    Installed on handlers, so records from any logger get them."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.conversation_id = conversation_id_var.get()
        record.session_id = session_id_var.get()
        record.turn_id = turn_id_var.get()
        span = trace.get_current_span().get_span_context()
        exported = span.is_valid and span.trace_flags.sampled  # not a dropped request span
        record.trace_id = format(span.trace_id, "032x") if exported else None
        record.span_id = format(span.span_id, "016x") if exported else None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _CONTEXT:
            line[key] = getattr(record, key, None)
        for key, value in vars(record).items():
            if key not in _RESERVED and key not in _CONTEXT and not key.startswith("_"):
                line[key] = value  # extra={...} fields
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable lines for a terminal, still with the conversation id."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)s %(name)s [conv=%(conversation_id)s] %(message)s"
        )


_HANDLER_NAME = "resolveai"


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """One stdout handler on the root logger, replacing any (including log_config.json's,
    which only covers uvicorn's lines before the app starts); uvicorn's loggers are routed
    to it, so access and error lines are JSON as well. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
