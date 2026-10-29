"""Typed tool results and errors. Nothing here ever carries a stack trace or raw exception text."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    NOT_VERIFIED = "not_verified"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_LOCKED = "verification_locked"
    ALREADY_VERIFIED = "already_verified"
    ORDER_NOT_FOUND = "order_not_found"
    ITEM_NOT_FOUND = "item_not_found"
    NOT_ELIGIBLE = "not_eligible"
    REQUIRES_HUMAN = "requires_human"
    TICKET_ALREADY_OPEN = "ticket_already_open"
    IDEMPOTENCY_KEY_REUSED = "idempotency_key_reused"
    ACTION_NOT_FOUND = "action_not_found"
    ACTION_NOT_PENDING = "action_not_pending"
    ACTION_NOT_CONFIRMED = "action_not_confirmed"
    CONFIRMATION_STALE = "confirmation_stale"
    ALREADY_DONE = "already_done"
    SEARCH_UNAVAILABLE = "search_unavailable"
    INTERNAL_ERROR = "internal_error"


class ToolError(BaseModel):
    code: ErrorCode
    message: str = Field(description="Customer-safe sentence the agent can say or paraphrase.")
    hint: str | None = Field(None, description="Guidance for the agent. Never read aloud.")
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    # True when a write tool returned the stored result of an earlier call with the same key.
    replayed: bool = False


class ToolFailure(Exception):
    """Raised inside tools for expected failures; the dispatcher turns it into a ToolError."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code.value)
        self.error = ToolError(
            code=code, message=message, hint=hint, retryable=retryable, details=details or {}
        )
