"""Events streamed out of a turn. JSON-serializable as-is for the WebSocket and CLI."""

from typing import Literal

from pydantic import BaseModel


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class ToolEvent(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    status: Literal["started", "finished"]
    ok: bool | None = None


class ActionEvent(BaseModel):
    type: Literal["action"] = "action"
    action_id: str
    tool: str
    state: str
    summary: str
    amount_display: str | None = None


class EscalationEvent(BaseModel):
    type: Literal["escalated"] = "escalated"
    reference: str
    reason: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    reply: str
    outcome: Literal["completed", "escalated", "failed"]
    iterations: int


AgentEvent = TokenEvent | ToolEvent | ActionEvent | EscalationEvent | ErrorEvent | DoneEvent
