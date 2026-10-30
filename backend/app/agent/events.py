"""Events streamed out of a turn. JSON-serializable as-is for the WebSocket and CLI."""

from typing import Literal

from pydantic import BaseModel, Field


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
    # This turn's model usage: prompt_tokens, completion_tokens, llm_calls (for cost reports).
    usage: dict[str, int] = Field(default_factory=dict)
    # The turn's id in turn_metrics (and on its trace span), for the replay view.
    turn_id: str | None = None


AgentEvent = TokenEvent | ToolEvent | ActionEvent | EscalationEvent | ErrorEvent | DoneEvent
