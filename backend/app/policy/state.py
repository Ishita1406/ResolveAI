"""Confirmation state machine for proposed write actions.

    proposed ──> awaiting_confirmation ──> confirmed ──> executed
        │               │      │               │  │
        └─> rejected    │      └─> expired     │  └─> rejected (re-check failed at execution)
                        └─> cancelled <────────┘

Write tools run only for an action in `confirmed` (or `executed`, which replays the stored
result). The dispatcher enforces this; see app.tools.registry.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.policy.config import Confirmation


class ActionState(StrEnum):
    PROPOSED = "proposed"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


ALLOWED_TRANSITIONS: dict[ActionState, frozenset[ActionState]] = {
    ActionState.PROPOSED: frozenset({ActionState.AWAITING_CONFIRMATION, ActionState.REJECTED}),
    ActionState.AWAITING_CONFIRMATION: frozenset(
        {ActionState.CONFIRMED, ActionState.CANCELLED, ActionState.EXPIRED}
    ),
    ActionState.CONFIRMED: frozenset(
        {ActionState.EXECUTED, ActionState.REJECTED, ActionState.CANCELLED}
    ),
}
TERMINAL_STATES = frozenset(set(ActionState) - set(ALLOWED_TRANSITIONS))
RUNNABLE_STATES = frozenset({ActionState.CONFIRMED, ActionState.EXECUTED})


class InvalidTransitionError(Exception):
    pass


def args_hash(args: BaseModel) -> str:
    """Canonical hash of a write tool's arguments, excluding the idempotency key."""
    payload = args.model_dump(mode="json", exclude={"idempotency_key"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Transition:
    source: ActionState
    target: ActionState
    at: datetime
    turn: int
    note: str | None


@dataclass
class PendingAction:
    id: str
    tool: str
    arguments: dict[str, Any]  # validated, JSON-safe, including the idempotency key
    args_hash: str
    idempotency_key: str
    proposed_at: datetime
    proposed_turn: int
    summary: str  # what the agent reads back for confirmation
    amount_paise: int | None
    verdict: str
    rule: str
    state: ActionState = ActionState.PROPOSED
    history: list[Transition] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def transition(
        self, target: ActionState, *, at: datetime, turn: int, note: str | None = None
    ) -> None:
        if target not in ALLOWED_TRANSITIONS.get(self.state, frozenset()):
            raise InvalidTransitionError(f"{self.state} -> {target} is not allowed")
        self.history.append(Transition(self.state, target, at, turn, note))
        self.state = target

    def is_expired(self, *, now: datetime, turn: int, config: Confirmation) -> bool:
        """Expired after more than N further customer turns or more than S seconds."""
        if self.state is not ActionState.AWAITING_CONFIRMATION:
            return False
        too_many_turns = turn - self.proposed_turn > config.expire_after_turns
        too_old = (now - self.proposed_at).total_seconds() > config.expire_after_seconds
        return too_many_turns or too_old
