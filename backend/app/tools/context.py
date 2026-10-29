import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.policy.audit import DecisionRecord
from app.policy.config import get_policy
from app.policy.state import PendingAction
from app.tools.errors import ErrorCode, ToolFailure


@dataclass(frozen=True)
class ToolCallRecord:
    tool: str
    ok: bool
    error_code: str | None
    at: datetime


@dataclass
class SessionState:
    """Per-conversation state. Only verify_customer may set the customer fields."""

    customer_id: int | None = None
    customer_name: str | None = None
    verified_at: datetime | None = None
    failed_verification_attempts: int = 0
    turn: int = 0  # customer turns so far; drives confirmation expiry
    history: list[ToolCallRecord] = field(default_factory=list)
    actions: dict[str, PendingAction] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return self.customer_id is not None

    @property
    def is_locked(self) -> bool:
        max_attempts = get_policy().verification.max_attempts
        return not self.is_verified and self.failed_verification_attempts >= max_attempts

    def action_for_key(self, idempotency_key: str) -> PendingAction | None:
        return next(
            (a for a in self.actions.values() if a.idempotency_key == idempotency_key), None
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ToolContext:
    """Everything a tool may use. The customer identity comes from here, never from arguments."""

    db: AsyncSession
    state: SessionState = field(default_factory=SessionState)
    conversation_id: int | None = None
    clock: Callable[[], datetime] = _utcnow
    rng: random.Random = field(default_factory=random.SystemRandom)
    pending_decisions: list[DecisionRecord] = field(default_factory=list)

    def now(self) -> datetime:
        return self.clock()

    def require_customer_id(self) -> int:
        if self.state.customer_id is None:
            raise ToolFailure(
                ErrorCode.NOT_VERIFIED,
                "Before I can help with that, I need to verify your account.",
                hint="Call verify_customer first.",
            )
        return self.state.customer_id
