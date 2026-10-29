import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.policy import MAX_VERIFICATION_ATTEMPTS
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
    history: list[ToolCallRecord] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return self.customer_id is not None

    @property
    def is_locked(self) -> bool:
        return (
            not self.is_verified and self.failed_verification_attempts >= MAX_VERIFICATION_ATTEMPTS
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
