import random
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.seed import ANCHOR
from app.policy import gate
from app.policy.state import ActionState, PendingAction, args_hash
from app.repositories.customers import get_customer_by_email
from app.tools.base import WRITE_TOOLS
from app.tools.context import ToolContext
from app.tools.errors import ToolResult
from app.tools.registry import call_tool


def make_ctx(db: AsyncSession) -> ToolContext:
    return ToolContext(db=db, clock=lambda: ANCHOR, rng=random.Random(7))


async def verify_as(ctx: ToolContext, email: str) -> int:
    """Verify `ctx` as the customer with this email via the real tool; returns customer id."""
    customer = await get_customer_by_email(ctx.db, email)
    assert customer is not None
    postal_code = customer.postal_code
    result = await call_tool(ctx, "verify_customer", {"email": email, "postal_code": postal_code})
    assert result.ok, result.error
    return ctx.state.customer_id


def new_key() -> str:
    return f"test-{uuid.uuid4().hex}"


async def run_write(ctx: ToolContext, tool: str, **args) -> ToolResult:
    """The real path: propose -> next customer turn -> confirm -> execute.

    A refusal at proposal is returned as a failed ToolResult, so callers can treat both
    stages alike.
    """
    proposed = await gate.propose(ctx, tool, args)
    if not proposed.ok:
        return ToolResult(tool=tool, ok=False, error=proposed.error)
    gate.register_customer_turn(ctx)
    confirmed = gate.confirm(ctx, proposed.action.action_id)
    assert confirmed.ok, confirmed.error
    executed = await gate.execute(ctx, proposed.action.action_id)
    return executed.tool_result


def last_action(ctx: ToolContext) -> PendingAction:
    return list(ctx.state.actions.values())[-1]


def force_confirm(
    ctx: ToolContext, tool: str, key: str, amount_paise: int | None = None, **args
) -> dict:
    """Plant a confirmed action without the gate, to test the layers beneath it
    (idempotency ledger, concurrency). Returns the arguments to pass to call_tool."""
    spec = WRITE_TOOLS[tool]
    validated = spec.input_model.model_validate({**args, "idempotency_key": key})
    action = PendingAction(
        id=uuid.uuid4().hex,
        tool=tool,
        arguments=validated.model_dump(mode="json"),
        args_hash=args_hash(validated),
        idempotency_key=key,
        proposed_at=ctx.now(),
        proposed_turn=ctx.state.turn,
        summary="(test)",
        amount_paise=amount_paise,
        verdict="approve",
        rule="test",
        state=ActionState.CONFIRMED,
    )
    ctx.state.actions[action.id] = action
    return action.arguments
