"""The path every write takes: propose -> (customer confirms) -> execute.

    propose(ctx, tool, args)      validate, ask the PolicyEngine, open an action awaiting
                                  confirmation (or reject it), return a read-back summary
    register_customer_turn(ctx)   call once per customer utterance; expires stale proposals
    confirm(ctx, action_id)       only from awaiting_confirmation, and only in a later turn
    execute(ctx, action_id)       runs the write tool; it re-checks policy under lock
    cancel(ctx, action_id)        customer declined or changed their mind

The idempotency key is minted here, once per action, so retrying execute() can never
repeat the write.
"""

import uuid
from typing import Any

from pydantic import BaseModel, ValidationError

from app.domain.enums import RefundRequestReason
from app.domain.formatting import format_inr
from app.policy.audit import Stage, flush_decisions
from app.policy.config import get_policy
from app.policy.engine import ISSUE_TEXT, PolicyDecision
from app.policy.state import (
    RUNNABLE_STATES,
    ActionState,
    PendingAction,
    args_hash,
)
from app.tools.base import WRITE_TOOLS
from app.tools.common import evaluate, find_item, policy_failure, resolve_order
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode, ToolError, ToolFailure, ToolResult
from app.tools.registry import call_tool, validation_error


class ActionView(BaseModel):
    action_id: str
    tool: str
    state: ActionState
    summary: str
    amount_paise: int | None
    amount_display: str | None
    verdict: str
    rule: str


class ActionResult(BaseModel):
    ok: bool
    action: ActionView | None = None
    error: ToolError | None = None
    tool_result: ToolResult | None = None


def view(action: PendingAction) -> ActionView:
    return ActionView(
        action_id=action.id,
        tool=action.tool,
        state=action.state,
        summary=action.summary,
        amount_paise=action.amount_paise,
        amount_display=format_inr(action.amount_paise) if action.amount_paise else None,
        verdict=action.verdict,
        rule=action.rule,
    )


def _fail(code: ErrorCode, message: str, hint: str, action: PendingAction | None = None):
    return ActionResult(
        ok=False,
        action=view(action) if action else None,
        error=ToolError(code=code, message=message, hint=hint),
    )


def expire_stale(ctx: ToolContext) -> list[PendingAction]:
    config = get_policy().confirmation
    now, turn = ctx.now(), ctx.state.turn
    expired = []
    for action in ctx.state.actions.values():
        if action.is_expired(now=now, turn=turn, config=config):
            action.transition(ActionState.EXPIRED, at=now, turn=turn, note="not confirmed in time")
            expired.append(action)
    return expired


def register_customer_turn(ctx: ToolContext) -> list[ActionView]:
    """Advance the turn counter; returns proposals that just expired (tell the customer)."""
    ctx.state.turn += 1
    return [view(a) for a in expire_stale(ctx)]


def _readback(tool: str, args: Any, order, decision: PolicyDecision) -> str:
    """What the agent says to get confirmation. States the amount when money moves."""
    n = order.order_number if order else None
    amount = format_inr(decision.amount_paise) if decision.amount_paise is not None else None
    match tool:
        case "create_return":
            item = find_item(order, args.sku)
            return f"start a return for the {item.product_name} from order {n}"
        case "issue_refund":
            what = (
                f"the {find_item(order, args.sku).product_name} on order {n}"
                if args.sku
                else f"order {n}"
            )
            why = {
                RefundRequestReason.RETURN: "returned item",
                RefundRequestReason.MISSING_DELIVERY: "missing delivery",
                RefundRequestReason.GOODWILL: "goodwill",
            }[args.reason]
            return f"refund {amount} to your original payment method for {what} ({why})"
        case "reship_order":
            items = [find_item(order, args.sku)] if args.sku else order.items
            names = ", ".join(i.product_name for i in items)
            return f"send a free replacement of {names} from order {n} for {ISSUE_TEXT[args.issue]}"
        case "create_ticket":
            about = f" about order {n}" if n else ""
            return f"open a support ticket{about} for our team to follow up"
        case "escalate_to_human":
            return "connect you with a colleague"
    return tool


async def propose(ctx: ToolContext, tool: str, arguments: dict[str, Any]) -> ActionResult:
    spec = WRITE_TOOLS.get(tool)
    if spec is None:
        return _fail(
            ErrorCode.UNKNOWN_TOOL,
            "Sorry, I can't do that.",
            f"No write tool named {tool!r}. Available: {', '.join(sorted(WRITE_TOOLS))}.",
        )
    action_id = uuid.uuid4().hex
    key = f"act-{action_id}"
    # The key is always ours; anything the model sent is ignored.
    payload = {k: v for k, v in arguments.items() if k != "idempotency_key"}
    try:
        args = spec.input_model.model_validate({**payload, "idempotency_key": key})
    except ValidationError as exc:
        return ActionResult(ok=False, error=validation_error(exc))
    try:
        if spec.requires_verification:
            ctx.require_customer_id()
    except ToolFailure as failure:
        return ActionResult(ok=False, error=failure.error)

    now, turn = ctx.now(), ctx.state.turn
    expire_stale(ctx)
    for other in ctx.state.actions.values():
        if other.state is ActionState.AWAITING_CONFIRMATION:
            other.transition(ActionState.CANCELLED, at=now, turn=turn, note="superseded")

    try:
        order = await resolve_order(ctx, tool, args)
        decision, _ = await evaluate(
            ctx, tool, args, order, stage=Stage.PROPOSAL, action_id=action_id
        )
        summary = _readback(tool, args, order, decision)
    except ToolFailure as failure:
        await ctx.db.rollback()
        await flush_decisions(ctx)
        return ActionResult(ok=False, error=failure.error)
    await ctx.db.rollback()
    await flush_decisions(ctx)

    action = PendingAction(
        id=action_id,
        tool=tool,
        arguments=args.model_dump(mode="json"),
        args_hash=args_hash(args),
        idempotency_key=key,
        proposed_at=now,
        proposed_turn=turn,
        summary=summary,
        amount_paise=decision.amount_paise,
        verdict=decision.verdict.value,
        rule=decision.rule,
    )
    ctx.state.actions[action_id] = action
    if not decision.approved:
        action.transition(ActionState.REJECTED, at=now, turn=turn, note=decision.rule)
        return ActionResult(ok=False, action=view(action), error=policy_failure(decision).error)
    action.transition(ActionState.AWAITING_CONFIRMATION, at=now, turn=turn)
    return ActionResult(ok=True, action=view(action))


def _get(ctx: ToolContext, action_id: str) -> PendingAction | None:
    return ctx.state.actions.get(action_id)


_NOT_PENDING = {
    ActionState.EXPIRED: "That request timed out before you confirmed. Shall I go over it again?",
    ActionState.CANCELLED: "That request was cancelled.",
    ActionState.EXECUTED: "That's already been done.",
    ActionState.REJECTED: "That request couldn't go ahead.",
    ActionState.CONFIRMED: "You've already confirmed that.",
}


def confirm(ctx: ToolContext, action_id: str) -> ActionResult:
    action = _get(ctx, action_id)
    if action is None:
        return _fail(ErrorCode.ACTION_NOT_FOUND, "Sorry, I lost track of that request.",
                     "Unknown action_id; propose again.")  # fmt: skip
    expire_stale(ctx)
    if action.state is not ActionState.AWAITING_CONFIRMATION:
        return _fail(
            ErrorCode.ACTION_NOT_PENDING,
            _NOT_PENDING.get(action.state, "That request isn't waiting for confirmation."),
            f"Action is {action.state}; propose again if the customer still wants it.",
            action,
        )
    if ctx.state.turn <= action.proposed_turn:
        return _fail(
            ErrorCode.ACTION_NOT_PENDING,
            "Just to be sure, could you confirm that?",
            "Confirmation must come from the customer in a turn after the proposal.",
            action,
        )
    action.transition(ActionState.CONFIRMED, at=ctx.now(), turn=ctx.state.turn)
    return ActionResult(ok=True, action=view(action))


def cancel(ctx: ToolContext, action_id: str, note: str = "customer declined") -> ActionResult:
    action = _get(ctx, action_id)
    if action is None:
        return _fail(ErrorCode.ACTION_NOT_FOUND, "Sorry, I lost track of that request.",
                     "Unknown action_id.")  # fmt: skip
    expire_stale(ctx)
    if action.state not in (ActionState.AWAITING_CONFIRMATION, ActionState.CONFIRMED):
        return _fail(
            ErrorCode.ACTION_NOT_PENDING,
            _NOT_PENDING.get(action.state, "There's nothing to cancel."),
            f"Action is {action.state}.",
            action,
        )
    action.transition(ActionState.CANCELLED, at=ctx.now(), turn=ctx.state.turn, note=note)
    return ActionResult(ok=True, action=view(action))


async def execute(ctx: ToolContext, action_id: str) -> ActionResult:
    """Run a confirmed action. Calling it again replays the stored result."""
    action = _get(ctx, action_id)
    if action is None:
        return _fail(ErrorCode.ACTION_NOT_FOUND, "Sorry, I lost track of that request.",
                     "Unknown action_id.")  # fmt: skip
    if action.state not in RUNNABLE_STATES:
        return _fail(
            ErrorCode.ACTION_NOT_CONFIRMED,
            _NOT_PENDING.get(action.state, "I need your confirmation before I do that."),
            f"Action is {action.state}; it must be confirmed first.",
            action,
        )
    result = await call_tool(ctx, action.tool, action.arguments)
    return ActionResult(ok=result.ok, action=view(action), error=result.error, tool_result=result)
