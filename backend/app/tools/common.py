from typing import Any

from app.db.models import Order, OrderItem
from app.domain.order_ids import match_order_numbers
from app.policy.audit import Stage, record_decision
from app.policy.engine import (
    PolicyDecision,
    PolicyFacts,
    ProposedAction,
    Reason,
    Verdict,
    get_engine,
)
from app.policy.facts import load_facts
from app.repositories import orders as order_repo
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode, ToolFailure

MAX_CANDIDATES = 3


def return_ref(return_id: int) -> str:
    return f"RET-{return_id:06d}"


def refund_ref(refund_id: int) -> str:
    return f"RF-{refund_id:06d}"


def ticket_ref(ticket_id: int) -> str:
    return f"TCK-{ticket_id:06d}"


async def owned_order_candidates(ctx: ToolContext, heard: str) -> list[str]:
    numbers = await order_repo.list_order_numbers_for_customer(ctx.db, ctx.require_customer_id())
    return [m.order_number for m in match_order_numbers(heard, numbers)[:MAX_CANDIDATES]]


async def load_owned_order(
    ctx: ToolContext, order_number: str, *, for_update: bool = False
) -> Order:
    """The verified customer's order, or ORDER_NOT_FOUND with their own closest matches."""
    customer_id = ctx.require_customer_id()
    order = await order_repo.get_order_for_customer(
        ctx.db, customer_id, order_number, for_update=for_update
    )
    if order is not None:
        return order
    candidates = await owned_order_candidates(ctx, order_number)
    message = f"I couldn't find order {order_number} on your account."
    if candidates:
        message += f" Did you mean {' or '.join(candidates)}?"
    raise ToolFailure(
        ErrorCode.ORDER_NOT_FOUND,
        message,
        hint="Read the number back to the customer. Candidates are only their own orders.",
        details={"candidates": candidates},
    )


def find_item(order: Order, sku: str) -> OrderItem:
    item = next((i for i in order.items if i.sku == sku), None)
    if item is None:
        names = ", ".join(i.product_name for i in order.items)
        raise ToolFailure(
            ErrorCode.ITEM_NOT_FOUND,
            f"I couldn't find that item on order {order.order_number}. It has: {names}.",
            hint="Use a sku from get_order.",
            details={"skus": [i.sku for i in order.items]},
        )
    return item


async def resolve_order(
    ctx: ToolContext, tool: str, args: Any, *, for_update: bool = False
) -> Order | None:
    """The order (and item) an action refers to, checked for ownership before policy runs."""
    if tool == "escalate_to_human":
        return None  # escalation never resolves orders: it must work unverified
    number = getattr(args, "order_number", None)
    if number is None:
        return None
    order = await load_owned_order(ctx, number, for_update=for_update)
    if getattr(args, "sku", None):
        find_item(order, args.sku)
    return order


async def evaluate(
    ctx: ToolContext, tool: str, args: Any, order: Order | None, *, stage: Stage, action_id: str
) -> tuple[PolicyDecision, PolicyFacts]:
    facts = await load_facts(
        ctx.db,
        now=ctx.now(),
        customer_id=ctx.state.customer_id,
        order=order,
        ticket_category=getattr(args, "category", None) if tool == "create_ticket" else None,
    )
    decision = get_engine().evaluate(ProposedAction(tool, args), facts)
    record_decision(ctx, decision, stage=stage, tool=tool, action_id=action_id)
    return decision, facts


def policy_failure(decision: PolicyDecision) -> ToolFailure:
    details: dict[str, Any] = {"reason": decision.reason, "rule": decision.rule}
    if decision.reason is Reason.DUPLICATE_TICKET:
        details["ticket_reference"] = decision.facts["ticket_reference"]
        return ToolFailure(
            ErrorCode.TICKET_ALREADY_OPEN,
            decision.message,
            hint="Share the existing reference. Don't open a duplicate.",
            details=details,
        )
    if decision.verdict is Verdict.REQUIRE_HUMAN:
        return ToolFailure(
            ErrorCode.REQUIRES_HUMAN,
            decision.message,
            hint="Offer escalate_to_human with reason policy_requires_review.",
            details=details,
        )
    hint = "Explain the reason. Do not retry or rephrase the request to get around it."
    if decision.reason in (Reason.ALREADY_REFUNDED, Reason.ALREADY_RESOLVED):
        # Live run: after "already refunded" the model offered a replacement, refused next.
        hint += (
            " The problem is already resolved, so no refund or replacement is available for "
            "it. Don't offer one; say what was done, and offer a ticket if they still need help."
        )
    return ToolFailure(ErrorCode.NOT_ELIGIBLE, decision.message, hint=hint, details=details)


async def authorize(
    ctx: ToolContext, tool: str, args: Any, order: Order | None
) -> tuple[PolicyDecision, PolicyFacts]:
    """Execution-time policy check, run under the caller's locks.

    Raises unless approved, and unless the amount still equals what the customer confirmed.
    """
    action = ctx.state.action_for_key(args.idempotency_key)
    action_id = action.id if action else "unknown"
    decision, facts = await evaluate(
        ctx, tool, args, order, stage=Stage.EXECUTION, action_id=action_id
    )
    if not decision.approved:
        raise policy_failure(decision)
    if action is not None and action.amount_paise != decision.amount_paise:
        raise ToolFailure(
            ErrorCode.CONFIRMATION_STALE,
            "Something changed since you confirmed, so let me check that again.",
            hint="Propose the action again and get a fresh confirmation.",
            details={"confirmed_paise": action.amount_paise, "now_paise": decision.amount_paise},
        )
    return decision, facts
