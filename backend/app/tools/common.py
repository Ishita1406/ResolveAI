from datetime import datetime

from app.db.models import Order, OrderItem
from app.domain.formatting import format_date, format_inr
from app.domain.order_ids import match_order_numbers
from app.domain.policy import Decision, Outcome, Reason
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


def reason_message(
    reason: Reason | None,
    order_number: str,
    *,
    deadline: datetime | None = None,
    eta: datetime | None = None,
    amount_paise: int | None = None,
) -> str:
    n = order_number
    match reason:
        case None:
            return "Eligible."
        case Reason.ORDER_CANCELLED:
            return (
                f"Order {n} was cancelled, so it can't be returned or refunded again. "
                "Refunds for cancellations are processed automatically."
            )
        case Reason.NOT_SHIPPED:
            return f"That item from order {n} hasn't shipped yet."
        case Reason.NOT_DELIVERED:
            return f"That item from order {n} hasn't been delivered yet."
        case Reason.WINDOW_EXPIRED:
            closed = f" on {format_date(deadline)}" if deadline else ""
            return f"The return window for order {n} closed{closed}."
        case Reason.ALREADY_RETURNED:
            return f"A return for that item on order {n} is already in progress or complete."
        case Reason.ALREADY_REFUNDED:
            return f"Order {n} has already been fully refunded."
        case Reason.ALREADY_RESOLVED:
            return f"Order {n} already has a replacement or refund for the missing delivery."
        case Reason.NO_RETURN_ON_FILE:
            return f"There's no return on file for that item on order {n} yet."
        case Reason.NOT_YET_OVERDUE:
            by = f" It's expected by {format_date(eta)}." if eta else ""
            return f"Order {n} is still on its way and isn't late yet.{by}"
        case Reason.DELIVERED_PER_CARRIER:
            return (
                f"The carrier shows order {n} as delivered, so a specialist needs to investigate."
            )
        case Reason.RETURNED_TO_SENDER:
            return f"The parcel for order {n} is being returned to us; a specialist will help."
        case Reason.ABOVE_AUTO_REFUND_CAP:
            amount = f" of {format_inr(amount_paise)}" if amount_paise is not None else ""
            return f"A refund{amount} is more than I can approve myself, so a specialist will."
        case Reason.GOODWILL_NEEDS_APPROVAL:
            return "Goodwill refunds need a specialist's approval."
        case Reason.LOST_IN_TRANSIT | Reason.OVERDUE:
            return f"Order {n} qualifies as a missing delivery."
    return f"That isn't possible for order {n}."


def policy_failure(decision: Decision, order_number: str, **message_args) -> ToolFailure:
    message = reason_message(decision.reason, order_number, **message_args)
    if decision.outcome is Outcome.REQUIRES_HUMAN:
        return ToolFailure(
            ErrorCode.REQUIRES_HUMAN,
            message,
            hint="Offer escalate_to_human with reason policy_requires_review.",
            details={"reason": decision.reason},
        )
    return ToolFailure(
        ErrorCode.NOT_ELIGIBLE,
        message,
        hint="Explain the reason. Do not retry or rephrase the request to get around it.",
        details={"reason": decision.reason},
    )
