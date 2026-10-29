"""Write tools. Each one locks the order, re-checks policy against current data, then writes.

Idempotency, transactions and error mapping are handled by the dispatcher (registry.py).
Amounts are always computed here from stored data; no tool accepts an amount argument.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from app.db.models import Order, OrderItem, Refund, Return, Shipment, Ticket
from app.domain.enums import (
    EscalationReason,
    OrderStatus,
    RefundReason,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)
from app.domain.formatting import format_date, format_inr
from app.domain.order_ids import spoken_confusables
from app.domain.policy import (
    Decision,
    Outcome,
    Reason,
    check_refund_cap,
    evaluate_item_return,
    evaluate_missing_delivery,
    refundable_balance_paise,
    shipment_for,
)
from app.repositories import orders as order_repo
from app.repositories.support import OPEN_TICKET_STATUSES
from app.tools.base import OrderNumber, Sku, ToolKind, ToolOutput, WriteInput, tool
from app.tools.common import (
    find_item,
    load_owned_order,
    policy_failure,
    refund_ref,
    return_ref,
    ticket_ref,
)
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode, ToolFailure

REFUND_TIMELINE = "5 to 7 business days"

# --- create_return --------------------------------------------------------------------------


class CreateReturnInput(WriteInput):
    order_number: OrderNumber
    sku: Sku
    reason: ReturnReason
    notes: str | None = Field(None, max_length=500, description="Caller's description, if any.")


class CreateReturnOutput(ToolOutput):
    return_reference: str
    order_number: str
    sku: str
    status: ReturnStatus
    return_by: str | None
    refund_amount_paise: int
    refund_amount_display: str
    refund_requires_human: bool


@tool(ToolKind.WRITE)
async def create_return(ctx: ToolContext, args: CreateReturnInput) -> CreateReturnOutput:
    """Open a return for one item on the verified customer's order.

    Only after the customer has confirmed. Fails with not_eligible if policy forbids it.
    This does not refund; call issue_refund (reason "return") afterwards if appropriate.
    """
    order = await load_owned_order(ctx, args.order_number, for_update=True)
    item = find_item(order, args.sku)
    assessment = evaluate_item_return(order, item, ctx.now())
    if not assessment.decision.eligible:
        raise policy_failure(assessment.decision, order.order_number, deadline=assessment.deadline)

    ret = Return(
        order_id=order.id,
        order_item_id=item.id,
        quantity=item.quantity,
        reason=args.reason,
        status=ReturnStatus.REQUESTED,
        notes=args.notes,
        idempotency_key=args.idempotency_key,
        requested_at=ctx.now(),
    )
    ctx.db.add(ret)
    await ctx.db.flush()
    return CreateReturnOutput(
        return_reference=return_ref(ret.id),
        order_number=order.order_number,
        sku=item.sku,
        status=ret.status,
        return_by=format_date(assessment.deadline) if assessment.deadline else None,
        refund_amount_paise=assessment.refund_amount_paise,
        refund_amount_display=format_inr(assessment.refund_amount_paise),
        refund_requires_human=assessment.refund_decision.outcome is Outcome.REQUIRES_HUMAN,
    )


# --- issue_refund ---------------------------------------------------------------------------


class RefundRequestReason(StrEnum):
    # Cancellation refunds are automatic and never issued by the agent.
    RETURN = "return"
    MISSING_DELIVERY = "missing_delivery"
    GOODWILL = "goodwill"


class IssueRefundInput(WriteInput):
    order_number: OrderNumber
    reason: RefundRequestReason
    sku: Sku | None = Field(
        None, description='Refund one item. Required for reason "return"; omit for whole order.'
    )

    @model_validator(mode="after")
    def _return_needs_item(self) -> "IssueRefundInput":
        if self.reason is RefundRequestReason.RETURN and self.sku is None:
            raise ValueError('sku is required when reason is "return"')
        return self


class IssueRefundOutput(ToolOutput):
    refund_reference: str
    order_number: str
    amount_paise: int
    amount_display: str
    status: RefundStatus
    reason: RefundReason
    expected_timeline: str


async def _has_replacement(ctx: ToolContext, order: Order) -> bool:
    stmt = select(Order.id).where(Order.replaces_order_id == order.id).limit(1)
    return await ctx.db.scalar(stmt) is not None


async def _missing_delivery_resolved(ctx: ToolContext, order: Order) -> bool:
    prior_refund = any(
        r.reason is RefundReason.MISSING_DELIVERY
        and r.status in (RefundStatus.PENDING, RefundStatus.SUCCEEDED)
        for r in order.refunds
    )
    return prior_refund or await _has_replacement(ctx, order)


def _missing_delivery_decision(order: Order, items: list[OrderItem], ctx: ToolContext) -> Decision:
    """Combine per-shipment decisions: any human review wins, then any refusal."""
    now = ctx.now()
    shipments: dict[int | None, Shipment | None] = {
        i.shipment_id: shipment_for(order, i) for i in items
    }
    decisions = [evaluate_missing_delivery(s, now) for s in shipments.values()]
    for outcome in (Outcome.REQUIRES_HUMAN, Outcome.NOT_ELIGIBLE):
        hit = next((d for d in decisions if d.outcome is outcome), None)
        if hit:
            return hit
    return decisions[0]


def _eta_of(order: Order, items: list[OrderItem]):
    etas = [s.estimated_delivery_at for i in items if (s := shipment_for(order, i))]
    return max((e for e in etas if e), default=None)


@tool(ToolKind.WRITE)
async def issue_refund(ctx: ToolContext, args: IssueRefundInput) -> IssueRefundOutput:
    """Refund the verified customer for a return or a missing delivery.

    Only after the customer has confirmed the amount from check_return_eligibility or
    get_order. The amount is computed by the backend and capped at what's still refundable.
    Refunds above the automatic limit, goodwill refunds and parcels the carrier marked
    delivered fail with requires_human.
    """
    order = await load_owned_order(ctx, args.order_number, for_update=True)
    n = order.order_number
    if order.status is OrderStatus.CANCELLED:
        raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, Reason.ORDER_CANCELLED), n)
    balance = refundable_balance_paise(order)
    if balance <= 0:
        raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, Reason.ALREADY_REFUNDED), n)

    linked_return: Return | None = None
    match args.reason:
        case RefundRequestReason.GOODWILL:
            raise policy_failure(
                Decision(Outcome.REQUIRES_HUMAN, Reason.GOODWILL_NEEDS_APPROVAL), n
            )
        case RefundRequestReason.RETURN:
            item = find_item(order, args.sku or "")
            refunded_returns = {r.return_id for r in order.refunds if r.return_id is not None}
            linked_return = next(
                (
                    r
                    for r in order.returns
                    if r.order_item_id == item.id
                    and r.status is not ReturnStatus.REJECTED
                    and r.id not in refunded_returns
                ),
                None,
            )
            if linked_return is None:
                already = any(r.order_item_id == item.id for r in order.returns)
                reason = Reason.ALREADY_REFUNDED if already else Reason.NO_RETURN_ON_FILE
                raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, reason), n)
            amount = min(item.line_total_paise, balance)
        case RefundRequestReason.MISSING_DELIVERY:
            items = [find_item(order, args.sku)] if args.sku else order.items
            if await _missing_delivery_resolved(ctx, order):
                raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, Reason.ALREADY_RESOLVED), n)
            decision = _missing_delivery_decision(order, items, ctx)
            if not decision.eligible:
                raise policy_failure(decision, n, eta=_eta_of(order, items))
            amount = min(sum(i.line_total_paise for i in items), balance) if args.sku else balance

    cap = check_refund_cap(amount)
    if not cap.eligible:
        raise policy_failure(cap, n, amount_paise=amount)

    refund = Refund(
        order_id=order.id,
        return_id=linked_return.id if linked_return else None,
        amount_paise=amount,
        status=RefundStatus.PENDING,
        reason=RefundReason(args.reason.value),
        idempotency_key=args.idempotency_key,
        requested_at=ctx.now(),
    )
    ctx.db.add(refund)
    await ctx.db.flush()
    return IssueRefundOutput(
        refund_reference=refund_ref(refund.id),
        order_number=n,
        amount_paise=amount,
        amount_display=format_inr(amount),
        status=refund.status,
        reason=refund.reason,
        expected_timeline=REFUND_TIMELINE,
    )


# --- reship_order ---------------------------------------------------------------------------


class ReshipOrderInput(WriteInput):
    order_number: OrderNumber
    sku: Sku | None = Field(None, description="Reship one item; omit to reship every item.")


class ReshipOrderOutput(ToolOutput):
    replacement_order_number: str
    original_order_number: str
    item_names: list[str]
    status: OrderStatus


async def _new_order_number(ctx: ToolContext, customer_id: int) -> str:
    """A free number that doesn't sound like any of the customer's existing orders."""
    taken = set(await ctx.db.scalars(select(Order.order_number)))
    own = await order_repo.list_order_numbers_for_customer(ctx.db, customer_id)
    blocked = taken.union(*(spoken_confusables(o) for o in own))
    for _ in range(1000):
        number = f"VX-{ctx.rng.randrange(1000, 10000)}"
        if number not in blocked:
            return number
    raise RuntimeError("no free order number")


@tool(ToolKind.WRITE)
async def reship_order(ctx: ToolContext, args: ReshipOrderInput) -> ReshipOrderOutput:
    """Send a free replacement for a lost or badly overdue parcel on the verified customer's order.

    Only after the customer has chosen a reshipment over a refund and confirmed. Creates a
    new order at no charge. Parcels the carrier marked delivered fail with requires_human.
    """
    customer_id = ctx.require_customer_id()
    order = await load_owned_order(ctx, args.order_number, for_update=True)
    n = order.order_number
    if order.status is OrderStatus.CANCELLED:
        raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, Reason.ORDER_CANCELLED), n)
    items = [find_item(order, args.sku)] if args.sku else order.items
    if await _missing_delivery_resolved(ctx, order):
        raise policy_failure(Decision(Outcome.NOT_ELIGIBLE, Reason.ALREADY_RESOLVED), n)
    decision = _missing_delivery_decision(order, items, ctx)
    if not decision.eligible:
        raise policy_failure(decision, n, eta=_eta_of(order, items))
    value = sum(i.line_total_paise for i in items)
    cap = check_refund_cap(value)
    if not cap.eligible:
        raise policy_failure(cap, n, amount_paise=value)

    replacement = Order(
        order_number=await _new_order_number(ctx, customer_id),
        customer_id=customer_id,
        status=OrderStatus.PLACED,
        subtotal_paise=0,
        shipping_fee_paise=0,
        total_paise=0,
        placed_at=ctx.now(),
        replaces_order_id=order.id,
        items=[
            OrderItem(
                sku=i.sku, product_name=i.product_name, quantity=i.quantity, unit_price_paise=0
            )
            for i in items
        ],
    )
    ctx.db.add(replacement)
    await ctx.db.flush()
    return ReshipOrderOutput(
        replacement_order_number=replacement.order_number,
        original_order_number=n,
        item_names=[i.product_name for i in items],
        status=replacement.status,
    )


# --- create_ticket --------------------------------------------------------------------------


class CreateTicketInput(WriteInput):
    category: TicketCategory
    summary: str = Field(min_length=10, max_length=1000, description="What the customer needs.")
    order_number: OrderNumber | None = None
    priority: TicketPriority = TicketPriority.NORMAL


class CreateTicketOutput(ToolOutput):
    ticket_reference: str
    status: TicketStatus
    category: TicketCategory
    order_number: str | None


@tool(ToolKind.WRITE)
async def create_ticket(ctx: ToolContext, args: CreateTicketInput) -> CreateTicketOutput:
    """Open a support ticket for follow-up by the support team, for the verified customer.

    Use when an issue can't be resolved in the call but doesn't need a live handoff.
    Fails with ticket_already_open if the same order already has an open ticket of this kind.
    """
    customer_id = ctx.require_customer_id()
    order = await load_owned_order(ctx, args.order_number) if args.order_number else None
    if order is not None:
        existing = await ctx.db.scalar(
            select(Ticket).where(
                Ticket.order_id == order.id,
                Ticket.category == args.category,
                Ticket.status.in_(OPEN_TICKET_STATUSES),
            )
        )
        if existing is not None:
            raise ToolFailure(
                ErrorCode.TICKET_ALREADY_OPEN,
                f"There's already an open ticket, {ticket_ref(existing.id)}, for order "
                f"{order.order_number}. Our team is working on it.",
                hint="Share the existing reference. Don't open a duplicate.",
                details={"ticket_reference": ticket_ref(existing.id)},
            )
    now = ctx.now()
    ticket = Ticket(
        customer_id=customer_id,
        order_id=order.id if order else None,
        conversation_id=ctx.conversation_id,
        category=args.category,
        status=TicketStatus.OPEN,
        priority=args.priority,
        summary=args.summary,
        created_at=now,
        updated_at=now,
    )
    ctx.db.add(ticket)
    await ctx.db.flush()
    return CreateTicketOutput(
        ticket_reference=ticket_ref(ticket.id),
        status=ticket.status,
        category=ticket.category,
        order_number=order.order_number if order else None,
    )


# --- escalate_to_human ----------------------------------------------------------------------


class EscalateToHumanInput(WriteInput):
    reason: EscalationReason
    summary: str = Field(
        min_length=10, max_length=2000, description="The issue and what has been tried so far."
    )
    order_number: str | None = Field(
        None, max_length=40, description="Order the caller mentioned, verified or not."
    )
    category: TicketCategory = TicketCategory.OTHER


class ActionAttempt(BaseModel):
    tool: str
    ok: bool
    error_code: str | None
    at: str


class Handoff(BaseModel):
    reason: EscalationReason
    summary: str
    verified: bool
    customer_name: str | None
    order_number: str | None = Field(description="Set only if verified and owned by the caller.")
    caller_order_reference: str | None = Field(description="Unverified reference, as given.")
    actions_attempted: list[ActionAttempt]


class EscalateToHumanOutput(ToolOutput):
    ticket_reference: str
    handoff: Handoff


@tool(ToolKind.WRITE, requires_verification=False)
async def escalate_to_human(ctx: ToolContext, args: EscalateToHumanInput) -> EscalateToHumanOutput:
    """Hand the call to a human agent with a structured summary. Works before verification.

    Use when the customer asks for a person, policy requires review, verification fails,
    the request keeps being misunderstood, a tool keeps failing, or you aren't confident.
    The tools already called in this session are attached automatically.
    """
    state = ctx.state
    order = None
    if state.is_verified and args.order_number:
        try:
            order = await load_owned_order(ctx, args.order_number)
        except ToolFailure:
            order = None  # keep the caller's reference; an escalation must not fail on it
    handoff = Handoff(
        reason=args.reason,
        summary=args.summary,
        verified=state.is_verified,
        customer_name=state.customer_name,
        order_number=order.order_number if order else None,
        caller_order_reference=None if order else args.order_number,
        actions_attempted=[
            ActionAttempt(tool=h.tool, ok=h.ok, error_code=h.error_code, at=h.at.isoformat())
            for h in state.history
        ],
    )
    now = ctx.now()
    ticket = Ticket(
        customer_id=state.customer_id,
        order_id=order.id if order else None,
        conversation_id=ctx.conversation_id,
        category=args.category,
        status=TicketStatus.ESCALATED,
        priority=TicketPriority.HIGH,
        summary=args.summary,
        details=handoff.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
    )
    ctx.db.add(ticket)
    await ctx.db.flush()
    return EscalateToHumanOutput(ticket_reference=ticket_ref(ticket.id), handoff=handoff)
