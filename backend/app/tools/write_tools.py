"""Write tools. Each one locks, asks the PolicyEngine again under the lock, then writes.

They run only for a confirmed action (enforced by the dispatcher in registry.py), and the
amount they act on must equal the amount the customer confirmed. No tool accepts an amount.
"""

from typing import Self

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from app.db.models import Order, OrderItem, Refund, Return, Ticket
from app.domain.enums import (
    EscalationReason,
    OrderStatus,
    RefundReason,
    RefundRequestReason,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)
from app.domain.formatting import format_date, format_inr
from app.domain.order_ids import spoken_confusables
from app.policy.config import Issue, get_policy
from app.policy.engine import Verdict, get_engine, shipment_for
from app.policy.facts import lock_customer
from app.repositories import orders as order_repo
from app.tools.base import OrderNumber, Sku, ToolKind, ToolOutput, WriteInput, tool
from app.tools.common import (
    authorize,
    find_item,
    load_owned_order,
    refund_ref,
    resolve_order,
    return_ref,
    ticket_ref,
)
from app.tools.context import ToolContext
from app.tools.errors import ToolFailure


async def _lock_and_load(ctx: ToolContext, tool: str, args: WriteInput) -> Order:
    await lock_customer(ctx.db, ctx.require_customer_id())
    order = await resolve_order(ctx, tool, args, for_update=True)
    if order is None:
        raise ValueError(f"{tool} needs an order")
    return order


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

    Fails with not_eligible if policy forbids it. This does not refund; propose
    issue_refund (reason "return") afterwards if appropriate.
    """
    order = await _lock_and_load(ctx, "create_return", args)
    item = find_item(order, args.sku)
    decision, facts = await authorize(ctx, "create_return", args, order)
    engine = get_engine()
    amount = decision.amount_paise or 0
    refund_check = engine.check_refund_caps(order, amount, facts)
    shipment = shipment_for(order, item)
    deadline = engine.return_deadline(shipment.delivered_at) if shipment else None

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
        return_by=format_date(deadline) if deadline else None,
        refund_amount_paise=amount,
        refund_amount_display=format_inr(amount),
        refund_requires_human=refund_check.verdict is not Verdict.APPROVE,
    )


# --- issue_refund ---------------------------------------------------------------------------


class IssueRefundInput(WriteInput):
    order_number: OrderNumber
    reason: RefundRequestReason
    sku: Sku | None = Field(
        None, description='Refund one item. Required for reason "return"; omit for whole order.'
    )

    @model_validator(mode="after")
    def _return_needs_item(self) -> Self:
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


@tool(ToolKind.WRITE)
async def issue_refund(ctx: ToolContext, args: IssueRefundInput) -> IssueRefundOutput:
    """Refund the verified customer for a returned item or a missing delivery.

    The amount is computed by policy from stored data, capped at what's still refundable,
    and subject to per-order and per-customer limits. Goodwill refunds, parcels the carrier
    marked delivered and amounts over a limit fail with requires_human.
    """
    order = await _lock_and_load(ctx, "issue_refund", args)
    decision, _ = await authorize(ctx, "issue_refund", args, order)
    linked_return = None
    if args.reason is RefundRequestReason.RETURN:
        item = find_item(order, args.sku or "")
        refunded = {r.return_id for r in order.refunds if r.return_id is not None}
        linked_return = next(
            r
            for r in order.returns
            if r.order_item_id == item.id
            and r.status is not ReturnStatus.REJECTED
            and r.id not in refunded
        )
    amount = decision.amount_paise or 0
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
        order_number=order.order_number,
        amount_paise=amount,
        amount_display=format_inr(amount),
        status=refund.status,
        reason=refund.reason,
        expected_timeline=get_policy().refunds.processing_time,
    )


# --- reship_order ---------------------------------------------------------------------------


class ReshipOrderInput(WriteInput):
    order_number: OrderNumber
    issue: Issue = Field(description="Why a replacement is needed.")
    sku: Sku | None = Field(
        None, description="Item to replace. Required unless the issue is missing_delivery."
    )

    @model_validator(mode="after")
    def _item_issues_need_sku(self) -> Self:
        if self.issue is not Issue.MISSING_DELIVERY and self.sku is None:
            raise ValueError("sku is required unless issue is missing_delivery")
        return self


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
    """Send a free replacement on the verified customer's order.

    For a lost or badly overdue parcel (issue missing_delivery), or for a damaged or wrong
    item inside the return window. Which issues allow a replacement is set by policy.
    Creates a new order at no charge.
    """
    customer_id = ctx.require_customer_id()
    order = await _lock_and_load(ctx, "reship_order", args)
    await authorize(ctx, "reship_order", args, order)
    items = [find_item(order, args.sku)] if args.sku else order.items
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
        original_order_number=order.order_number,
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
    order = await resolve_order(ctx, "create_ticket", args)
    await authorize(ctx, "create_ticket", args, order)
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
    await authorize(ctx, "escalate_to_human", args, None)
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
