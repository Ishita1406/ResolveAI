"""Read tools: no database writes. verify_customer only changes in-memory session state."""

import hmac
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from app.db.models import Order, Shipment, ShipmentEvent
from app.domain.enums import OrderStatus, RefundStatus, ShipmentEventCode, ShipmentStatus
from app.domain.formatting import format_date, format_inr
from app.domain.order_ids import match_order_numbers, parse_spoken_order_number
from app.domain.policy import (
    MAX_VERIFICATION_ATTEMPTS,
    Outcome,
    Reason,
    evaluate_item_return,
    is_overdue,
    refundable_balance_paise,
)
from app.repositories import customers as customer_repo
from app.repositories import orders as order_repo
from app.tools.base import OrderNumber, Sku, ToolInput, ToolKind, ToolOutput, tool
from app.tools.common import (
    MAX_CANDIDATES,
    find_item,
    load_owned_order,
    reason_message,
    refund_ref,
    return_ref,
)
from app.tools.context import ToolContext
from app.tools.errors import ErrorCode, ToolFailure

# --- verify_customer ------------------------------------------------------------------------


class VerifyCustomerInput(ToolInput):
    phone: str | None = Field(
        None, max_length=20, description="Registered mobile number as the caller said it."
    )
    email: str | None = Field(None, max_length=254, description="Registered email address.")
    postal_code: str = Field(
        pattern=r"^[0-9]{6}$", description="6-digit PIN code of the account address."
    )

    @model_validator(mode="after")
    def _one_identifier(self) -> Self:
        if (self.phone is None) == (self.email is None):
            raise ValueError("provide exactly one of phone or email")
        return self


class VerifyCustomerOutput(ToolOutput):
    verified: bool = True
    first_name: str


@tool(ToolKind.READ, requires_verification=False)
async def verify_customer(ctx: ToolContext, args: VerifyCustomerInput) -> VerifyCustomerOutput:
    """Verify the caller with their registered phone number or email plus their account PIN code.

    Must succeed before any order details are shared or any action is taken. The same
    message is returned whether the account doesn't exist or the details don't match.
    After 3 failed attempts verification is locked for the rest of the call.
    """
    state = ctx.state
    if state.is_locked:
        raise _locked()
    if args.phone is not None:
        customer = await customer_repo.get_customer_by_phone(ctx.db, args.phone)
    else:
        customer = await customer_repo.get_customer_by_email(ctx.db, args.email or "")
    matches = customer is not None and hmac.compare_digest(
        customer.postal_code.encode(), args.postal_code.encode()
    )
    if not matches:
        state.failed_verification_attempts += 1
        if state.is_locked:
            raise _locked()
        remaining = MAX_VERIFICATION_ATTEMPTS - state.failed_verification_attempts
        raise ToolFailure(
            ErrorCode.VERIFICATION_FAILED,
            "Those details don't match our records. Could you check them and try again?",
            hint="Ask the caller to repeat their details. Don't say which detail was wrong.",
            details={"attempts_remaining": remaining},
        )
    if state.customer_id is not None and state.customer_id != customer.id:
        raise ToolFailure(
            ErrorCode.ALREADY_VERIFIED,
            "This call is already verified for a different account, so I can't switch accounts.",
            hint="One account per call. Offer escalate_to_human if the caller insists.",
        )
    state.customer_id = customer.id
    state.customer_name = customer.full_name
    state.verified_at = state.verified_at or ctx.now()
    return VerifyCustomerOutput(first_name=customer.full_name.split()[0])


def _locked() -> ToolFailure:
    return ToolFailure(
        ErrorCode.VERIFICATION_LOCKED,
        "I'm sorry, I couldn't verify your account. Let me connect you with a colleague.",
        hint="Call escalate_to_human with reason verification_failed.",
    )


# --- shared views ---------------------------------------------------------------------------


class OrderSummary(BaseModel):
    order_number: str
    status: OrderStatus
    placed_on: str
    item_names: list[str]
    total_paise: int
    total_display: str


def _summary(order: Order) -> OrderSummary:
    return OrderSummary(
        order_number=order.order_number,
        status=order.status,
        placed_on=format_date(order.placed_at),
        item_names=[i.product_name for i in order.items],
        total_paise=order.total_paise,
        total_display=format_inr(order.total_paise),
    )


# --- list_recent_orders ---------------------------------------------------------------------


class ListRecentOrdersInput(ToolInput):
    limit: int = Field(5, ge=1, le=10, description="How many orders to return, newest first.")


class ListRecentOrdersOutput(ToolOutput):
    orders: list[OrderSummary]


@tool(ToolKind.READ)
async def list_recent_orders(
    ctx: ToolContext, args: ListRecentOrdersInput
) -> ListRecentOrdersOutput:
    """List the verified customer's most recent orders, newest first.

    Use when the caller doesn't know their order number or has several orders in flight.
    """
    orders = await order_repo.list_orders_for_customer(ctx.db, ctx.require_customer_id())
    return ListRecentOrdersOutput(orders=[_summary(o) for o in orders[: args.limit]])


# --- get_order ------------------------------------------------------------------------------


class OrderLookupInput(ToolInput):
    order_number: OrderNumber


class ItemView(BaseModel):
    sku: str
    name: str
    quantity: int
    unit_price_paise: int
    line_total_paise: int
    fulfilment: str = Field(description="not_shipped, or the status of its shipment.")


class ReturnView(BaseModel):
    reference: str
    sku: str | None
    status: str
    reason: str


class RefundView(BaseModel):
    reference: str
    amount_paise: int
    status: str
    reason: str


class GetOrderOutput(ToolOutput):
    order_number: str
    status: OrderStatus
    placed_on: str
    cancelled_on: str | None
    is_replacement: bool
    items: list[ItemView]
    subtotal_paise: int
    shipping_fee_paise: int
    total_paise: int
    total_display: str
    refunded_paise: int
    refund_pending_paise: int
    refundable_balance_paise: int
    returns: list[ReturnView]
    refunds: list[RefundView]


@tool(ToolKind.READ)
async def get_order(ctx: ToolContext, args: OrderLookupInput) -> GetOrderOutput:
    """Get one of the verified customer's orders: items, amounts, returns and refunds.

    Amounts are integer paise (₹1 = 100 paise); *_display fields are ready to speak.
    """
    order = await load_owned_order(ctx, args.order_number)
    shipments = {s.id: s for s in order.shipments}
    skus = {i.id: i.sku for i in order.items}

    def refunded(status: RefundStatus) -> int:
        return sum(r.amount_paise for r in order.refunds if r.status is status)

    return GetOrderOutput(
        order_number=order.order_number,
        status=order.status,
        placed_on=format_date(order.placed_at),
        cancelled_on=format_date(order.cancelled_at) if order.cancelled_at else None,
        is_replacement=order.replaces_order_id is not None,
        items=[
            ItemView(
                sku=i.sku,
                name=i.product_name,
                quantity=i.quantity,
                unit_price_paise=i.unit_price_paise,
                line_total_paise=i.line_total_paise,
                fulfilment=shipments[i.shipment_id].status if i.shipment_id else "not_shipped",
            )
            for i in order.items
        ],
        subtotal_paise=order.subtotal_paise,
        shipping_fee_paise=order.shipping_fee_paise,
        total_paise=order.total_paise,
        total_display=format_inr(order.total_paise),
        refunded_paise=refunded(RefundStatus.SUCCEEDED),
        refund_pending_paise=refunded(RefundStatus.PENDING),
        refundable_balance_paise=refundable_balance_paise(order),
        returns=[
            ReturnView(
                reference=return_ref(r.id),
                sku=skus.get(r.order_item_id) if r.order_item_id else None,
                status=r.status,
                reason=r.reason,
            )
            for r in order.returns
        ],
        refunds=[
            RefundView(
                reference=refund_ref(r.id),
                amount_paise=r.amount_paise,
                status=r.status,
                reason=r.reason,
            )
            for r in order.refunds
        ],
    )


# --- get_shipment_status --------------------------------------------------------------------


class EventView(BaseModel):
    code: ShipmentEventCode
    occurred_at: datetime
    on: str
    location: str | None
    description: str


class ShipmentView(BaseModel):
    tracking_number: str
    carrier: str
    status: ShipmentStatus
    shipped_on: str | None
    expected_by: str | None
    delivered_on: str | None
    is_overdue: bool
    item_names: list[str]
    latest_event: EventView | None
    events: list[EventView]


class GetShipmentStatusOutput(ToolOutput):
    order_number: str
    order_status: OrderStatus
    shipments: list[ShipmentView]
    unshipped_item_names: list[str]


def _event(e: ShipmentEvent) -> EventView:
    return EventView(
        code=e.code,
        occurred_at=e.occurred_at,
        on=format_date(e.occurred_at),
        location=e.location,
        description=e.description,
    )


def _shipment(s: Shipment, now: datetime) -> ShipmentView:
    events = [_event(e) for e in s.events]
    return ShipmentView(
        tracking_number=s.tracking_number,
        carrier=s.carrier,
        status=s.status,
        shipped_on=format_date(s.shipped_at) if s.shipped_at else None,
        expected_by=format_date(s.estimated_delivery_at) if s.estimated_delivery_at else None,
        delivered_on=format_date(s.delivered_at) if s.delivered_at else None,
        is_overdue=is_overdue(s, now),
        item_names=[i.product_name for i in s.items],
        latest_event=events[-1] if events else None,
        events=events,
    )


@tool(ToolKind.READ)
async def get_shipment_status(ctx: ToolContext, args: OrderLookupInput) -> GetShipmentStatusOutput:
    """Get delivery status for one of the verified customer's orders.

    Returns every shipment (an order can ship in parts) with carrier scans, ETA and an
    overdue flag, plus any items that haven't shipped yet.
    """
    order = await load_owned_order(ctx, args.order_number)
    now = ctx.now()
    return GetShipmentStatusOutput(
        order_number=order.order_number,
        order_status=order.status,
        shipments=[_shipment(s, now) for s in order.shipments],
        unshipped_item_names=[i.product_name for i in order.items if i.shipment_id is None],
    )


# --- check_return_eligibility ---------------------------------------------------------------


class CheckReturnEligibilityInput(ToolInput):
    order_number: OrderNumber
    sku: Sku | None = Field(None, description="Check one item; omit to check every item.")


class ItemEligibility(BaseModel):
    sku: str
    name: str
    outcome: Outcome
    reason: Reason | None
    explanation: str
    return_by: str | None
    refund_amount_paise: int
    refund_amount_display: str
    refund_requires_human: bool


class CheckReturnEligibilityOutput(ToolOutput):
    order_number: str
    items: list[ItemEligibility]


@tool(ToolKind.READ)
async def check_return_eligibility(
    ctx: ToolContext, args: CheckReturnEligibilityInput
) -> CheckReturnEligibilityOutput:
    """Check whether items on the verified customer's order can be returned, and for how much.

    Decided by policy code (return window, delivery, prior returns and refunds). An
    eligible item may still need a specialist to approve its refund (refund_requires_human).
    """
    order = await load_owned_order(ctx, args.order_number)
    items = [find_item(order, args.sku)] if args.sku else order.items
    now = ctx.now()
    results = []
    for item in items:
        a = evaluate_item_return(order, item, now)
        results.append(
            ItemEligibility(
                sku=item.sku,
                name=item.product_name,
                outcome=a.decision.outcome,
                reason=a.decision.reason,
                explanation=reason_message(
                    a.decision.reason, order.order_number, deadline=a.deadline
                ),
                return_by=format_date(a.deadline) if a.deadline else None,
                refund_amount_paise=a.refund_amount_paise,
                refund_amount_display=format_inr(a.refund_amount_paise),
                refund_requires_human=a.refund_decision.outcome is Outcome.REQUIRES_HUMAN,
            )
        )
    return CheckReturnEligibilityOutput(order_number=order.order_number, items=results)


# --- find_order_candidates ------------------------------------------------------------------


class FindOrderCandidatesInput(ToolInput):
    heard: str = Field(
        min_length=1,
        max_length=80,
        description='The order number exactly as heard, e.g. "fifteen eighty" or "VX 158".',
    )


class OrderCandidate(OrderSummary):
    match: str = Field(description="exact, sounds_alike, transposed, one_digit_off or partial")
    score: float


class FindOrderCandidatesOutput(ToolOutput):
    heard_digits: str | None
    exact_match: bool
    candidates: list[OrderCandidate]


@tool(ToolKind.READ)
async def find_order_candidates(
    ctx: ToolContext, args: FindOrderCandidatesInput
) -> FindOrderCandidatesOutput:
    """Match a possibly misheard order number against the verified customer's own orders.

    Returns up to 3 candidates, best first, with item names to help disambiguate. Read the
    chosen number back to the caller before acting on it, even for an exact match.
    """
    orders = await order_repo.list_orders_for_customer(ctx.db, ctx.require_customer_id())
    by_number = {o.order_number: o for o in orders}
    matches = match_order_numbers(args.heard, by_number)[:MAX_CANDIDATES]
    return FindOrderCandidatesOutput(
        heard_digits=parse_spoken_order_number(args.heard),
        exact_match=any(m.reason == "exact" for m in matches),
        candidates=[
            OrderCandidate(
                **_summary(by_number[m.order_number]).model_dump(), match=m.reason, score=m.score
            )
            for m in matches
        ],
    )
