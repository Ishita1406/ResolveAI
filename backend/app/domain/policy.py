"""Deterministic business rules. The LLM proposes; these functions decide.

Functions take ORM objects with the relevant relationships loaded and an explicit `now`
(never the wall clock), so the same inputs always give the same decision.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from app.db.models import Order, OrderItem, Shipment
from app.domain.enums import OrderStatus, RefundStatus, ReturnStatus, ShipmentStatus

RETURN_WINDOW_DAYS = 30
AUTO_REFUND_CAP_PAISE = 10_000_00  # ₹10,000
FREE_SHIPPING_THRESHOLD_PAISE = 499_00
SHIPPING_FEE_PAISE = 40_00
# A parcel this many days past its ETA (and not delivered) counts as missing.
MISSING_DELIVERY_GRACE_DAYS = 3
MAX_VERIFICATION_ATTEMPTS = 3


class Outcome(StrEnum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    REQUIRES_HUMAN = "requires_human"


class Reason(StrEnum):
    ORDER_CANCELLED = "order_cancelled"
    NOT_SHIPPED = "not_shipped"
    NOT_DELIVERED = "not_delivered"
    WINDOW_EXPIRED = "window_expired"
    ALREADY_RETURNED = "already_returned"
    ALREADY_REFUNDED = "already_refunded"
    ALREADY_RESOLVED = "already_resolved"
    NO_RETURN_ON_FILE = "no_return_on_file"
    NOT_YET_OVERDUE = "not_yet_overdue"
    DELIVERED_PER_CARRIER = "delivered_per_carrier"
    RETURNED_TO_SENDER = "returned_to_sender"
    ABOVE_AUTO_REFUND_CAP = "above_auto_refund_cap"
    GOODWILL_NEEDS_APPROVAL = "goodwill_needs_approval"
    LOST_IN_TRANSIT = "lost_in_transit"
    OVERDUE = "overdue"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    reason: Reason | None = None

    @property
    def eligible(self) -> bool:
        return self.outcome is Outcome.ELIGIBLE


ELIGIBLE = Decision(Outcome.ELIGIBLE)


def _no(reason: Reason) -> Decision:
    return Decision(Outcome.NOT_ELIGIBLE, reason)


def _human(reason: Reason) -> Decision:
    return Decision(Outcome.REQUIRES_HUMAN, reason)


def shipment_for(order: Order, item: OrderItem) -> Shipment | None:
    return next((s for s in order.shipments if s.id == item.shipment_id), None)


def refundable_balance_paise(order: Order) -> int:
    """Order total minus refunds that are done or in flight. Never negative."""
    committed = sum(
        r.amount_paise
        for r in order.refunds
        if r.status in (RefundStatus.SUCCEEDED, RefundStatus.PENDING)
    )
    return max(order.total_paise - committed, 0)


def return_deadline(delivered_at: datetime) -> datetime:
    # Exact-timestamp window: eligible while now <= delivered_at + 30 days.
    return delivered_at + timedelta(days=RETURN_WINDOW_DAYS)


def check_refund_cap(amount_paise: int) -> Decision:
    if amount_paise > AUTO_REFUND_CAP_PAISE:
        return _human(Reason.ABOVE_AUTO_REFUND_CAP)
    return ELIGIBLE


@dataclass(frozen=True)
class ReturnAssessment:
    decision: Decision
    deadline: datetime | None
    refund_amount_paise: int
    refund_decision: Decision


def evaluate_item_return(order: Order, item: OrderItem, now: datetime) -> ReturnAssessment:
    shipment = shipment_for(order, item)
    deadline = (
        return_deadline(shipment.delivered_at) if shipment and shipment.delivered_at else None
    )
    balance = refundable_balance_paise(order)
    amount = min(item.line_total_paise, balance)

    def result(decision: Decision) -> ReturnAssessment:
        refund = check_refund_cap(amount) if decision.eligible else decision
        return ReturnAssessment(decision, deadline, amount, refund)

    if order.status is OrderStatus.CANCELLED:
        return result(_no(Reason.ORDER_CANCELLED))
    if shipment is None:
        return result(_no(Reason.NOT_SHIPPED))
    if deadline is None:
        return result(_no(Reason.NOT_DELIVERED))
    if any(
        r.order_item_id == item.id and r.status is not ReturnStatus.REJECTED for r in order.returns
    ):
        return result(_no(Reason.ALREADY_RETURNED))
    if balance <= 0:
        return result(_no(Reason.ALREADY_REFUNDED))
    if now > deadline:
        return result(_no(Reason.WINDOW_EXPIRED))
    return result(ELIGIBLE)


def is_overdue(shipment: Shipment, now: datetime, grace_days: int = 0) -> bool:
    eta = shipment.estimated_delivery_at
    return (
        shipment.delivered_at is None and eta is not None and now > eta + timedelta(days=grace_days)
    )


def evaluate_missing_delivery(shipment: Shipment | None, now: datetime) -> Decision:
    """Whether a reship or refund for a non-arriving parcel can be done automatically."""
    if shipment is None:
        return _no(Reason.NOT_SHIPPED)
    match shipment.status:
        case ShipmentStatus.LOST:
            return Decision(Outcome.ELIGIBLE, Reason.LOST_IN_TRANSIT)
        case ShipmentStatus.DELIVERED:
            # Carrier says delivered: needs an investigation, not an automatic remedy.
            return _human(Reason.DELIVERED_PER_CARRIER)
        case ShipmentStatus.RETURNED_TO_SENDER:
            return _human(Reason.RETURNED_TO_SENDER)
    if is_overdue(shipment, now, MISSING_DELIVERY_GRACE_DAYS):
        return Decision(Outcome.ELIGIBLE, Reason.OVERDUE)
    return _no(Reason.NOT_YET_OVERDUE)
