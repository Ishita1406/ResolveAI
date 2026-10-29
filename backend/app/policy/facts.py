from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Customer, Order, Refund, Ticket
from app.domain.enums import RefundReason, RefundStatus, TicketCategory, TicketStatus
from app.policy.config import get_policy
from app.policy.engine import PolicyFacts

OPEN_TICKET_STATUSES = (TicketStatus.OPEN, TicketStatus.PENDING_CUSTOMER, TicketStatus.ESCALATED)


async def lock_customer(db: AsyncSession, customer_id: int) -> None:
    """Serialize money-moving writes per customer (the per-customer cap spans orders).

    Lock order is always customer, then order. FOR NO KEY UPDATE, not FOR UPDATE: inserting
    a row with a foreign key to customers (e.g. the idempotency ledger) takes KEY SHARE on
    the customer, which FOR UPDATE would conflict with, deadlocking two concurrent writers.
    """
    stmt = select(Customer.id).where(Customer.id == customer_id).with_for_update(key_share=True)
    await db.execute(stmt)


async def customer_refunds_in_window_paise(
    db: AsyncSession, customer_id: int, now: datetime
) -> int:
    """Pending + succeeded refunds requested in the rolling window, excluding cancellations
    (money back for goods never sent is not a remedy and doesn't count toward the cap)."""
    since = now - timedelta(days=get_policy().refunds.per_customer_window_days)
    stmt = (
        select(func.coalesce(func.sum(Refund.amount_paise), 0))
        .join(Order, Order.id == Refund.order_id)
        .where(
            Order.customer_id == customer_id,
            Refund.status.in_((RefundStatus.PENDING, RefundStatus.SUCCEEDED)),
            Refund.reason != RefundReason.CANCELLATION,
            Refund.requested_at >= since,
        )
    )
    return int(await db.scalar(stmt))


async def load_facts(
    db: AsyncSession,
    *,
    now: datetime,
    customer_id: int | None,
    order: Order | None = None,
    ticket_category: TicketCategory | None = None,
) -> PolicyFacts:
    refunds = await customer_refunds_in_window_paise(db, customer_id, now) if customer_id else 0
    has_replacement = False
    open_ticket = None
    if order is not None:
        has_replacement = (
            await db.scalar(select(Order.id).where(Order.replaces_order_id == order.id).limit(1))
            is not None
        )
        if ticket_category is not None:
            open_ticket = await db.scalar(
                select(Ticket.id).where(
                    Ticket.order_id == order.id,
                    Ticket.category == ticket_category,
                    Ticket.status.in_(OPEN_TICKET_STATUSES),
                )
            )
    return PolicyFacts(
        now=now,
        order=order,
        customer_refunds_in_window_paise=refunds,
        has_replacement=has_replacement,
        open_ticket_reference=f"TCK-{open_ticket:06d}" if open_ticket else None,
    )
