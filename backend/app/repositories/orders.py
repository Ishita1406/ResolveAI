from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Order, Shipment
from app.domain.enums import ACTIVE_ORDER_STATUSES, OrderStatus

# Everything an agent needs to talk about one order, loaded up front (relationships are
# lazy="raise", so anything not listed here cannot be touched by accident).
_ORDER_DETAIL = (
    selectinload(Order.items),
    selectinload(Order.shipments).selectinload(Shipment.events),
    selectinload(Order.shipments).selectinload(Shipment.items),
    selectinload(Order.returns),
    selectinload(Order.refunds),
)


def _normalize_order_number(order_number: str) -> str:
    return order_number.strip().upper()


async def get_order_by_number(session: AsyncSession, order_number: str) -> Order | None:
    """Unscoped lookup. Customer-facing paths should use get_order_for_customer."""
    stmt = (
        select(Order)
        .where(Order.order_number == _normalize_order_number(order_number))
        .options(*_ORDER_DETAIL)
    )
    return await session.scalar(stmt)


async def get_order_for_customer(
    session: AsyncSession, customer_id: int, order_number: str, *, for_update: bool = False
) -> Order | None:
    """Ownership-scoped lookup: another customer's order is indistinguishable from none.

    for_update locks the order row until the transaction ends and refreshes any copy already
    in the session, so write paths re-check eligibility against current data.
    """
    stmt = (
        select(Order)
        .where(
            Order.order_number == _normalize_order_number(order_number),
            Order.customer_id == customer_id,
        )
        .options(*_ORDER_DETAIL)
    )
    if for_update:
        # NO KEY UPDATE: doesn't block (or deadlock with) FK inserts that reference the order.
        stmt = stmt.with_for_update(of=Order, key_share=True).execution_options(
            populate_existing=True
        )
    return await session.scalar(stmt)


async def list_orders_for_customer(
    session: AsyncSession,
    customer_id: int,
    *,
    statuses: Iterable[OrderStatus] | None = None,
) -> list[Order]:
    """Newest first, with items loaded."""
    stmt = (
        select(Order)
        .where(Order.customer_id == customer_id)
        .options(selectinload(Order.items))
        .order_by(Order.placed_at.desc(), Order.id.desc())
    )
    if statuses is not None:
        stmt = stmt.where(Order.status.in_(list(statuses)))
    return list(await session.scalars(stmt))


async def list_active_orders_for_customer(session: AsyncSession, customer_id: int) -> list[Order]:
    return await list_orders_for_customer(session, customer_id, statuses=ACTIVE_ORDER_STATUSES)


async def list_order_numbers_for_customer(session: AsyncSession, customer_id: int) -> list[str]:
    """Candidate set for matching a misheard order number against the caller's own orders."""
    stmt = (
        select(Order.order_number)
        .where(Order.customer_id == customer_id)
        .order_by(Order.order_number)
    )
    return list(await session.scalars(stmt))


async def get_shipment_by_tracking_number(
    session: AsyncSession, tracking_number: str
) -> Shipment | None:
    stmt = (
        select(Shipment)
        .where(Shipment.tracking_number == tracking_number.strip().upper())
        .options(selectinload(Shipment.events), selectinload(Shipment.items))
    )
    return await session.scalar(stmt)
