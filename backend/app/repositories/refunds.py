from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Refund, Return
from app.domain.enums import RefundStatus


async def list_refunds_for_order(session: AsyncSession, order_id: int) -> list[Refund]:
    stmt = select(Refund).where(Refund.order_id == order_id).order_by(Refund.id)
    return list(await session.scalars(stmt))


async def get_refund_totals_paise(session: AsyncSession, order_id: int) -> dict[RefundStatus, int]:
    """Sum of refund amounts per status (every status present, zero if none)."""
    stmt = (
        select(Refund.status, func.sum(Refund.amount_paise))
        .where(Refund.order_id == order_id)
        .group_by(Refund.status)
    )
    totals = dict.fromkeys(RefundStatus, 0)
    for status, amount in await session.execute(stmt):
        totals[status] = int(amount)
    return totals


async def get_refund_by_idempotency_key(session: AsyncSession, key: str) -> Refund | None:
    return await session.scalar(select(Refund).where(Refund.idempotency_key == key))


async def list_returns_for_order(session: AsyncSession, order_id: int) -> list[Return]:
    stmt = select(Return).where(Return.order_id == order_id).order_by(Return.id)
    return list(await session.scalars(stmt))
