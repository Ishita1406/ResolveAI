from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Ticket
from app.domain.enums import TicketStatus

OPEN_TICKET_STATUSES = (TicketStatus.OPEN, TicketStatus.PENDING_CUSTOMER, TicketStatus.ESCALATED)


async def get_ticket(session: AsyncSession, ticket_id: int) -> Ticket | None:
    return await session.get(Ticket, ticket_id)


async def list_tickets_for_customer(
    session: AsyncSession, customer_id: int, *, open_only: bool = False
) -> list[Ticket]:
    stmt = (
        select(Ticket)
        .where(Ticket.customer_id == customer_id)
        .order_by(Ticket.created_at.desc(), Ticket.id.desc())
    )
    if open_only:
        stmt = stmt.where(Ticket.status.in_(OPEN_TICKET_STATUSES))
    return list(await session.scalars(stmt))


async def list_tickets_for_order(session: AsyncSession, order_id: int) -> list[Ticket]:
    stmt = select(Ticket).where(Ticket.order_id == order_id).order_by(Ticket.id)
    return list(await session.scalars(stmt))


async def get_conversation(session: AsyncSession, conversation_id: int) -> Conversation | None:
    """With turns loaded, in turn order."""
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.turns))
    )
    return await session.scalar(stmt)
