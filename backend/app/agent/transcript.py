"""Conversation transcript in Postgres (conversations / conversation_turns).

Writes are best-effort: a transcript failure is logged and never fails the customer's turn.
"""

import logging
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, ConversationTurn
from app.domain.enums import ConversationChannel, TurnRole
from app.tools.context import SessionState

log = logging.getLogger(__name__)


async def start_conversation(
    db: AsyncSession, *, channel: ConversationChannel, now: datetime
) -> int:
    conversation = Conversation(channel=channel, started_at=now)
    db.add(conversation)
    await db.flush()
    conversation_id = conversation.id  # read before commit expires the object
    await db.commit()
    return conversation_id


async def record_turns(
    db: AsyncSession,
    conversation_id: int,
    entries: list[tuple[TurnRole, str]],
    *,
    state: SessionState,
    now: datetime,
) -> int | None:
    """Append turns and persist verification once it has happened. Returns the first new
    turn's index (None if nothing could be recorded)."""
    try:
        start = await db.scalar(
            select(func.coalesce(func.max(ConversationTurn.turn_index) + 1, 0)).where(
                ConversationTurn.conversation_id == conversation_id
            )
        )
        db.add_all(
            ConversationTurn(
                conversation_id=conversation_id,
                turn_index=start + i,
                role=role,
                content=content,
                created_at=now,
            )
            for i, (role, content) in enumerate(entries)
            if content
        )
        if state.customer_id is not None:
            await db.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id, Conversation.customer_id.is_(None))
                .values(customer_id=state.customer_id, verified_at=state.verified_at)
            )
        await db.commit()
        return start
    except Exception:
        log.exception("could not record transcript for conversation %s", conversation_id)
        await db.rollback()
        return None


async def end_conversation(db: AsyncSession, conversation_id: int, *, now: datetime) -> bool:
    """Record when the call or chat ended. Only the first end counts (a chat can reconnect
    and end again); True if this was it."""
    try:
        result = await db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.ended_at.is_(None))
            .values(ended_at=now)
        )
        await db.commit()
        return result.rowcount == 1
    except Exception:
        log.exception("could not record the end of conversation %s", conversation_id)
        await db.rollback()
        return False
