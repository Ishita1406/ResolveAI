"""Per-turn metrics in the turn_metrics table, and LLM cost.

Two writers share a row through the turn id: the agent (usage, cost, model and tool
timings, written when the turn is saved) and the voice session (time to first audio and its
stage breakdown, written when the reply has been spoken). Each upserts only its own
columns, so the order they arrive in doesn't matter. Writes are best-effort, like the
transcript: a failure is logged and never affects the call.
"""

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TurnMetric

log = logging.getLogger(__name__)

DbFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
COLUMNS = frozenset(c.name for c in TurnMetric.__table__.columns) - {"turn_id"}


def cost_usd(
    model: str, usage: dict[str, Any], prices: dict[str, dict[str, float]]
) -> float | None:
    """USD for one call from per-million-token prices ({"input": .., "output": ..}); None
    when the model has no price (Langfuse then prices it from its own model table)."""
    price = prices.get(model)
    if price is None:
        return None
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return round(
        (prompt * price.get("input", 0.0) + completion * price.get("output", 0.0)) / 1e6, 8
    )


async def save(db: AsyncSession, turn_id: str, **values: Any) -> None:
    """Upsert this writer's columns for a turn. Unknown keys are a programming error."""
    unknown = set(values) - COLUMNS
    if unknown:
        raise ValueError(f"unknown turn_metrics columns: {sorted(unknown)}")
    row = {k: v for k, v in values.items() if v is not None}
    statement = insert(TurnMetric).values(turn_id=turn_id, **row)
    if row:
        statement = statement.on_conflict_do_update(index_elements=["turn_id"], set_=row)
    else:
        statement = statement.on_conflict_do_nothing(index_elements=["turn_id"])
    await db.execute(statement)
    await db.commit()


async def save_quietly(db_factory: DbFactory, turn_id: str, **values: Any) -> None:
    try:
        async with db_factory() as db:
            await save(db, turn_id, **values)
    except Exception:
        log.exception("could not save turn metrics", extra={"metrics_turn_id": turn_id})
