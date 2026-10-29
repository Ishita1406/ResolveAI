"""Concurrent write calls on separate connections, with real commits (cleaned up afterwards)."""

import asyncio

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.models import Refund, ToolInvocation
from app.tools.registry import call_tool
from tests.tool_helpers import force_confirm, make_ctx, new_key, verify_as

pytestmark = [pytest.mark.anyio, pytest.mark.db]

REFUND = {"order_number": "VX-8015", "reason": "missing_delivery"}


@pytest.fixture
async def engine(test_db_settings):
    engine = create_async_engine(test_db_settings.database_url(), poolclass=NullPool)
    keys: list[str] = []
    yield engine, keys
    async with AsyncSession(engine) as session, session.begin():
        await session.execute(delete(Refund).where(Refund.idempotency_key.in_(keys)))
        await session.execute(
            delete(ToolInvocation).where(ToolInvocation.idempotency_key.in_(keys))
        )
    await engine.dispose()


async def _refund(engine, key):
    async with AsyncSession(engine) as session:
        ctx = make_ctx(session)
        await verify_as(ctx, "meera.pillai@example.com")
        # Each session holds its own confirmed action carrying the given key.
        args = force_confirm(ctx, "issue_refund", key, 3_499_00, **REFUND)
        return await call_tool(ctx, "issue_refund", args)


async def _refund_count(engine, keys):
    async with AsyncSession(engine) as session:
        stmt = select(func.count()).select_from(Refund).where(Refund.idempotency_key.in_(keys))
        return await session.scalar(stmt)


async def test_same_key_concurrently_refunds_once(engine):
    engine, keys = engine
    key = new_key()
    keys.append(key)
    a, b = await asyncio.gather(_refund(engine, key), _refund(engine, key))
    assert a.ok and b.ok, (a.error, b.error)
    assert sorted([a.replayed, b.replayed]) == [False, True]
    assert a.data == b.data
    assert await _refund_count(engine, keys) == 1


async def test_different_keys_concurrently_cannot_double_refund(engine):
    engine, keys = engine
    keys.extend([new_key(), new_key()])
    a, b = await asyncio.gather(_refund(engine, keys[0]), _refund(engine, keys[1]))
    outcomes = sorted([a.ok, b.ok])
    assert outcomes == [False, True]
    loser = a if not a.ok else b
    assert loser.error.details == {"reason": "already_refunded", "rule": "refunds.balance"}
    assert await _refund_count(engine, keys) == 1
