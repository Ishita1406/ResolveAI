import uuid
from datetime import timedelta

import pytest
from redis.asyncio import Redis

from app.agent.session import (
    AgentSession,
    InMemorySessionStore,
    RedisSessionStore,
    SessionBusyError,
)
from app.config import get_settings
from app.policy.state import ActionState, PendingAction
from app.tools.context import ToolCallRecord
from tests.policy_helpers import NOW

pytestmark = pytest.mark.anyio


def _full_session() -> AgentSession:
    session = AgentSession(conversation_id=7, created_at=NOW, updated_at=NOW)
    state = session.state
    state.customer_id, state.customer_name, state.verified_at = 3, "Neha Gupta", NOW
    state.turn = 4
    state.history.append(ToolCallRecord("verify_customer", True, None, NOW))
    action = PendingAction(id="a1", tool="issue_refund", arguments={"order_number": "VX-4411"},
                           args_hash="h", idempotency_key="act-a1", proposed_at=NOW,
                           proposed_turn=3, summary="refund ₹1,198", amount_paise=119800,
                           verdict="approve", rule="refunds.caps")  # fmt: skip
    action.transition(ActionState.AWAITING_CONFIRMATION, at=NOW, turn=3)
    action.transition(ActionState.CONFIRMED, at=NOW + timedelta(seconds=5), turn=4, note="yes")
    state.actions[action.id] = action
    session.history.append([{"role": "user", "content": "<customer_message>hi</customer_message>"}])
    session.current_order_number = "VX-4411"
    return session


def _assert_same(loaded: AgentSession, original: AgentSession) -> None:
    assert loaded == original
    action = loaded.state.actions["a1"]
    assert isinstance(action, PendingAction) and action.state is ActionState.CONFIRMED
    assert action.history[-1].note == "yes"
    assert loaded.state.is_verified and loaded.state.history[0].tool == "verify_customer"


async def test_memory_store_round_trips_through_json():
    store, session = InMemorySessionStore(), _full_session()
    await store.save(session)
    _assert_same(await store.load(session.session_id), session)
    assert await store.load("missing") is None


async def test_memory_store_lock_is_exclusive():
    store = InMemorySessionStore()
    async with store.lock("s"):
        with pytest.raises(SessionBusyError):
            async with store.lock("s"):
                pass
    async with store.lock("s"):  # released
        pass


@pytest.fixture
async def redis():
    client = Redis.from_url(get_settings().redis_url)
    try:
        await client.ping()
    except Exception as exc:  # fail loudly: these tests are meant to run against Redis
        pytest.fail(f"Redis not reachable at REDIS_URL ({type(exc).__name__}); "
                    "start it with `docker compose up -d redis`")  # fmt: skip
    yield client
    await client.aclose()


@pytest.mark.redis
async def test_redis_store_round_trip_ttl_and_lock(redis):
    prefix = f"test:{uuid.uuid4().hex}:"
    store = RedisSessionStore(redis, ttl_s=60, lock_timeout_s=5, prefix=prefix)
    session = _full_session()
    try:
        await store.save(session)
        _assert_same(await store.load(session.session_id), session)
        assert 0 < await redis.ttl(f"{prefix}{session.session_id}") <= 60

        async with store.lock(session.session_id):
            other = RedisSessionStore(redis, ttl_s=60, lock_timeout_s=5, prefix=prefix)
            with pytest.raises(SessionBusyError):
                async with other.lock(session.session_id):
                    pass
        async with store.lock(session.session_id):
            pass
    finally:
        keys = [k async for k in redis.scan_iter(f"{prefix}*")]
        if keys:
            await redis.delete(*keys)
