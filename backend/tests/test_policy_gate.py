"""The confirmation flow end to end, against the seeded test database."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from app.db.models import Order, PolicyDecisionLog, Refund
from app.db.seed import ANCHOR
from app.domain.enums import RefundReason, RefundStatus
from app.policy import gate
from app.policy.config import get_policy
from app.policy.facts import customer_refunds_in_window_paise
from app.policy.state import ActionState
from app.tools.errors import ErrorCode
from app.tools.registry import call_tool
from tests.tool_helpers import make_ctx, new_key, verify_as

pytestmark = [pytest.mark.anyio, pytest.mark.db]

REFUND_8015 = {"order_number": "VX-8015", "reason": "missing_delivery"}


class Clock:
    def __init__(self):
        self.now = ANCHOR

    def __call__(self):
        return self.now


async def _decisions(ctx, action_id):
    rows = await ctx.db.scalars(
        select(PolicyDecisionLog)
        .where(PolicyDecisionLog.action_id == action_id)
        .order_by(PolicyDecisionLog.id)
    )
    return [(r.stage, r.verdict, r.rule) for r in rows]


async def _propose_refund(ctx):
    await verify_as(ctx, "meera.pillai@example.com")
    proposed = await gate.propose(ctx, "issue_refund", REFUND_8015)
    assert proposed.ok, proposed.error
    return proposed.action


async def test_full_lifecycle_and_audit_trail(ctx):
    action = await _propose_refund(ctx)
    assert action.state is ActionState.AWAITING_CONFIRMATION
    assert action.summary == (
        "refund ₹3,499 to your original payment method for order VX-8015 (missing delivery)"
    )
    assert (action.amount_paise, action.amount_display) == (3_499_00, "₹3,499")

    gate.register_customer_turn(ctx)
    assert gate.confirm(ctx, action.action_id).action.state is ActionState.CONFIRMED
    executed = await gate.execute(ctx, action.action_id)
    assert executed.ok and executed.action.state is ActionState.EXECUTED
    assert executed.tool_result.data["amount_paise"] == 3_499_00

    stored = ctx.state.actions[action.action_id]
    assert [(t.source, t.target) for t in stored.history] == [
        (ActionState.PROPOSED, ActionState.AWAITING_CONFIRMATION),
        (ActionState.AWAITING_CONFIRMATION, ActionState.CONFIRMED),
        (ActionState.CONFIRMED, ActionState.EXECUTED),
    ]
    assert await _decisions(ctx, action.action_id) == [
        ("proposal", "approve", "refunds.caps"),
        ("execution", "approve", "refunds.caps"),
    ]


async def test_cannot_confirm_in_the_same_turn(ctx):
    action = await _propose_refund(ctx)
    early = gate.confirm(ctx, action.action_id)
    assert early.error.code is ErrorCode.ACTION_NOT_PENDING
    assert "later turn" in early.error.hint or "after the proposal" in early.error.hint
    gate.register_customer_turn(ctx)
    assert gate.confirm(ctx, action.action_id).ok


async def test_write_tool_refuses_without_confirmed_action(ctx):
    action = await _propose_refund(ctx)
    executed = await gate.execute(ctx, action.action_id)  # awaiting, not confirmed
    assert executed.error.code is ErrorCode.ACTION_NOT_CONFIRMED
    direct = await call_tool(ctx, "issue_refund", {**REFUND_8015, "idempotency_key": new_key()})
    assert direct.error.code is ErrorCode.ACTION_NOT_CONFIRMED
    assert await ctx.db.scalar(select(func.count()).select_from(Refund)
                               .where(Refund.idempotency_key.like("act-%"))) == 0  # fmt: skip


async def test_expires_after_configured_turns(ctx):
    action = await _propose_refund(ctx)
    n = get_policy().confirmation.expire_after_turns
    for _ in range(n):
        assert gate.register_customer_turn(ctx) == []
    expired = gate.register_customer_turn(ctx)
    assert [a.action_id for a in expired] == [action.action_id]
    confirm = gate.confirm(ctx, action.action_id)
    assert confirm.error.code is ErrorCode.ACTION_NOT_PENDING
    assert "timed out" in confirm.error.message
    assert (await gate.execute(ctx, action.action_id)).error.code is ErrorCode.ACTION_NOT_CONFIRMED


async def test_confirmation_on_last_allowed_turn_succeeds(ctx):
    action = await _propose_refund(ctx)
    for _ in range(get_policy().confirmation.expire_after_turns):
        gate.register_customer_turn(ctx)
    assert gate.confirm(ctx, action.action_id).ok


async def test_expires_after_configured_seconds(tx_db):
    ctx = make_ctx(tx_db)
    clock = Clock()
    ctx.clock = clock
    action = await _propose_refund(ctx)
    gate.register_customer_turn(ctx)
    clock.now = ANCHOR + timedelta(seconds=get_policy().confirmation.expire_after_seconds + 1)
    result = gate.confirm(ctx, action.action_id)
    assert result.error.code is ErrorCode.ACTION_NOT_PENDING
    assert ctx.state.actions[action.action_id].state is ActionState.EXPIRED


async def test_new_proposal_supersedes_the_unconfirmed_one(ctx):
    first = await _propose_refund(ctx)
    second = await gate.propose(
        ctx, "reship_order", {"order_number": "VX-8015", "issue": "missing_delivery"}
    )
    assert second.ok, second.error
    stored = ctx.state.actions[first.action_id]
    assert stored.state is ActionState.CANCELLED
    assert stored.history[-1].note == "superseded"
    gate.register_customer_turn(ctx)
    assert gate.confirm(ctx, first.action_id).error.code is ErrorCode.ACTION_NOT_PENDING
    assert gate.confirm(ctx, second.action.action_id).ok


async def test_cancel(ctx):
    action = await _propose_refund(ctx)
    gate.register_customer_turn(ctx)
    assert gate.cancel(ctx, action.action_id).action.state is ActionState.CANCELLED
    assert (await gate.execute(ctx, action.action_id)).error.code is ErrorCode.ACTION_NOT_CONFIRMED
    assert gate.cancel(ctx, action.action_id).error.code is ErrorCode.ACTION_NOT_PENDING


async def test_amount_change_after_confirmation_blocks_execution(ctx):
    action = await _propose_refund(ctx)
    gate.register_customer_turn(ctx)
    gate.confirm(ctx, action.action_id)
    # Something else refunds ₹1 on the order between confirmation and execution.
    order_id = await ctx.db.scalar(select(Order.id).where(Order.order_number == "VX-8015"))
    ctx.db.add(Refund(order_id=order_id, amount_paise=1_00, status=RefundStatus.PENDING,
                      reason=RefundReason.GOODWILL, idempotency_key=new_key(),
                      requested_at=ANCHOR))  # fmt: skip
    await ctx.db.commit()

    executed = await gate.execute(ctx, action.action_id)
    assert executed.error.code is ErrorCode.CONFIRMATION_STALE
    assert executed.error.details == {"confirmed_paise": 3_499_00, "now_paise": 3_498_00}
    assert ctx.state.actions[action.action_id].state is ActionState.REJECTED
    refunds = await ctx.db.scalar(
        select(func.count()).select_from(Refund).where(Refund.order_id == order_id)
    )
    assert refunds == 1  # only the one inserted above


async def test_policy_is_rechecked_at_execution_and_the_refusal_is_logged(tx_db):
    ctx = make_ctx(tx_db)
    clock = Clock()
    ctx.clock = clock
    await verify_as(ctx, "neha.gupta@example.com")
    args = {"order_number": "VX-4411", "sku": "TSH-BLK-M", "reason": "damaged"}
    proposed = await gate.propose(ctx, "create_return", args)
    assert proposed.ok
    gate.register_customer_turn(ctx)
    gate.confirm(ctx, proposed.action.action_id)
    clock.now = ANCHOR + timedelta(days=2)  # VX-4411's window closes at ANCHOR + 1 day

    executed = await gate.execute(ctx, proposed.action.action_id)
    assert executed.error.code is ErrorCode.NOT_ELIGIBLE
    assert executed.error.details["reason"] == "window_expired"
    assert ctx.state.actions[proposed.action.action_id].state is ActionState.REJECTED
    # The write rolled back, but its decision was persisted separately.
    assert await _decisions(ctx, proposed.action.action_id) == [
        ("proposal", "approve", "returns.window_days"),
        ("execution", "reject", "returns.window_days"),
    ]


async def test_requires_human_at_proposal_never_awaits_confirmation(ctx):
    await verify_as(ctx, "sameer.khan@example.com")
    proposed = await gate.propose(ctx, "issue_refund", {"order_number": "VX-3107",
                                                        "reason": "missing_delivery"})  # fmt: skip
    assert proposed.error.code is ErrorCode.REQUIRES_HUMAN
    assert proposed.action.state is ActionState.REJECTED
    assert proposed.action.verdict == "require_human"
    stmt = select(PolicyDecisionLog).where(PolicyDecisionLog.action_id == proposed.action.action_id)
    row = await ctx.db.scalar(stmt)
    assert row.facts["delivered_at"].startswith("2026-10-27")
    assert row.policy_version.startswith(get_policy().version)


async def test_proposal_validation_and_verification(ctx):
    unknown = await gate.propose(ctx, "delete_order", {})
    assert unknown.error.code is ErrorCode.UNKNOWN_TOOL
    unverified = await gate.propose(ctx, "issue_refund", REFUND_8015)
    assert unverified.error.code is ErrorCode.NOT_VERIFIED
    invalid = await gate.propose(ctx, "issue_refund", {"order_number": "VX-8015"})
    assert invalid.error.code is ErrorCode.INVALID_ARGUMENTS
    escalation = await gate.propose(ctx, "escalate_to_human", {
        "reason": "customer_requested", "summary": "Caller wants a person."})  # fmt: skip
    assert escalation.ok and escalation.action.summary == "connect you with a colleague"


async def test_unknown_action_id(ctx):
    assert gate.confirm(ctx, "nope").error.code is ErrorCode.ACTION_NOT_FOUND
    assert (await gate.execute(ctx, "nope")).error.code is ErrorCode.ACTION_NOT_FOUND


# --- facts ----------------------------------------------------------------------------------


async def test_customer_refund_window_sums_remedies_not_cancellations(ctx):
    vikram = await verify_as(ctx, "vikram.singh@example.com")
    assert await customer_refunds_in_window_paise(ctx.db, vikram, ANCHOR) == 3_499_00 + 1_299_00
    ananya = await verify_as(make_ctx(ctx.db), "ananya.rao@example.com")
    assert await customer_refunds_in_window_paise(ctx.db, ananya, ANCHOR) == 0


async def test_customer_refund_window_boundary(ctx):
    rahul = await verify_as(ctx, "rahul.verma@example.com")
    order_id = await ctx.db.scalar(select(Order.id).where(Order.order_number == "VX-1508"))
    window = timedelta(days=get_policy().refunds.per_customer_window_days)
    for amount, at in ((100, ANCHOR - window), (1, ANCHOR - window - timedelta(microseconds=1))):
        ctx.db.add(Refund(order_id=order_id, amount_paise=amount, status=RefundStatus.SUCCEEDED,
                          reason=RefundReason.GOODWILL, idempotency_key=new_key(),
                          requested_at=at))  # fmt: skip
    await ctx.db.flush()
    assert await customer_refunds_in_window_paise(ctx.db, rahul, ANCHOR) == 100
