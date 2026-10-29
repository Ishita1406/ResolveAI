"""Write tools through the real path (propose -> confirm -> execute), plus the idempotency
ledger beneath it (force_confirm)."""

import pytest
from sqlalchemy import func, select

from app.db.models import Order, PolicyDecisionLog, Refund, Return, Ticket, ToolInvocation
from app.domain.order_ids import spoken_confusables
from app.policy import gate
from app.tools.errors import ErrorCode
from app.tools.registry import call_tool
from tests.tool_helpers import force_confirm, last_action, make_ctx, new_key, run_write, verify_as

pytestmark = [pytest.mark.anyio, pytest.mark.db]


async def ok(ctx, tool, **args):
    result = await run_write(ctx, tool, **args)
    assert result.ok, result.error
    return result.data


async def err(ctx, tool, **args):
    result = await run_write(ctx, tool, **args)
    assert not result.ok, result.data
    return result.error


async def count(ctx, model, *where):
    return await ctx.db.scalar(select(func.count()).select_from(model).where(*where))


# --- create_return --------------------------------------------------------------------------


def forget_session_actions(ctx):
    """As on a later call: the policy, not this session's memory, must refuse the repeat."""
    ctx.state.actions.clear()


async def test_create_return_and_replay(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    data = await ok(ctx, "create_return", order_number="VX-4411", sku="TSH-BLK-M",
                    reason="damaged")  # fmt: skip
    assert data["return_reference"].startswith("RET-")
    assert data["status"] == "requested"
    assert data["refund_amount_paise"] == 2 * 599_00

    action = last_action(ctx)
    replay = await gate.execute(ctx, action.id)  # e.g. a retry after a timeout
    assert replay.ok and replay.tool_result.replayed
    assert replay.tool_result.data == data
    assert await count(ctx, Return, Return.idempotency_key == action.idempotency_key) == 1


async def test_model_supplied_idempotency_key_is_ignored(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    await ok(ctx, "create_return", order_number="VX-4411", sku="TSH-BLK-M", reason="damaged",
             idempotency_key="chosen-by-the-model")  # fmt: skip
    assert last_action(ctx).idempotency_key.startswith("act-")
    assert await count(ctx, Return, Return.idempotency_key == "chosen-by-the-model") == 0


async def test_reused_key_with_different_arguments_is_rejected_at_both_layers(tx_db):
    ctx = make_ctx(tx_db)
    await verify_as(ctx, "neha.gupta@example.com")
    key = new_key()
    base = {"order_number": "VX-4411", "sku": "TSH-BLK-M"}
    first = force_confirm(ctx, "create_return", key, 2 * 599_00, reason="damaged", **base)
    assert (await call_tool(ctx, "create_return", first)).ok

    # Same session: the confirmed action for this key has other arguments.
    changed = {**first, "reason": "wrong_item"}
    result = await call_tool(ctx, "create_return", changed)
    assert result.error.code is ErrorCode.ACTION_NOT_CONFIRMED

    # Even with a confirmed action carrying the key (another session), the ledger refuses.
    other = make_ctx(tx_db)
    await verify_as(other, "neha.gupta@example.com")
    second = force_confirm(other, "create_return", key, 2 * 599_00, reason="wrong_item", **base)
    result = await call_tool(other, "create_return", second)
    assert result.error.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


async def test_ledger_never_replays_another_customers_result(tx_db):
    rahul, rohan = make_ctx(tx_db), make_ctx(tx_db)
    await verify_as(rahul, "rahul.verma@example.com")
    await verify_as(rohan, "rohan.iyer@example.com")
    key = new_key()
    args = {"category": "other", "summary": "Question about invoices."}
    first = await call_tool(
        rahul, "create_ticket", force_confirm(rahul, "create_ticket", key, **args)
    )
    assert first.ok
    second = await call_tool(
        rohan, "create_ticket", force_confirm(rohan, "create_ticket", key, **args)
    )
    assert second.error.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


async def test_ineligible_return_is_refused_at_proposal_and_logged(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    error = await err(ctx, "create_return", order_number="VX-4410", sku="JNS-IND-32",
                      reason="no_longer_needed")  # fmt: skip
    assert error.code is ErrorCode.NOT_ELIGIBLE
    assert error.details == {"reason": "window_expired", "rule": "returns.window_days"}
    action = last_action(ctx)
    assert action.state == "rejected"
    assert await count(ctx, Return, Return.idempotency_key == action.idempotency_key) == 0
    assert await count(ctx, ToolInvocation) == 0
    logged = await ctx.db.scalar(
        select(PolicyDecisionLog).where(PolicyDecisionLog.action_id == action.id)
    )
    assert (logged.stage, logged.verdict, logged.rule) == (
        "proposal",
        "reject",
        "returns.window_days",
    )


async def test_unknown_item_fails_before_policy(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    error = await err(ctx, "create_return", order_number="VX-4411", sku="NOPE-1", reason="damaged")
    assert error.code is ErrorCode.ITEM_NOT_FOUND
    assert ctx.state.actions == {}


async def test_second_return_for_same_item_is_refused(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    args = {"order_number": "VX-4411", "sku": "TSH-BLK-M", "reason": "damaged"}
    await ok(ctx, "create_return", **args)
    forget_session_actions(ctx)
    error = await err(ctx, "create_return", **args)
    assert error.details["reason"] == "already_returned"


async def test_cannot_return_another_customers_order(ctx):
    await verify_as(ctx, "rohan.iyer@example.com")
    before = await count(ctx, Return)
    error = await err(ctx, "create_return", order_number="VX-1350", sku="WCH-SMT-02",
                      reason="damaged")  # fmt: skip
    assert error.code is ErrorCode.ORDER_NOT_FOUND
    assert await count(ctx, Return) == before


# --- issue_refund ---------------------------------------------------------------------------


async def test_refund_after_return_is_idempotent_and_single(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    await ok(ctx, "create_return", order_number="VX-4411", sku="TSH-BLK-M", reason="damaged")
    args = {"order_number": "VX-4411", "reason": "return", "sku": "TSH-BLK-M"}
    data = await ok(ctx, "issue_refund", **args)
    assert (data["amount_paise"], data["amount_display"], data["status"]) == (
        2 * 599_00, "₹1,198", "pending"
    )  # fmt: skip
    key = last_action(ctx).idempotency_key
    replay = await gate.execute(ctx, last_action(ctx).id)
    assert replay.tool_result.replayed and replay.tool_result.data == data
    assert await count(ctx, Refund, Refund.idempotency_key == key) == 1

    forget_session_actions(ctx)
    retry = await err(ctx, "issue_refund", **args)  # a new action cannot refund twice
    assert retry.details["reason"] == "already_refunded"


async def test_refund_above_order_cap_needs_human_and_writes_nothing(ctx):
    await verify_as(ctx, "arjun.malhotra@example.com")
    ret = await ok(ctx, "create_return", order_number="VX-7302", sku="LAP-PRO-16",
                   reason="damaged")  # fmt: skip
    assert ret["refund_requires_human"] is True
    before = await count(ctx, Refund)
    error = await err(ctx, "issue_refund", order_number="VX-7302", reason="return",
                      sku="LAP-PRO-16")  # fmt: skip
    assert error.code is ErrorCode.REQUIRES_HUMAN
    assert error.details == {
        "reason": "over_order_refund_cap",
        "rule": "refunds.per_order_cap_inr",
    }
    assert "₹84,999" in error.message
    assert await count(ctx, Refund) == before


@pytest.mark.parametrize(
    ("email", "args", "code", "reason"),
    [
        ("rahul.verma@example.com",
         {"order_number": "VX-1508", "reason": "return", "sku": "PWB-10K"},
         ErrorCode.NOT_ELIGIBLE, "no_return_on_file"),
        ("vikram.singh@example.com", {"order_number": "VX-5120", "reason": "missing_delivery"},
         ErrorCode.NOT_ELIGIBLE, "already_refunded"),
        ("ananya.rao@example.com", {"order_number": "VX-6031", "reason": "missing_delivery"},
         ErrorCode.NOT_ELIGIBLE, "order_cancelled"),
        ("rahul.verma@example.com", {"order_number": "VX-1508", "reason": "goodwill"},
         ErrorCode.REQUIRES_HUMAN, "goodwill_requested"),
        ("sameer.khan@example.com", {"order_number": "VX-3107", "reason": "missing_delivery"},
         ErrorCode.REQUIRES_HUMAN, "delivered_per_carrier"),
        ("kavya.nair@example.com", {"order_number": "VX-3302", "reason": "missing_delivery"},
         ErrorCode.NOT_ELIGIBLE, "not_yet_overdue"),
        ("aarav.mehta@example.com",
         {"order_number": "VX-2201", "reason": "missing_delivery", "sku": "LMP-DSK-LED"},
         ErrorCode.NOT_ELIGIBLE, "not_shipped"),
    ],
)  # fmt: skip
async def test_refund_policy_refusals(ctx, email, args, code, reason):
    await verify_as(ctx, email)
    before = await count(ctx, Refund)
    error = await err(ctx, "issue_refund", **args)
    assert (error.code, error.details["reason"]) == (code, reason)
    assert await count(ctx, Refund) == before


async def test_whole_order_claim_with_a_carrier_delivered_part_needs_human(ctx):
    await verify_as(ctx, "aarav.mehta@example.com")
    error = await err(ctx, "issue_refund", order_number="VX-2201", reason="missing_delivery")
    assert error.details["reason"] == "delivered_per_carrier"


async def test_not_yet_overdue_message_gives_eta(ctx):
    await verify_as(ctx, "kavya.nair@example.com")
    error = await err(ctx, "issue_refund", order_number="VX-3302", reason="missing_delivery")
    assert error.message == (
        "Order VX-3302 is still on its way and isn't late yet. It's expected by 31 October 2026."
    )


async def test_missing_delivery_refund_then_reship_is_already_resolved(ctx):
    await verify_as(ctx, "meera.pillai@example.com")
    data = await ok(ctx, "issue_refund", order_number="VX-8015", reason="missing_delivery")
    assert data["amount_paise"] == 3_499_00
    error = await err(ctx, "reship_order", order_number="VX-8015", issue="missing_delivery")
    assert error.details["reason"] == "already_resolved"


# --- reship_order ---------------------------------------------------------------------------


async def test_reship_creates_free_replacement_once(ctx):
    customer_id = await verify_as(ctx, "meera.pillai@example.com")
    data = await ok(ctx, "reship_order", order_number="VX-8015", issue="missing_delivery")
    number = data["replacement_order_number"]
    assert data["item_names"] == ["Bluetooth speaker"]

    replacement = await ctx.db.scalar(select(Order).where(Order.order_number == number))
    original_id = await ctx.db.scalar(select(Order.id).where(Order.order_number == "VX-8015"))
    assert (replacement.customer_id, replacement.total_paise) == (customer_id, 0)
    assert replacement.replaces_order_id == original_id
    assert "VX-8015" not in spoken_confusables(number)

    replay = await gate.execute(ctx, last_action(ctx).id)
    assert replay.tool_result.data["replacement_order_number"] == number
    forget_session_actions(ctx)
    again = await err(ctx, "reship_order", order_number="VX-8015", issue="missing_delivery")
    assert again.details["reason"] == "already_resolved"
    refund = await err(ctx, "issue_refund", order_number="VX-8015", reason="missing_delivery")
    assert refund.details["reason"] == "already_resolved"
    assert "no refund or replacement is available" in refund.hint


async def test_reship_damaged_item_inside_window(ctx):
    await verify_as(ctx, "rahul.verma@example.com")
    data = await ok(ctx, "reship_order", order_number="VX-1508", issue="damaged", sku="PWB-10K")
    assert data["item_names"] == ["Power bank 10,000 mAh"]


async def test_reship_not_offered_for_items_no_longer_needed(ctx):
    await verify_as(ctx, "rahul.verma@example.com")
    error = await err(ctx, "reship_order", order_number="VX-1508", issue="no_longer_needed",
                      sku="PWB-10K")  # fmt: skip
    assert error.details == {"reason": "remedy_not_allowed", "rule": "remedies.no_longer_needed"}
    assert error.message == (
        "For an item you no longer need, I can offer a refund, but not a free replacement."
    )


async def test_reship_refused_when_carrier_says_delivered(ctx):
    await verify_as(ctx, "sameer.khan@example.com")
    error = await err(ctx, "reship_order", order_number="VX-3107", issue="missing_delivery")
    assert error.code is ErrorCode.REQUIRES_HUMAN
    assert "escalate_to_human" in error.hint


# --- create_ticket --------------------------------------------------------------------------


async def test_duplicate_open_ticket_is_refused(ctx):
    await verify_as(ctx, "sameer.khan@example.com")
    error = await err(ctx, "create_ticket", category="missing_delivery", order_number="VX-3107",
                      summary="Customer says the earbuds never arrived.")  # fmt: skip
    assert error.code is ErrorCode.TICKET_ALREADY_OPEN
    assert error.details["ticket_reference"].startswith("TCK-")
    assert error.details["rule"] == "tickets.no_duplicates"


async def test_create_ticket(ctx):
    customer_id = await verify_as(ctx, "rahul.verma@example.com")
    data = await ok(ctx, "create_ticket", category="damaged_product", order_number="vx 1508",
                    summary="Power bank gets hot while charging.", priority="high")  # fmt: skip
    assert data["order_number"] == "VX-1508"
    ticket = await ctx.db.scalar(select(Ticket).order_by(Ticket.id.desc()).limit(1))
    assert (ticket.customer_id, ticket.status, ticket.priority) == (customer_id, "open", "high")


# --- escalate_to_human ----------------------------------------------------------------------


async def test_escalation_works_unverified_and_records_attempts(ctx):
    await call_tool(ctx, "verify_customer", {"email": "priya.sharma@example.com",
                                             "postal_code": "000000"})  # fmt: skip
    data = await ok(ctx, "escalate_to_human", reason="verification_failed",
                    summary="Caller could not verify; asked about an order.",
                    order_number="thirteen fifty")  # fmt: skip
    handoff = data["handoff"]
    assert (handoff["verified"], handoff["customer"], handoff["order_number"]) == (
        False, None, None
    )  # fmt: skip
    assert handoff["issue"] == "Caller could not verify; asked about an order."
    assert handoff["steps_taken"] == ["verify_customer: failed (verification_failed)"]
    assert handoff["caller_order_reference"] == "thirteen fifty"
    assert [(a["tool"], a["error_code"]) for a in handoff["actions_attempted"]] == [
        ("verify_customer", "verification_failed")
    ]
    ticket = await ctx.db.scalar(select(Ticket).order_by(Ticket.id.desc()).limit(1))
    assert (ticket.customer_id, ticket.order_id, ticket.status) == (None, None, "escalated")
    assert ticket.details == handoff


async def test_escalation_never_links_another_customers_order(ctx):
    await verify_as(ctx, "rohan.iyer@example.com")
    data = await ok(ctx, "escalate_to_human", reason="customer_requested",
                    summary="Wants a person about order thirteen fifty.",
                    order_number="VX-1350")  # fmt: skip
    assert data["handoff"]["order_number"] is None
    assert data["handoff"]["caller_order_reference"] == "VX-1350"


async def test_escalation_links_own_order_and_lists_attempts(ctx):
    await verify_as(ctx, "arjun.malhotra@example.com")
    ret = await ok(ctx, "create_return", order_number="VX-7302", sku="LAP-PRO-16",
                   reason="damaged")  # fmt: skip
    assert ret["refund_requires_human"]
    data = await ok(ctx, "escalate_to_human", reason="policy_requires_review",
                    summary="Damaged laptop; refund above automatic limit.",
                    order_number="VX-7302", category="damaged_product")  # fmt: skip
    handoff = data["handoff"]
    assert handoff["verified"] is True
    assert handoff["customer"]["name"] == "Arjun Malhotra"
    assert handoff["order_number"] == "VX-7302"
    assert any(s.startswith("proposed: start a return for the ProBook 16 laptop")
               and s.endswith("-> executed") for s in handoff["steps_taken"])  # fmt: skip
    assert [a["tool"] for a in handoff["actions_attempted"]] == ["verify_customer", "create_return"]
