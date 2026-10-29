import pytest
from sqlalchemy import func, select

from app.db.models import Order, Refund, Return, Ticket, ToolInvocation
from app.domain.order_ids import spoken_confusables
from app.tools.errors import ErrorCode
from app.tools.registry import call_tool
from tests.tool_helpers import make_ctx, new_key, verify_as

pytestmark = [pytest.mark.anyio, pytest.mark.db]


async def call(ctx, tool, **args):
    args.setdefault("idempotency_key", new_key())
    return await call_tool(ctx, tool, args)


async def ok(ctx, tool, **args):
    result = await call(ctx, tool, **args)
    assert result.ok, result.error
    return result.data


async def err(ctx, tool, **args):
    result = await call(ctx, tool, **args)
    assert not result.ok, result.data
    return result.error


async def count(ctx, model, *where):
    return await ctx.db.scalar(select(func.count()).select_from(model).where(*where))


# --- create_return --------------------------------------------------------------------------


async def test_create_return_and_replay(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    key = new_key()
    args = {"order_number": "VX-4411", "sku": "TSH-BLK-M", "reason": "damaged"}
    first = await call(ctx, "create_return", idempotency_key=key, **args)
    assert first.ok and not first.replayed
    assert first.data["return_reference"].startswith("RET-")
    assert first.data["status"] == "requested"
    assert first.data["refund_amount_paise"] == 2 * 599_00

    second = await call(ctx, "create_return", idempotency_key=key, **args)
    assert second.ok and second.replayed
    assert second.data == first.data
    assert await count(ctx, Return, Return.idempotency_key == key) == 1


async def test_reused_key_with_different_arguments_is_rejected(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    key = new_key()
    await ok(ctx, "create_return", idempotency_key=key, order_number="VX-4411", sku="TSH-BLK-M",
             reason="damaged")  # fmt: skip
    error = await err(ctx, "create_return", idempotency_key=key, order_number="VX-4411",
                      sku="TSH-BLK-M", reason="wrong_item")  # fmt: skip
    assert error.code is ErrorCode.IDEMPOTENCY_KEY_REUSED
    error = await err(ctx, "create_ticket", idempotency_key=key, category="other",
                      summary="Same key, different tool.")  # fmt: skip
    assert error.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


async def test_key_from_another_customer_never_replays_their_result(tx_db):
    rahul, rohan = make_ctx(tx_db), make_ctx(tx_db)
    await verify_as(rahul, "rahul.verma@example.com")
    await verify_as(rohan, "rohan.iyer@example.com")
    key = new_key()
    args = {"category": "other", "summary": "Question about invoices."}
    await ok(rahul, "create_ticket", idempotency_key=key, **args)
    error = await err(rohan, "create_ticket", idempotency_key=key, **args)
    assert error.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


async def test_ineligible_return_writes_nothing(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    key = new_key()
    error = await err(ctx, "create_return", idempotency_key=key, order_number="VX-4410",
                      sku="JNS-IND-32", reason="no_longer_needed")  # fmt: skip
    assert error.code is ErrorCode.NOT_ELIGIBLE
    assert error.details == {"reason": "window_expired"}
    assert await count(ctx, Return, Return.idempotency_key == key) == 0
    assert await count(ctx, ToolInvocation, ToolInvocation.idempotency_key == key) == 0


async def test_failed_call_does_not_burn_the_key(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    key = new_key()
    args = {"order_number": "VX-4411", "reason": "damaged"}
    error = await err(ctx, "create_return", idempotency_key=key, sku="NOPE-1", **args)
    assert error.code is ErrorCode.ITEM_NOT_FOUND
    # Nothing was stored, so the same key with corrected arguments goes through.
    assert (await call(ctx, "create_return", idempotency_key=key, sku="TSH-BLK-M", **args)).ok


async def test_second_return_for_same_item_is_refused(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    args = {"order_number": "VX-4411", "sku": "TSH-BLK-M", "reason": "damaged"}
    await ok(ctx, "create_return", **args)
    error = await err(ctx, "create_return", **args)  # new key: a genuinely new request
    assert error.details == {"reason": "already_returned"}


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
    key = new_key()
    args = {"order_number": "VX-4411", "reason": "return", "sku": "TSH-BLK-M"}
    first = await call(ctx, "issue_refund", idempotency_key=key, **args)
    assert first.ok, first.error
    assert first.data["amount_paise"] == 2 * 599_00
    assert first.data["amount_display"] == "₹1,198"
    assert first.data["status"] == "pending"
    replay = await call(ctx, "issue_refund", idempotency_key=key, **args)
    assert replay.replayed and replay.data == first.data
    assert await count(ctx, Refund, Refund.idempotency_key == key) == 1

    retry = await err(ctx, "issue_refund", **args)  # a new key cannot refund twice
    assert retry.details == {"reason": "already_refunded"}


async def test_refund_above_cap_needs_human_and_writes_nothing(ctx):
    await verify_as(ctx, "arjun.malhotra@example.com")
    ret = await ok(ctx, "create_return", order_number="VX-7302", sku="LAP-PRO-16",
                   reason="damaged")  # fmt: skip
    assert ret["refund_requires_human"] is True
    key = new_key()
    error = await err(ctx, "issue_refund", idempotency_key=key, order_number="VX-7302",
                      reason="return", sku="LAP-PRO-16")  # fmt: skip
    assert error.code is ErrorCode.REQUIRES_HUMAN
    assert error.details == {"reason": "above_auto_refund_cap"}
    assert "₹84,999" in error.message
    assert await count(ctx, Refund, Refund.idempotency_key == key) == 0


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
         ErrorCode.REQUIRES_HUMAN, "goodwill_needs_approval"),
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
    assert (error.code, error.details) == (code, {"reason": reason})
    assert await count(ctx, Refund) == before


async def test_whole_order_claim_with_a_carrier_delivered_part_needs_human(ctx):
    # VX-2201: keyboard and mouse delivered per carrier, lamp never shipped.
    await verify_as(ctx, "aarav.mehta@example.com")
    error = await err(ctx, "issue_refund", order_number="VX-2201", reason="missing_delivery")
    assert error.details == {"reason": "delivered_per_carrier"}


async def test_not_yet_overdue_message_gives_eta(ctx):
    await verify_as(ctx, "kavya.nair@example.com")
    error = await err(ctx, "issue_refund", order_number="VX-3302", reason="missing_delivery")
    assert error.message == (
        "Order VX-3302 is still on its way and isn't late yet. It's expected by 31 October 2026."
    )


async def test_missing_delivery_refund_for_overdue_parcel(ctx):
    await verify_as(ctx, "meera.pillai@example.com")
    data = await ok(ctx, "issue_refund", order_number="VX-8015", reason="missing_delivery")
    assert data["amount_paise"] == 3_499_00
    error = await err(ctx, "reship_order", order_number="VX-8015")
    assert error.details == {"reason": "already_resolved"}


# --- reship_order ---------------------------------------------------------------------------


async def test_reship_creates_free_replacement_once(ctx):
    customer_id = await verify_as(ctx, "meera.pillai@example.com")
    key = new_key()
    first = await call(ctx, "reship_order", idempotency_key=key, order_number="VX-8015")
    assert first.ok, first.error
    number = first.data["replacement_order_number"]
    assert first.data["item_names"] == ["Bluetooth speaker"]

    replacement = await ctx.db.scalar(select(Order).where(Order.order_number == number))
    original_id = await ctx.db.scalar(select(Order.id).where(Order.order_number == "VX-8015"))
    assert (replacement.customer_id, replacement.total_paise) == (customer_id, 0)
    assert replacement.replaces_order_id == original_id
    assert "VX-8015" not in spoken_confusables(number)

    replay = await call(ctx, "reship_order", idempotency_key=key, order_number="VX-8015")
    assert replay.replayed and replay.data["replacement_order_number"] == number
    again = await err(ctx, "reship_order", order_number="VX-8015")
    assert again.details == {"reason": "already_resolved"}
    refund = await err(ctx, "issue_refund", order_number="VX-8015", reason="missing_delivery")
    assert refund.details == {"reason": "already_resolved"}


async def test_reship_refused_when_carrier_says_delivered(ctx):
    await verify_as(ctx, "sameer.khan@example.com")
    error = await err(ctx, "reship_order", order_number="VX-3107")
    assert error.code is ErrorCode.REQUIRES_HUMAN
    assert "escalate_to_human" in error.hint


# --- create_ticket --------------------------------------------------------------------------


async def test_duplicate_open_ticket_is_refused(ctx):
    await verify_as(ctx, "sameer.khan@example.com")
    error = await err(ctx, "create_ticket", category="missing_delivery", order_number="VX-3107",
                      summary="Customer says the earbuds never arrived.")  # fmt: skip
    assert error.code is ErrorCode.TICKET_ALREADY_OPEN
    assert error.details["ticket_reference"].startswith("TCK-")


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
    assert handoff["verified"] is False
    assert handoff["customer_name"] is None
    assert handoff["order_number"] is None
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


async def test_escalation_links_own_order(ctx):
    await verify_as(ctx, "arjun.malhotra@example.com")
    await call(ctx, "issue_refund", order_number="VX-7302", reason="goodwill")
    data = await ok(ctx, "escalate_to_human", reason="policy_requires_review",
                    summary="Damaged laptop; refund above automatic limit.",
                    order_number="VX-7302", category="damaged_product")  # fmt: skip
    handoff = data["handoff"]
    assert (handoff["verified"], handoff["customer_name"]) == (True, "Arjun Malhotra")
    assert handoff["order_number"] == "VX-7302"
    assert [a["tool"] for a in handoff["actions_attempted"]] == ["verify_customer", "issue_refund"]
