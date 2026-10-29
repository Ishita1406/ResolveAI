from dataclasses import replace

import pytest
from sqlalchemy import func, select

from app.db.models import ToolInvocation
from app.repositories.customers import get_customer_by_email
from app.tools.errors import ErrorCode
from app.tools.registry import ALL_TOOLS, call_tool
from tests.tool_helpers import new_key, verify_as

pytestmark = [pytest.mark.anyio, pytest.mark.db]


async def ok(ctx, tool, args=None):
    result = await call_tool(ctx, tool, args or {})
    assert result.ok, result.error
    return result.data


async def err(ctx, tool, args):
    result = await call_tool(ctx, tool, args)
    assert not result.ok, result.data
    return result.error


# --- verification gate ----------------------------------------------------------------------

GATED_CALLS = {
    "list_recent_orders": {},
    "get_order": {"order_number": "VX-1508"},
    "get_shipment_status": {"order_number": "VX-1508"},
    "check_return_eligibility": {"order_number": "VX-1508"},
    "find_order_candidates": {"heard": "fifteen oh eight"},
    "create_return": {"order_number": "VX-1508", "sku": "PWB-10K", "reason": "damaged"},
    "issue_refund": {"order_number": "VX-8015", "reason": "missing_delivery"},
    "reship_order": {"order_number": "VX-8015"},
    "create_ticket": {"category": "other", "summary": "Customer has a question."},
}


def test_gated_calls_cover_every_gated_tool():
    assert set(GATED_CALLS) == {n for n, s in ALL_TOOLS.items() if s.requires_verification}


@pytest.mark.parametrize("tool", sorted(GATED_CALLS))
async def test_tools_refuse_before_verification(ctx, tool):
    args = dict(GATED_CALLS[tool])
    if ALL_TOOLS[tool].kind == "write":
        args["idempotency_key"] = new_key()
    error = await err(ctx, tool, args)
    assert error.code is ErrorCode.NOT_VERIFIED
    assert error.message == "Before I can help with that, I need to verify your account."
    assert await ctx.db.scalar(select(func.count()).select_from(ToolInvocation)) == 0


# --- verify_customer ------------------------------------------------------------------------


async def _postal(ctx, email):
    customer = await get_customer_by_email(ctx.db, email)
    return customer.postal_code, customer.phone


async def test_verify_by_spoken_phone_format(ctx):
    postal, phone = await _postal(ctx, "aarav.mehta@example.com")
    local = phone.removeprefix("+91")
    data = await ok(
        ctx, "verify_customer", {"phone": f"{local[:5]} {local[5:]}", "postal_code": postal}
    )
    assert data == {"verified": True, "first_name": "Aarav"}
    assert ctx.state.is_verified


async def test_wrong_details_and_unknown_account_look_identical(ctx):
    postal, _ = await _postal(ctx, "aarav.mehta@example.com")
    wrong_pin = "111111" if postal != "111111" else "222222"
    a = await err(
        ctx, "verify_customer", {"email": "aarav.mehta@example.com", "postal_code": wrong_pin}
    )
    b = await err(ctx, "verify_customer", {"email": "ghost@example.com", "postal_code": postal})
    assert a.code is b.code is ErrorCode.VERIFICATION_FAILED
    assert a.message == b.message
    assert (a.details, b.details) == ({"attempts_remaining": 2}, {"attempts_remaining": 1})
    assert not ctx.state.is_verified


async def test_lockout_after_three_failures_even_with_correct_details(ctx):
    postal, _ = await _postal(ctx, "neha.gupta@example.com")
    bad = {"email": "neha.gupta@example.com", "postal_code": "000000"}
    codes = [(await err(ctx, "verify_customer", bad)).code for _ in range(3)]
    assert codes == [
        ErrorCode.VERIFICATION_FAILED,
        ErrorCode.VERIFICATION_FAILED,
        ErrorCode.VERIFICATION_LOCKED,
    ]
    good = {"email": "neha.gupta@example.com", "postal_code": postal}
    assert (await err(ctx, "verify_customer", good)).code is ErrorCode.VERIFICATION_LOCKED
    assert not ctx.state.is_verified


async def test_cannot_switch_account_mid_call(ctx):
    rahul_id = await verify_as(ctx, "rahul.verma@example.com")
    assert await verify_as(ctx, "rahul.verma@example.com") == rahul_id  # same account: fine
    postal, _ = await _postal(ctx, "rohan.iyer@example.com")
    error = await err(
        ctx, "verify_customer", {"email": "rohan.iyer@example.com", "postal_code": postal}
    )
    assert error.code is ErrorCode.ALREADY_VERIFIED
    assert ctx.state.customer_id == rahul_id


# --- list_recent_orders / get_order ---------------------------------------------------------


async def test_list_recent_orders_newest_first_with_limit(ctx):
    await verify_as(ctx, "kavya.nair@example.com")
    data = await ok(ctx, "list_recent_orders", {"limit": 3})
    assert [o["order_number"] for o in data["orders"]] == ["VX-3301", "VX-3302", "VX-3303"]
    data = await ok(ctx, "list_recent_orders")
    assert len(data["orders"]) == 4
    assert data["orders"][0]["total_display"] == "₹899"


async def test_get_order_amounts_returns_and_refunds(ctx):
    await verify_as(ctx, "vikram.singh@example.com")
    data = await ok(ctx, "get_order", {"order_number": "vx-5121"})
    assert data["refunded_paise"] == 1_299_00
    assert data["refundable_balance_paise"] == data["total_paise"] - 1_299_00
    assert [(r["sku"], r["status"]) for r in data["returns"]] == [("KTL-ELC-15", "completed")]
    assert data["refunds"][0]["reference"].startswith("RF-")


async def test_get_order_shows_unshipped_item(ctx):
    await verify_as(ctx, "aarav.mehta@example.com")
    data = await ok(ctx, "get_order", {"order_number": "VX-2201"})
    assert {i["sku"]: i["fulfilment"] for i in data["items"]} == {
        "KBD-MEC-01": "delivered",
        "MSE-WL-01": "delivered",
        "LMP-DSK-LED": "not_shipped",
    }


async def test_other_customers_order_is_not_found_and_suggests_own(ctx):
    await verify_as(ctx, "rohan.iyer@example.com")
    error = await err(ctx, "get_order", {"order_number": "VX-1350"})  # Priya's order
    assert error.code is ErrorCode.ORDER_NOT_FOUND
    assert error.details == {"candidates": ["VX-1315"]}
    assert "Did you mean VX-1315?" in error.message
    assert "Smartwatch" not in error.model_dump_json()


# --- get_shipment_status --------------------------------------------------------------------


async def test_split_shipments(ctx):
    await verify_as(ctx, "aarav.mehta@example.com")
    data = await ok(ctx, "get_shipment_status", {"order_number": "VX-2202"})
    assert [s["status"] for s in data["shipments"]] == ["delivered", "in_transit"]
    assert data["unshipped_item_names"] == []
    data = await ok(ctx, "get_shipment_status", {"order_number": "VX-2201"})
    assert data["unshipped_item_names"] == ["LED desk lamp"]


async def test_overdue_flag(ctx):
    await verify_as(ctx, "meera.pillai@example.com")
    data = await ok(ctx, "get_shipment_status", {"order_number": "VX-8015"})
    (shipment,) = data["shipments"]
    assert shipment["is_overdue"] is True
    assert shipment["latest_event"]["code"] == "in_transit"


async def test_not_overdue_and_delivery_note(ctx):
    await verify_as(ctx, "kavya.nair@example.com")
    data = await ok(ctx, "get_shipment_status", {"order_number": "VX-3302"})
    assert data["shipments"][0]["is_overdue"] is False


# --- check_return_eligibility ---------------------------------------------------------------


async def _eligibility(ctx, email, order_number, sku=None):
    await verify_as(ctx, email)
    args = {"order_number": order_number} | ({"sku": sku} if sku else {})
    return {i["sku"]: i for i in (await ok(ctx, "check_return_eligibility", args))["items"]}


@pytest.mark.parametrize(
    ("email", "order", "sku", "outcome", "reason"),
    [
        ("neha.gupta@example.com", "VX-4410", "JNS-IND-32", "not_eligible", "window_expired"),
        ("neha.gupta@example.com", "VX-4411", "TSH-BLK-M", "eligible", None),
        ("vikram.singh@example.com", "VX-5120", "SPK-BT-10", "not_eligible", "already_returned"),
        ("ananya.rao@example.com", "VX-6031", "SAR-SLK-01", "not_eligible", "order_cancelled"),
        ("aarav.mehta@example.com", "VX-2201", "LMP-DSK-LED", "not_eligible", "not_shipped"),
        ("aarav.mehta@example.com", "VX-2202", "CHR-USBC-65", "not_eligible", "not_delivered"),
        ("arjun.malhotra@example.com", "VX-7302", "LAP-PRO-16", "eligible", None),
    ],
)
async def test_return_eligibility_edge_cases(ctx, email, order, sku, outcome, reason):
    item = (await _eligibility(ctx, email, order, sku))[sku]
    assert (item["outcome"], item["reason"]) == (outcome, reason)


async def test_expired_window_explains_date(ctx):
    item = (await _eligibility(ctx, "neha.gupta@example.com", "VX-4410"))["JNS-IND-32"]
    assert item["explanation"] == "The return window for order VX-4410 closed on 28 October 2026."


async def test_high_value_item_is_returnable_but_refund_needs_human(ctx):
    item = (await _eligibility(ctx, "arjun.malhotra@example.com", "VX-7302"))["LAP-PRO-16"]
    assert item["refund_requires_human"] is True
    assert item["refund_amount_display"] == "₹84,999"


async def test_unknown_sku(ctx):
    await verify_as(ctx, "neha.gupta@example.com")
    error = await err(ctx, "check_return_eligibility", {"order_number": "VX-4411", "sku": "NOPE-1"})
    assert error.code is ErrorCode.ITEM_NOT_FOUND
    assert error.details == {"skus": ["TSH-BLK-M"]}


# --- find_order_candidates ------------------------------------------------------------------


async def test_candidates_for_sound_alike_pair(ctx):
    await verify_as(ctx, "rahul.verma@example.com")
    data = await ok(ctx, "find_order_candidates", {"heard": "fifteen eighty"})
    assert data["heard_digits"] == "1580"
    assert data["exact_match"] is True
    assert [(c["order_number"], c["match"]) for c in data["candidates"]] == [
        ("VX-1580", "exact"),
        ("VX-1508", "sounds_alike"),
    ]
    assert data["candidates"][1]["item_names"] == ["Power bank 10,000 mAh"]


async def test_candidates_never_include_other_customers_orders(ctx):
    await verify_as(ctx, "rohan.iyer@example.com")
    data = await ok(ctx, "find_order_candidates", {"heard": "thirteen fifty"})
    assert [c["order_number"] for c in data["candidates"]] == ["VX-1315"]
    assert data["exact_match"] is False


async def test_no_candidates_for_gibberish(ctx):
    await verify_as(ctx, "rahul.verma@example.com")
    data = await ok(ctx, "find_order_candidates", {"heard": "the blue jumper"})
    assert data == {"heard_digits": None, "exact_match": False, "candidates": []}
    # A stray number word yields too few digits to match anything.
    data = await ok(ctx, "find_order_candidates", {"heard": "the blue one"})
    assert (data["heard_digits"], data["candidates"]) == ("1", [])


# --- internal errors ------------------------------------------------------------------------


async def test_unexpected_exception_becomes_generic_retryable_error(ctx, monkeypatch, caplog):
    async def boom(_ctx, _args):
        raise RuntimeError("db password is hunter2")

    monkeypatch.setitem(ALL_TOOLS, "get_order", replace(ALL_TOOLS["get_order"], handler=boom))
    await verify_as(ctx, "rahul.verma@example.com")
    error = await err(ctx, "get_order", {"order_number": "VX-1508"})
    assert error.code is ErrorCode.INTERNAL_ERROR
    assert error.retryable is True
    assert "hunter2" not in error.model_dump_json()
    assert "hunter2" in caplog.text  # full detail stays in the server log
