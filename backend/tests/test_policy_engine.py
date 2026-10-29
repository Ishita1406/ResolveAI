"""Every PolicyEngine rule, with boundaries, on in-memory data (no database)."""

import json
from datetime import timedelta

import pytest

from app.domain.enums import (
    OrderStatus,
    RefundReason,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    ShipmentStatus,
)
from app.policy.config import Issue, Remedy, get_policy
from app.policy.engine import PolicyEngine, Reason, Verdict
from tests.policy_helpers import (
    NOW,
    TICK,
    add_refund,
    add_return,
    decide,
    engine,
    make_order,
    policy_dict,
)

POLICY = get_policy()
ENGINE = PolicyEngine(POLICY)
WINDOW = timedelta(days=POLICY.returns.window_days)
GRACE = timedelta(days=POLICY.missing_delivery.grace_days_after_eta)
ORDER_CAP = POLICY.refunds.per_order_cap_paise
CUSTOMER_CAP = POLICY.refunds.per_customer_cap_paise
RESHIP_CAP = POLICY.reship.max_value_paise


def outcome(decision):
    return decision.verdict, decision.reason, decision.rule


def test_shipped_defaults_match_the_brief():
    assert POLICY.returns.window_days == 30
    assert ORDER_CAP == 10_000_00
    assert CUSTOMER_CAP == 15_000_00


# --- returns.window_days --------------------------------------------------------------------


def test_return_exactly_at_window_end_is_allowed():
    order = make_order(delivered_at=NOW - WINDOW)
    d = decide(ENGINE, "create_return", order, sku="SKU-0")
    assert outcome(d) == (Verdict.APPROVE, None, "returns.window_days")
    assert d.amount_paise == 1_000_00


def test_return_one_microsecond_after_window_is_refused():
    order = make_order(delivered_at=NOW - WINDOW - TICK)
    d = decide(ENGINE, "create_return", order, sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, Reason.WINDOW_EXPIRED, "returns.window_days")
    assert "closed on" in d.message


def test_return_window_follows_config():
    eng = engine(returns={"window_days": 7})
    assert decide(eng, "create_return", make_order(delivered_at=NOW - timedelta(days=7)),
                  sku="SKU-0").approved  # fmt: skip
    assert not decide(eng, "create_return", make_order(delivered_at=NOW - timedelta(days=8)),
                      sku="SKU-0").approved  # fmt: skip


@pytest.mark.parametrize(
    ("order_kwargs", "reason", "rule"),
    [
        ({"status": OrderStatus.CANCELLED, "shipment_status": None}, Reason.ORDER_CANCELLED,
         "order.status"),
        ({"status": OrderStatus.PLACED, "shipment_status": None}, Reason.NOT_SHIPPED,
         "item.fulfilment"),
        ({"status": OrderStatus.SHIPPED, "shipment_status": ShipmentStatus.IN_TRANSIT},
         Reason.NOT_DELIVERED, "item.fulfilment"),
    ],
)  # fmt: skip
def test_return_requires_a_delivered_live_order(order_kwargs, reason, rule):
    d = decide(ENGINE, "create_return", make_order(**order_kwargs), sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, reason, rule)


def test_one_return_per_item_unless_rejected():
    order = make_order()
    ret = add_return(order)
    d = decide(ENGINE, "create_return", order, sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, Reason.ALREADY_RETURNED, "returns.one_per_item")
    ret.status = ReturnStatus.REJECTED
    assert decide(ENGINE, "create_return", order, sku="SKU-0").approved


def test_return_refused_once_order_fully_refunded():
    order = make_order()
    add_refund(order, order.total_paise, reason=RefundReason.MISSING_DELIVERY)
    d = decide(ENGINE, "create_return", order, sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, Reason.ALREADY_REFUNDED, "refunds.balance")


# --- refunds.per_order_cap_inr --------------------------------------------------------------


def _lost(total: int):
    return make_order(prices=(total,), status=OrderStatus.SHIPPED,
                      shipment_status=ShipmentStatus.LOST)  # fmt: skip


def test_refund_exactly_at_order_cap_is_approved():
    d = decide(ENGINE, "issue_refund", _lost(ORDER_CAP), reason="missing_delivery")
    assert d.approved and d.amount_paise == ORDER_CAP


def test_refund_one_paisa_over_order_cap_needs_human():
    d = decide(ENGINE, "issue_refund", _lost(ORDER_CAP + 1), reason="missing_delivery")
    assert outcome(d) == (
        Verdict.REQUIRE_HUMAN, Reason.OVER_ORDER_REFUND_CAP, "refunds.per_order_cap_inr"
    )  # fmt: skip
    assert d.facts["cap_paise"] == ORDER_CAP


def test_order_cap_counts_earlier_refunds():
    order = make_order(prices=(6_000_00, 5_000_00))
    for i in (0, 1):
        add_return(order, i, rid=500 + i)
    add_refund(order, 6_000_00, return_id=500, status=RefundStatus.PENDING)
    # 6,000 already committed + 4,000 would be exactly the cap; 5,000 is over it.
    d = decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-1")
    assert d.reason is Reason.OVER_ORDER_REFUND_CAP
    assert d.facts["committed_paise"] == 6_000_00
    order.items[1].unit_price_paise = 4_000_00
    order.total_paise = order.subtotal_paise = 10_000_00
    assert decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-1").approved


# --- refunds.per_customer_cap_inr -----------------------------------------------------------


def test_refund_exactly_at_customer_cap_is_approved():
    d = decide(ENGINE, "issue_refund", _lost(1_000_00), reason="missing_delivery",
               customer_refunds_in_window_paise=CUSTOMER_CAP - 1_000_00)  # fmt: skip
    assert d.approved


def test_refund_one_paisa_over_customer_cap_needs_human():
    d = decide(ENGINE, "issue_refund", _lost(1_000_00), reason="missing_delivery",
               customer_refunds_in_window_paise=CUSTOMER_CAP - 1_000_00 + 1)  # fmt: skip
    assert outcome(d) == (
        Verdict.REQUIRE_HUMAN, Reason.OVER_CUSTOMER_REFUND_CAP, "refunds.per_customer_cap_inr"
    )  # fmt: skip


def test_order_cap_is_checked_before_customer_cap():
    d = decide(ENGINE, "issue_refund", _lost(ORDER_CAP + 1), reason="missing_delivery",
               customer_refunds_in_window_paise=CUSTOMER_CAP)  # fmt: skip
    assert d.reason is Reason.OVER_ORDER_REFUND_CAP


# --- refunds: other rules -------------------------------------------------------------------


def test_refund_for_return_needs_an_unrefunded_return():
    order = make_order()
    d = decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, Reason.NO_RETURN_ON_FILE, "refunds.requires_return")
    add_return(order)
    assert decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-0").approved
    add_refund(order, 1_00, return_id=500)
    d = decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-0")
    assert d.reason is Reason.ALREADY_REFUNDED


def test_refund_amount_is_capped_at_remaining_balance():
    order = make_order(prices=(1_000_00,))
    add_return(order)
    add_refund(order, 300_00, reason=RefundReason.GOODWILL)
    d = decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-0")
    assert d.amount_paise == 700_00


def test_cancelled_and_fully_refunded_orders():
    cancelled = make_order(status=OrderStatus.CANCELLED, shipment_status=None)
    assert decide(ENGINE, "issue_refund", cancelled, reason="goodwill").reason is (
        Reason.ORDER_CANCELLED
    )
    refunded = make_order()
    add_refund(refunded, refunded.total_paise)
    assert decide(ENGINE, "issue_refund", refunded, reason="goodwill").reason is (
        Reason.ALREADY_REFUNDED
    )


# --- missing_delivery.grace_days_after_eta --------------------------------------------------


def _in_transit(eta):
    return make_order(status=OrderStatus.SHIPPED, shipment_status=ShipmentStatus.IN_TRANSIT,
                      eta=eta)  # fmt: skip


def test_parcel_exactly_at_grace_end_is_not_yet_missing():
    d = decide(ENGINE, "issue_refund", _in_transit(NOW - GRACE), reason="missing_delivery")
    assert outcome(d) == (
        Verdict.REJECT, Reason.NOT_YET_OVERDUE, "missing_delivery.grace_days_after_eta"
    )  # fmt: skip
    assert "expected by" in d.message


def test_parcel_one_microsecond_past_grace_is_missing():
    d = decide(ENGINE, "issue_refund", _in_transit(NOW - GRACE - TICK), reason="missing_delivery")
    assert d.approved


def test_parcel_without_eta_is_never_presumed_missing():
    d = decide(ENGINE, "issue_refund", _in_transit(None), reason="missing_delivery")
    assert d.reason is Reason.NOT_YET_OVERDUE


def test_one_missing_delivery_remedy_per_order():
    order = _lost(1_000_00)
    assert decide(ENGINE, "reship_order", order, issue="missing_delivery",
                  has_replacement=True).reason is Reason.ALREADY_RESOLVED  # fmt: skip
    add_refund(order, 1_00, reason=RefundReason.MISSING_DELIVERY, status=RefundStatus.PENDING)
    assert decide(ENGINE, "reship_order", order, issue="missing_delivery").reason is (
        Reason.ALREADY_RESOLVED
    )


def test_human_review_outranks_refusal_across_shipments():
    order = make_order(prices=(1_00, 2_00))
    order.items[1].shipment_id = None  # second item never shipped; first was delivered
    d = decide(ENGINE, "issue_refund", order, reason="missing_delivery")
    assert d.reason is Reason.DELIVERED_PER_CARRIER


# --- escalate_when.* ------------------------------------------------------------------------

ESCALATION_CASES = {
    "carrier_marked_delivered": (
        lambda: make_order(), "issue_refund", {"reason": "missing_delivery"},
        Reason.DELIVERED_PER_CARRIER),
    "returned_to_sender": (
        lambda: make_order(status=OrderStatus.SHIPPED,
                           shipment_status=ShipmentStatus.RETURNED_TO_SENDER),
        "issue_refund", {"reason": "missing_delivery"}, Reason.RETURNED_TO_SENDER),
    "goodwill_refund_requested": (
        lambda: make_order(), "issue_refund", {"reason": "goodwill"}, Reason.GOODWILL_REQUESTED),
    "over_order_refund_cap": (
        lambda: _lost(ORDER_CAP + 1), "issue_refund", {"reason": "missing_delivery"},
        Reason.OVER_ORDER_REFUND_CAP),
    "over_customer_refund_cap": (
        lambda: _lost(1_00), "issue_refund",
        {"reason": "missing_delivery", "customer_refunds_in_window_paise": CUSTOMER_CAP},
        Reason.OVER_CUSTOMER_REFUND_CAP),
    "over_reship_value_cap": (
        lambda: _lost(RESHIP_CAP + 1), "reship_order", {"issue": "missing_delivery"},
        Reason.OVER_RESHIP_VALUE_CAP),
}  # fmt: skip


def test_every_escalation_flag_has_a_case():
    assert set(ESCALATION_CASES) == set(policy_dict()["escalate_when"])


@pytest.mark.parametrize("flag", sorted(ESCALATION_CASES))
@pytest.mark.parametrize("escalate", [True, False])
def test_escalation_flags_choose_human_or_refusal(flag, escalate):
    make, tool, kwargs, reason = ESCALATION_CASES[flag]
    eng = engine(escalate_when={flag: escalate})
    d = decide(eng, tool, make(), **kwargs)
    assert d.reason is reason
    assert d.rule.split(".")[-1] in {flag, "per_order_cap_inr", "per_customer_cap_inr",
                                     "max_value_inr"}  # fmt: skip
    assert d.verdict is (Verdict.REQUIRE_HUMAN if escalate else Verdict.REJECT)
    assert d.message  # speakable either way


def test_human_and_refusal_messages_differ():
    order = make_order()
    human = decide(engine(escalate_when={"goodwill_refund_requested": True}), "issue_refund",
                   order, reason="goodwill")  # fmt: skip
    refuse = decide(engine(escalate_when={"goodwill_refund_requested": False}), "issue_refund",
                    order, reason="goodwill")  # fmt: skip
    assert "specialist" in human.message
    assert human.message != refuse.message


# --- reship.max_value_inr and remedies ------------------------------------------------------


def test_reship_exactly_at_value_cap_is_approved():
    d = decide(ENGINE, "reship_order", _lost(RESHIP_CAP), issue="missing_delivery")
    assert outcome(d) == (Verdict.APPROVE, None, "reship.max_value_inr")
    assert d.amount_paise == RESHIP_CAP


def test_reship_one_paisa_over_value_cap_needs_human():
    d = decide(ENGINE, "reship_order", _lost(RESHIP_CAP + 1), issue="missing_delivery")
    assert outcome(d) == (
        Verdict.REQUIRE_HUMAN, Reason.OVER_RESHIP_VALUE_CAP, "reship.max_value_inr"
    )  # fmt: skip


@pytest.mark.parametrize("issue", list(Issue))
def test_reship_allowed_exactly_when_configured(issue):
    order = _lost(1_00) if issue is Issue.MISSING_DELIVERY else make_order()
    d = decide(ENGINE, "reship_order", order, issue=issue.value,
               sku=None if issue is Issue.MISSING_DELIVERY else "SKU-0")  # fmt: skip
    allowed = POLICY.allows(issue, Remedy.RESHIP)
    assert d.approved is allowed
    if not allowed:
        assert outcome(d) == (Verdict.REJECT, Reason.REMEDY_NOT_ALLOWED, f"remedies.{issue}")


@pytest.mark.parametrize("issue", [i for i in Issue if i is not Issue.MISSING_DELIVERY])
def test_refund_for_returned_item_follows_remedies(issue):
    order = make_order()
    add_return(order, reason=ReturnReason(issue.value))
    allowed = decide(ENGINE, "issue_refund", order, reason="return", sku="SKU-0")
    assert allowed.approved is POLICY.allows(issue, Remedy.REFUND)
    no_refunds = engine(remedies={**policy_dict()["remedies"], issue.value: ["reship"]})
    d = decide(no_refunds, "issue_refund", order, reason="return", sku="SKU-0")
    assert outcome(d) == (Verdict.REJECT, Reason.REMEDY_NOT_ALLOWED, f"remedies.{issue}")


def test_missing_delivery_refund_can_be_disabled():
    eng = engine(remedies={**policy_dict()["remedies"], "missing_delivery": ["reship"]})
    d = decide(eng, "issue_refund", _lost(1_00), reason="missing_delivery")
    assert d.rule == "remedies.missing_delivery"
    assert d.message == "For a missing delivery, I can offer a free replacement, but not a refund."


def test_item_reship_respects_window_and_prior_refund():
    late = make_order(delivered_at=NOW - WINDOW - TICK)
    d = decide(ENGINE, "reship_order", late, issue="damaged", sku="SKU-0")
    assert d.reason is Reason.WINDOW_EXPIRED
    edge = make_order(delivered_at=NOW - WINDOW)
    assert decide(ENGINE, "reship_order", edge, issue="damaged", sku="SKU-0").approved
    refunded = make_order()
    add_return(refunded)
    add_refund(refunded, 1_00, return_id=500)
    d = decide(ENGINE, "reship_order", refunded, issue="wrong_item", sku="SKU-0")
    assert d.reason is Reason.ALREADY_REFUNDED


# --- tickets and escalation -----------------------------------------------------------------


def test_duplicate_ticket_rule():
    order = make_order()
    d = decide(ENGINE, "create_ticket", order, category="missing_delivery",
               open_ticket_reference="TCK-000001")  # fmt: skip
    assert outcome(d) == (Verdict.REJECT, Reason.DUPLICATE_TICKET, "tickets.no_duplicates")
    assert "TCK-000001" in d.message
    assert decide(ENGINE, "create_ticket", order, category="other").approved


def test_escalation_is_always_allowed():
    d = decide(ENGINE, "escalate_to_human", None)
    assert outcome(d) == (Verdict.APPROVE, None, "escalation.always_allowed")


# --- decision properties --------------------------------------------------------------------


def test_decisions_are_deterministic_and_json_safe():
    order = make_order(delivered_at=NOW - WINDOW - TICK)
    a = decide(ENGINE, "create_return", order, sku="SKU-0")
    b = decide(ENGINE, "create_return", order, sku="SKU-0")
    assert a == b
    json.dumps(a.facts)
    assert a.facts["deadline"] == (NOW - TICK).isoformat()


def test_unknown_tool_has_no_policy():
    with pytest.raises(ValueError, match="no policy"):
        decide(ENGINE, "delete_everything", make_order())
