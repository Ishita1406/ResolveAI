import pytest
from sqlalchemy import select
from sqlalchemy.exc import InvalidRequestError

from app.db.models import Conversation
from app.db.seed import ANCHOR
from app.domain.enums import (
    OrderStatus,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    ShipmentEventCode,
    ShipmentStatus,
    TicketCategory,
    TurnRole,
)
from app.repositories import customers, orders, refunds, support

pytestmark = [pytest.mark.anyio, pytest.mark.db]


async def _customer(session, email):
    customer = await customers.get_customer_by_email(session, email)
    assert customer is not None
    return customer


# --- customers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("98765 43210", "+919876543210"),
        ("+91-98765-43210", "+919876543210"),
        ("09876543210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("12345", None),
        ("5876543210", None),
    ],
)
def test_normalize_phone(raw, expected):
    assert customers.normalize_phone(raw) == expected


async def test_get_customer_by_email_is_case_and_space_insensitive(db_session):
    customer = await customers.get_customer_by_email(db_session, "  Aarav.Mehta@EXAMPLE.com ")
    assert customer is not None
    assert customer.full_name == "Aarav Mehta"
    assert await customers.get_customer(db_session, customer.id) is customer


async def test_get_customer_by_phone_accepts_spoken_formats(db_session):
    aarav = await _customer(db_session, "aarav.mehta@example.com")
    local = aarav.phone.removeprefix("+91")
    for variant in (aarav.phone, f"0{local}", f"{local[:5]} {local[5:]}"):
        assert await customers.get_customer_by_phone(db_session, variant) is aarav
    assert await customers.get_customer_by_phone(db_session, "not a phone") is None


async def test_duplicate_names_return_every_match(db_session):
    matches = await customers.find_customers_by_name(db_session, "  priya   SHARMA ")
    assert {c.email for c in matches} == {
        "priya.sharma@example.com",
        "priya.sharma2@example.com",
    }


async def test_unknown_customer_is_none(db_session):
    assert await customers.get_customer_by_email(db_session, "nobody@example.com") is None
    assert await customers.get_customer(db_session, 10_000_000) is None


# --- orders ------------------------------------------------------------------------------


async def test_partial_shipment_with_unshipped_item(db_session):
    order = await orders.get_order_by_number(db_session, " vx-2201 ")
    assert order is not None
    assert order.status == OrderStatus.PARTIALLY_SHIPPED
    assert len(order.items) == 3
    assert len(order.shipments) == 1
    unshipped = [i for i in order.items if i.shipment_id is None]
    assert [i.sku for i in unshipped] == ["LMP-DSK-LED"]
    assert len(order.shipments[0].items) == 2


async def test_split_shipments_one_delivered_one_in_transit(db_session):
    order = await orders.get_order_by_number(db_session, "VX-2202")
    assert order.status == OrderStatus.SHIPPED
    assert [s.status for s in order.shipments] == [
        ShipmentStatus.DELIVERED,
        ShipmentStatus.IN_TRANSIT,
    ]


async def test_delivered_but_reported_missing_has_delivery_scan(db_session):
    order = await orders.get_order_by_number(db_session, "VX-3107")
    events = order.shipments[0].events
    assert events[-1].code == ShipmentEventCode.DELIVERED
    assert "front door" in events[-1].description
    assert [e.occurred_at for e in events] == sorted(e.occurred_at for e in events)


async def test_order_lookup_is_scoped_to_owner(db_session):
    rohan = await _customer(db_session, "rohan.iyer@example.com")
    assert (await orders.get_order_for_customer(db_session, rohan.id, "VX-1315")) is not None
    # VX-1350 sounds like VX-1315 but belongs to priya.sharma@example.com.
    assert await orders.get_order_for_customer(db_session, rohan.id, "VX-1350") is None
    assert await orders.get_order_by_number(db_session, "VX-1350") is not None


async def test_unknown_order_number_is_none(db_session):
    assert await orders.get_order_by_number(db_session, "VX-0000") is None


async def test_multiple_active_orders_newest_first(db_session):
    kavya = await _customer(db_session, "kavya.nair@example.com")
    active = await orders.list_active_orders_for_customer(db_session, kavya.id)
    assert [o.order_number for o in active] == ["VX-3301", "VX-3302", "VX-3303"]
    everything = await orders.list_orders_for_customer(db_session, kavya.id)
    assert [o.order_number for o in everything][-1] == "VX-3190"


async def test_list_orders_filters_by_status(db_session):
    ananya = await _customer(db_session, "ananya.rao@example.com")
    cancelled = await orders.list_orders_for_customer(
        db_session, ananya.id, statuses=[OrderStatus.CANCELLED]
    )
    assert {o.order_number for o in cancelled} == {"VX-6031", "VX-6032"}
    assert await orders.list_active_orders_for_customer(db_session, ananya.id) == []


async def test_order_numbers_for_sound_alike_recovery(db_session):
    rahul = await _customer(db_session, "rahul.verma@example.com")
    assert await orders.list_order_numbers_for_customer(db_session, rahul.id) == [
        "VX-1508",
        "VX-1580",
    ]


async def test_order_lists_do_not_lazy_load(db_session):
    kavya = await _customer(db_session, "kavya.nair@example.com")
    order = (await orders.list_orders_for_customer(db_session, kavya.id))[0]
    assert order.items  # eager-loaded
    with pytest.raises(InvalidRequestError):
        _ = order.shipments  # not loaded by this query, and lazy="raise"


async def test_get_shipment_by_tracking_number(db_session):
    order = await orders.get_order_by_number(db_session, "VX-8015")
    tracking = order.shipments[0].tracking_number
    shipment = await orders.get_shipment_by_tracking_number(db_session, tracking.lower())
    assert shipment is not None
    assert shipment.status == ShipmentStatus.IN_TRANSIT
    # Stuck: ETA has passed and the latest scan is older than the ETA.
    assert shipment.estimated_delivery_at < ANCHOR
    assert shipment.events[-1].code == ShipmentEventCode.IN_TRANSIT
    assert shipment.events[-1].occurred_at < shipment.estimated_delivery_at
    assert await orders.get_shipment_by_tracking_number(db_session, "NOPE") is None


# --- returns & refunds ---------------------------------------------------------------------


async def test_fully_refunded_order(db_session):
    order = await orders.get_order_by_number(db_session, "VX-5120")
    totals = await refunds.get_refund_totals_paise(db_session, order.id)
    assert totals[RefundStatus.SUCCEEDED] == order.total_paise
    assert totals[RefundStatus.PENDING] == 0
    (ret,) = await refunds.list_returns_for_order(db_session, order.id)
    assert (ret.reason, ret.status) == (ReturnReason.DAMAGED, ReturnStatus.COMPLETED)


async def test_partially_refunded_order(db_session):
    order = await orders.get_order_by_number(db_session, "VX-5121")
    totals = await refunds.get_refund_totals_paise(db_session, order.id)
    assert totals[RefundStatus.SUCCEEDED] == 1_299_00
    assert totals[RefundStatus.SUCCEEDED] < order.total_paise


async def test_cancelled_order_with_pending_refund(db_session):
    order = await orders.get_order_by_number(db_session, "VX-6032")
    totals = await refunds.get_refund_totals_paise(db_session, order.id)
    assert totals == {
        RefundStatus.PENDING: order.total_paise,
        RefundStatus.SUCCEEDED: 0,
        RefundStatus.FAILED: 0,
    }


async def test_order_without_refunds_has_zero_totals(db_session):
    order = await orders.get_order_by_number(db_session, "VX-7302")
    assert await refunds.list_refunds_for_order(db_session, order.id) == []
    assert set((await refunds.get_refund_totals_paise(db_session, order.id)).values()) == {0}


async def test_refund_by_idempotency_key(db_session):
    order = await orders.get_order_by_number(db_session, "VX-6031")
    (refund,) = await refunds.list_refunds_for_order(db_session, order.id)
    found = await refunds.get_refund_by_idempotency_key(db_session, refund.idempotency_key)
    assert found is refund
    assert await refunds.get_refund_by_idempotency_key(db_session, "missing") is None


# --- tickets & conversations ---------------------------------------------------------------


async def test_open_missing_delivery_ticket_with_conversation(db_session):
    sameer = await _customer(db_session, "sameer.khan@example.com")
    (ticket,) = await support.list_tickets_for_customer(db_session, sameer.id, open_only=True)
    assert ticket.category == TicketCategory.MISSING_DELIVERY
    order = await orders.get_order_by_number(db_session, "VX-3107")
    assert ticket.order_id == order.id
    assert await support.list_tickets_for_order(db_session, order.id) == [ticket]
    assert await support.get_ticket(db_session, ticket.id) is ticket

    conversation = await support.get_conversation(db_session, ticket.conversation_id)
    assert conversation.customer_id == sameer.id
    assert conversation.verified_at is not None
    assert [t.turn_index for t in conversation.turns] == list(range(len(conversation.turns)))


async def test_unverified_conversation_has_no_customer(db_session):
    conv_id = await db_session.scalar(
        select(Conversation.id).where(Conversation.customer_id.is_(None))
    )
    conversation = await support.get_conversation(db_session, conv_id)
    assert conversation.verified_at is None
    assert conversation.turns[1].role == TurnRole.CUSTOMER
    assert "thirteen fifty" in conversation.turns[1].content
