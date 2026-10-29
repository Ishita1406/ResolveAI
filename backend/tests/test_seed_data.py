"""Invariants over the seeded test database, including every documented edge case."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import selectinload

from app.db.models import Customer, Order, Refund, Shipment
from app.db.seed import ANCHOR
from app.domain.enums import OrderStatus, RefundStatus, ShipmentStatus
from app.domain.order_ids import spoken_confusables
from app.domain.policy import AUTO_REFUND_CAP_PAISE, RETURN_WINDOW_DAYS

pytestmark = [pytest.mark.anyio, pytest.mark.db]


async def _all_orders(session) -> list[Order]:
    stmt = select(Order).options(
        selectinload(Order.items),
        selectinload(Order.shipments).selectinload(Shipment.events),
        selectinload(Order.refunds),
    )
    return list(await session.scalars(stmt))


async def test_row_counts(db_session):
    assert await db_session.scalar(select(func.count()).select_from(Customer)) == 50
    assert await db_session.scalar(select(func.count()).select_from(Order)) == 200


async def test_money_columns_are_integer_paise(db_session):
    rows = await db_session.execute(
        text(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND column_name LIKE '%paise'"
        )
    )
    assert {(r.table_name, r.column_name): r.data_type for r in rows} == {
        ("orders", "subtotal_paise"): "bigint",
        ("orders", "shipping_fee_paise"): "bigint",
        ("orders", "total_paise"): "bigint",
        ("order_items", "unit_price_paise"): "bigint",
        ("refunds", "amount_paise"): "bigint",
    }


async def test_order_amounts_add_up(db_session):
    for order in await _all_orders(db_session):
        assert order.subtotal_paise == sum(i.line_total_paise for i in order.items)
        assert order.total_paise == order.subtotal_paise + order.shipping_fee_paise


async def test_refunds_never_exceed_order_total(db_session):
    for order in await _all_orders(db_session):
        committed = sum(
            r.amount_paise
            for r in order.refunds
            if r.status in (RefundStatus.SUCCEEDED, RefundStatus.PENDING)
        )
        assert committed <= order.total_paise, order.order_number


async def test_order_status_matches_shipments(db_session):
    for order in await _all_orders(db_session):
        shipped = [i for i in order.items if i.shipment_id is not None]
        delivered_ids = {s.id for s in order.shipments if s.status == ShipmentStatus.DELIVERED}
        n = order.order_number
        match order.status:
            case OrderStatus.PLACED | OrderStatus.CANCELLED:
                assert not order.shipments, n
            case OrderStatus.PARTIALLY_SHIPPED:
                assert 0 < len(shipped) < len(order.items), n
            case OrderStatus.SHIPPED:
                assert len(shipped) == len(order.items), n
                assert not all(i.shipment_id in delivered_ids for i in order.items), n
            case OrderStatus.DELIVERED:
                assert all(i.shipment_id in delivered_ids for i in order.items), n


async def test_shipment_events_are_chronological_and_not_in_future(db_session):
    for order in await _all_orders(db_session):
        for shipment in order.shipments:
            times = [e.occurred_at for e in shipment.events]
            assert times == sorted(times), order.order_number
            assert times[-1] <= ANCHOR, order.order_number


async def test_only_designed_sound_alike_pairs_exist(db_session):
    numbers = set(await db_session.scalars(select(Order.order_number)))
    pairs = {tuple(sorted((a, b))) for a in numbers for b in spoken_confusables(a) if b in numbers}
    assert pairs == {("VX-1315", "VX-1350"), ("VX-1508", "VX-1580")}


async def _delivered_at(session, number):
    stmt = select(Shipment.delivered_at).join(Order).where(Order.order_number == number)
    return await session.scalar(stmt)


async def test_return_window_boundary_cases(db_session):
    window = timedelta(days=RETURN_WINDOW_DAYS)
    expired = await _delivered_at(db_session, "VX-4410")
    eligible = await _delivered_at(db_session, "VX-4411")
    assert expired + window == ANCHOR - timedelta(days=1)
    assert eligible + window == ANCHOR + timedelta(days=1)


async def test_high_value_order_exceeds_cap_inside_window(db_session):
    order = await db_session.scalar(select(Order).where(Order.order_number == "VX-7302"))
    assert order.total_paise > AUTO_REFUND_CAP_PAISE
    delivered_at = await _delivered_at(db_session, "VX-7302")
    assert ANCHOR - delivered_at < timedelta(days=RETURN_WINDOW_DAYS)


async def test_refund_idempotency_keys_unique(db_session):
    keys = list(await db_session.scalars(select(Refund.idempotency_key)))
    assert len(keys) == len(set(keys))
