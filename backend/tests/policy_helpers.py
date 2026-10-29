"""In-memory orders and policy overrides for pure PolicyEngine tests (no database)."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import yaml

from app.db.models import Order, OrderItem, Refund, Return, Shipment
from app.domain.enums import (
    OrderStatus,
    RefundReason,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    ShipmentStatus,
)
from app.policy.config import DEFAULT_POLICY_FILE, PolicyConfig
from app.policy.engine import PolicyEngine, PolicyFacts, ProposedAction

NOW = datetime(2026, 10, 29, 6, 30, tzinfo=UTC)
TICK = timedelta(microseconds=1)


def policy_dict() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_POLICY_FILE.read_text())


def policy_with(**overrides: dict[str, Any]) -> PolicyConfig:
    """Default policy with sections patched, e.g. policy_with(returns={"window_days": 7})."""
    raw = copy.deepcopy(policy_dict())
    for section, values in overrides.items():
        raw[section] = {**raw[section], **values} if isinstance(values, dict) else values
    return PolicyConfig.model_validate(raw)


def engine(**overrides: dict[str, Any]) -> PolicyEngine:
    return PolicyEngine(policy_with(**overrides))


def make_order(
    *,
    prices: tuple[int, ...] = (1_000_00,),
    status: OrderStatus = OrderStatus.DELIVERED,
    shipment_status: ShipmentStatus | None = ShipmentStatus.DELIVERED,
    delivered_at: datetime | None = NOW - timedelta(days=10),
    eta: datetime | None = None,
    number: str = "VX-1000",
) -> Order:
    """One order, one shipment holding every item (or none if shipment_status is None)."""
    shipment = None
    if shipment_status is not None:
        shipment = Shipment(
            id=10,
            status=shipment_status,
            delivered_at=delivered_at if shipment_status is ShipmentStatus.DELIVERED else None,
            estimated_delivery_at=eta,
        )
    items = [
        OrderItem(
            id=100 + i,
            sku=f"SKU-{i}",
            product_name=f"Item {i}",
            quantity=1,
            unit_price_paise=price,
            shipment_id=shipment.id if shipment else None,
        )
        for i, price in enumerate(prices)
    ]
    total = sum(prices)
    return Order(
        id=1,
        order_number=number,
        status=status,
        subtotal_paise=total,
        shipping_fee_paise=0,
        total_paise=total,
        placed_at=NOW - timedelta(days=15),
        items=items,
        shipments=[shipment] if shipment else [],
        returns=[],
        refunds=[],
    )


def add_return(order: Order, item_index: int = 0, *, reason=ReturnReason.DAMAGED, rid: int = 500):
    ret = Return(
        id=rid,
        order_item_id=order.items[item_index].id,
        quantity=1,
        reason=reason,
        status=ReturnStatus.REQUESTED,
        requested_at=NOW - timedelta(days=1),
    )
    order.returns.append(ret)
    return ret


def add_refund(
    order: Order,
    amount: int,
    *,
    reason=RefundReason.RETURN,
    status=RefundStatus.SUCCEEDED,
    return_id: int | None = None,
    rid: int = 900,
):
    refund = Refund(
        id=rid,
        amount_paise=amount,
        status=status,
        reason=reason,
        return_id=return_id,
        idempotency_key=f"k-{rid}",
        requested_at=NOW - timedelta(days=1),
    )
    order.refunds.append(refund)
    return refund


def decide(eng: PolicyEngine, tool: str, order: Order | None = None, **kwargs: Any):
    facts_keys = {"customer_refunds_in_window_paise", "has_replacement", "open_ticket_reference"}
    facts = PolicyFacts(
        now=kwargs.pop("now", NOW),
        order=order,
        **{k: kwargs.pop(k) for k in list(kwargs) if k in facts_keys},
    )
    defaults = {"order_number": order.order_number if order else None, "sku": None}
    args = SimpleNamespace(**(defaults | kwargs))
    return eng.evaluate(ProposedAction(tool, args), facts)
