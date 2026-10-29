"""Deterministic mock e-commerce data. Edge cases are documented in docs/seed-cases.md.

python -m app.db.seed           # refuses if data already exists
python -m app.db.seed --reset   # truncates all domain tables first
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    Base,
    Conversation,
    ConversationTurn,
    Customer,
    Order,
    OrderItem,
    Refund,
    Return,
    Shipment,
    ShipmentEvent,
    Ticket,
)
from app.domain.enums import (
    ConversationChannel,
    ConversationOutcome,
    OrderStatus,
    RefundReason,
    RefundStatus,
    ReturnReason,
    ReturnStatus,
    ShipmentEventCode,
    ShipmentStatus,
    TicketCategory,
    TicketPriority,
    TicketStatus,
    TurnRole,
)
from app.domain.order_ids import spoken_confusables
from app.policy.config import get_policy

SEED = 20261029
# The dataset's "now". Fixed so relative cases (e.g. "return window expired by 1 day") stay
# exact; the policy engine and evals must use this as their clock when reasoning about seed data.
ANCHOR = datetime(2026, 10, 29, 6, 30, tzinfo=UTC)  # 12:00 IST
TOTAL_CUSTOMERS = 50
TOTAL_ORDERS = 200


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_paise: int


CATALOG = {
    p.sku: p
    for p in (
        Product("TSH-BLK-M", "Cotton crew T-shirt, black, M", 599_00),
        Product("JNS-IND-32", "Slim fit jeans, indigo, 32", 1_899_00),
        Product("SNK-WHT-9", "Canvas sneakers, white, UK 9", 2_499_00),
        Product("KTL-ELC-15", "Electric kettle 1.5 L", 1_299_00),
        Product("BTL-STL-1", "Steel water bottle 1 L", 449_00),
        Product("EAR-TWS-01", "Wireless earbuds", 2_999_00),
        Product("PWB-10K", "Power bank 10,000 mAh", 1_199_00),
        Product("CHR-USBC-65", "65 W USB-C charger", 1_799_00),
        Product("BKS-NOV-01", "Paperback novel", 399_00),
        Product("YGA-MAT-6", "Yoga mat 6 mm", 899_00),
        Product("LMP-DSK-LED", "LED desk lamp", 1_499_00),
        Product("CKW-NS-24", "Non-stick frying pan 24 cm", 1_099_00),
        Product("BED-SHT-Q", "Cotton bedsheet set, queen", 1_699_00),
        Product("WCH-SMT-02", "Smartwatch", 4_999_00),
        Product("SPK-BT-10", "Bluetooth speaker", 3_499_00),
        Product("MSE-WL-01", "Wireless mouse", 799_00),
        Product("KBD-MEC-01", "Mechanical keyboard", 3_999_00),
        Product("SAR-SLK-01", "Silk saree", 6_499_00),
        Product("TEA-ASM-500", "Assam tea 500 g", 349_00),
        Product("SOC-CTN-3P", "Cotton socks, 3 pack", 299_00),
    )
}
# Not in the generic catalog: only the high-value edge case orders it.
LAPTOP = Product("LAP-PRO-16", "ProBook 16 laptop, 32 GB", 84_999_00)

FIRST_NAMES = (
    "Aditi", "Akash", "Amit", "Anjali", "Deepak", "Divya", "Farhan", "Gaurav", "Ishaan", "Jaya",
    "Karan", "Lakshmi", "Manish", "Nikhil", "Pooja", "Rajesh", "Ritu", "Sanjay", "Shreya", "Sneha",
    "Suresh", "Tanvi", "Uday", "Varun", "Zoya",
)  # fmt: skip
LAST_NAMES = (
    "Agarwal", "Banerjee", "Chopra", "Das", "Desai", "Joshi", "Kapoor", "Kulkarni", "Menon",
    "Mishra", "Patel", "Reddy", "Saxena", "Shetty", "Sinha", "Thakur", "Yadav",
)  # fmt: skip
# First three digits of real-looking PIN codes: Mumbai, Delhi, Bengaluru, Chennai, Kolkata,
# Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow.
PIN_PREFIXES = ("400", "110", "560", "600", "700", "500", "411", "380", "302", "226")
HUBS = ("Bhiwandi", "Gurugram", "Hosakote", "Chennai", "Kolkata", "Hyderabad", "Pune", "Nagpur")
# Fictional carriers.
CARRIERS = (("SwiftShip", "SS"), ("DakExpress", "DX"), ("Konnect Logistics", "KL"))


@dataclass
class Dataset:
    customers: list[Customer] = field(default_factory=list)
    conversations: list[Conversation] = field(default_factory=list)
    tickets: list[Ticket] = field(default_factory=list)

    @property
    def orders(self) -> list[Order]:
        return [o for c in self.customers for o in c.orders]


def _days(n: float) -> timedelta:
    return timedelta(days=n)


def _hours(n: float) -> timedelta:
    return timedelta(hours=n)


class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.data = Dataset()
        self._phones: set[str] = set()
        self._tracking: set[str] = set()
        self._refund_seq = 0
        self._return_seq = 0

    def customer(self, full_name: str, email: str | None = None) -> Customer:
        n = len(self.data.customers) + 1
        if email is None:
            first, last = full_name.lower().split(" ", 1)
            email = f"{first}.{last}{n:02d}@example.com"
        phone = f"+91{self.rng.randint(6, 9)}{self.rng.randrange(10**9):09d}"
        while phone in self._phones:
            phone = f"+91{self.rng.randint(6, 9)}{self.rng.randrange(10**9):09d}"
        self._phones.add(phone)
        customer = Customer(
            full_name=full_name,
            email=email,
            phone=phone,
            postal_code=f"{self.rng.choice(PIN_PREFIXES)}{self.rng.randrange(1000):03d}",
            created_at=ANCHOR - _days(self.rng.randint(150, 900)),
        )
        self.data.customers.append(customer)
        return customer

    def order(
        self,
        customer: Customer,
        number: str,
        placed_at: datetime,
        lines: list[tuple[Product, int]],
        status: OrderStatus,
        cancelled_at: datetime | None = None,
    ) -> Order:
        items = [
            OrderItem(sku=p.sku, product_name=p.name, quantity=q, unit_price_paise=p.price_paise)
            for p, q in lines
        ]
        subtotal = sum(i.line_total_paise for i in items)
        rules = get_policy().shipping
        free = subtotal >= rules.free_shipping_threshold_paise
        shipping = 0 if free else rules.standard_fee_paise
        order = Order(
            order_number=number,
            status=status,
            subtotal_paise=subtotal,
            shipping_fee_paise=shipping,
            total_paise=subtotal + shipping,
            placed_at=placed_at,
            cancelled_at=cancelled_at,
            items=items,
        )
        customer.orders.append(order)
        return order

    def ship(
        self,
        order: Order,
        items: list[OrderItem],
        *,
        shipped_at: datetime,
        status: ShipmentStatus,
        eta: datetime,
        delivered_at: datetime | None = None,
        delivery_note: str = "Delivered",
    ) -> Shipment:
        carrier, prefix = self.rng.choice(CARRIERS)
        tracking = f"{prefix}{self.rng.randrange(10**10):010d}"
        while tracking in self._tracking:
            tracking = f"{prefix}{self.rng.randrange(10**10):010d}"
        self._tracking.add(tracking)
        origin, dest = self.rng.sample(HUBS, 2)

        def event(code: ShipmentEventCode, at: datetime, where: str | None, desc: str):
            return ShipmentEvent(code=code, occurred_at=at, location=where, description=desc)

        events = [
            event(ShipmentEventCode.LABEL_CREATED, shipped_at - _hours(6), None, "Label created"),
            event(ShipmentEventCode.PICKED_UP, shipped_at, origin, "Picked up by carrier"),
        ]
        in_transit_at = shipped_at + _hours(20)
        if in_transit_at <= ANCHOR:
            events.append(
                event(ShipmentEventCode.IN_TRANSIT, in_transit_at, dest, "Arrived at hub")
            )
        if status in (ShipmentStatus.OUT_FOR_DELIVERY, ShipmentStatus.DELIVERED):
            out_at = (delivered_at or ANCHOR) - _hours(5)
            events.append(
                event(ShipmentEventCode.OUT_FOR_DELIVERY, out_at, dest, "Out for delivery")
            )
        if status == ShipmentStatus.DELIVERED:
            if delivered_at is None:
                raise ValueError("delivered shipment needs delivered_at")
            events.append(event(ShipmentEventCode.DELIVERED, delivered_at, dest, delivery_note))

        shipment = Shipment(
            carrier=carrier,
            tracking_number=tracking,
            status=status,
            shipped_at=shipped_at,
            estimated_delivery_at=eta,
            delivered_at=delivered_at,
            events=events,
        )
        order.shipments.append(shipment)
        for item in items:
            item.shipment = shipment
        return shipment

    def return_(
        self,
        order: Order,
        item: OrderItem,
        *,
        reason: ReturnReason,
        status: ReturnStatus,
        requested_at: datetime,
        resolved_at: datetime | None = None,
    ) -> Return:
        self._return_seq += 1
        return Return(
            order=order,
            item=item,
            quantity=item.quantity,
            reason=reason,
            status=status,
            idempotency_key=f"seed-return-{self._return_seq:04d}",
            requested_at=requested_at,
            resolved_at=resolved_at,
        )

    def refund(
        self,
        order: Order,
        amount_paise: int,
        *,
        reason: RefundReason,
        status: RefundStatus,
        requested_at: datetime,
        processed_at: datetime | None = None,
        return_: Return | None = None,
    ) -> Refund:
        self._refund_seq += 1
        return Refund(
            order=order,
            return_=return_,
            amount_paise=amount_paise,
            status=status,
            reason=reason,
            idempotency_key=f"seed-refund-{self._refund_seq:04d}",
            requested_at=requested_at,
            processed_at=processed_at,
        )

    def ticket(
        self,
        *,
        customer: Customer,
        order: Order | None,
        category: TicketCategory,
        status: TicketStatus,
        priority: TicketPriority,
        summary: str,
        created_at: datetime,
        conversation: Conversation | None = None,
    ) -> Ticket:
        ticket = Ticket(
            customer=customer,
            order=order,
            conversation=conversation,
            category=category,
            status=status,
            priority=priority,
            summary=summary,
            created_at=created_at,
            updated_at=created_at + _hours(1),
        )
        self.data.tickets.append(ticket)
        return ticket

    def conversation(
        self,
        *,
        customer: Customer | None,
        started_at: datetime,
        verified: bool,
        outcome: ConversationOutcome,
        turns: list[tuple[TurnRole, str]],
    ) -> Conversation:
        conv = Conversation(
            customer=customer,
            channel=ConversationChannel.VOICE,
            started_at=started_at,
            verified_at=started_at + timedelta(seconds=40) if verified else None,
            ended_at=started_at + timedelta(seconds=20 * len(turns)),
            outcome=outcome,
            turns=[
                ConversationTurn(
                    turn_index=i,
                    role=role,
                    content=content,
                    created_at=started_at + timedelta(seconds=20 * i),
                )
                for i, (role, content) in enumerate(turns)
            ],
        )
        self.data.conversations.append(conv)
        return conv


def _p(sku: str) -> Product:
    return CATALOG[sku]


def _delivered_order(
    b: _Builder,
    customer: Customer,
    number: str,
    delivered_days_ago: float,
    lines: list[tuple[Product, int]],
) -> Order:
    """Single-shipment order delivered `delivered_days_ago` days before ANCHOR."""
    delivered_at = ANCHOR - _days(delivered_days_ago)
    placed_at = delivered_at - _days(4)
    order = b.order(customer, number, placed_at, lines, OrderStatus.DELIVERED)
    b.ship(
        order,
        order.items,
        shipped_at=placed_at + _days(1),
        status=ShipmentStatus.DELIVERED,
        eta=delivered_at + _hours(6),
        delivered_at=delivered_at,
    )
    return order


def _edge_cases(b: _Builder) -> None:
    A = ANCHOR

    # Partial shipments.
    aarav = b.customer("Aarav Mehta", "aarav.mehta@example.com")
    o = b.order(
        aarav,
        "VX-2201",
        A - _days(6),
        [(_p("KBD-MEC-01"), 1), (_p("MSE-WL-01"), 1), (_p("LMP-DSK-LED"), 1)],
        OrderStatus.PARTIALLY_SHIPPED,
    )
    b.ship(
        o,
        o.items[:2],
        shipped_at=A - _days(5),
        status=ShipmentStatus.DELIVERED,
        eta=A - _days(3),
        delivered_at=A - _days(3),
    )  # lamp is backordered: never shipped
    o = b.order(
        aarav,
        "VX-2202",
        A - _days(5),
        [(_p("WCH-SMT-02"), 1), (_p("CHR-USBC-65"), 1)],
        OrderStatus.SHIPPED,
    )
    b.ship(
        o,
        o.items[:1],
        shipped_at=A - _days(4),
        status=ShipmentStatus.DELIVERED,
        eta=A - _days(2),
        delivered_at=A - _days(2),
    )
    b.ship(
        o, o.items[1:], shipped_at=A - _days(2), status=ShipmentStatus.IN_TRANSIT, eta=A + _days(2)
    )
    _delivered_order(b, aarav, "VX-2140", 40, [(_p("SOC-CTN-3P"), 2)])

    # Multiple active orders for one customer.
    kavya = b.customer("Kavya Nair", "kavya.nair@example.com")
    b.order(kavya, "VX-3301", A - _hours(5), [(_p("YGA-MAT-6"), 1)], OrderStatus.PLACED)
    o = b.order(kavya, "VX-3302", A - _days(2), [(_p("SNK-WHT-9"), 1)], OrderStatus.SHIPPED)
    b.ship(o, o.items, shipped_at=A - _days(1), status=ShipmentStatus.IN_TRANSIT, eta=A + _days(2))
    o = b.order(kavya, "VX-3303", A - _days(4), [(_p("BED-SHT-Q"), 2)], OrderStatus.SHIPPED)
    b.ship(o, o.items, shipped_at=A - _days(3), status=ShipmentStatus.OUT_FOR_DELIVERY, eta=A)
    _delivered_order(b, kavya, "VX-3190", 20, [(_p("TEA-ASM-500"), 2)])

    # Delivered but reported missing.
    sameer = b.customer("Sameer Khan", "sameer.khan@example.com")
    o = b.order(sameer, "VX-3107", A - _days(6), [(_p("EAR-TWS-01"), 1)], OrderStatus.DELIVERED)
    b.ship(
        o,
        o.items,
        shipped_at=A - _days(5),
        status=ShipmentStatus.DELIVERED,
        eta=A - _days(2),
        delivered_at=A - _days(2),
        delivery_note="Delivered, left at front door",
    )
    conv = b.conversation(
        customer=sameer,
        started_at=A - _days(1),
        verified=True,
        outcome=ConversationOutcome.ESCALATED,
        turns=[
            (TurnRole.ASSISTANT, "Thanks for calling. How can I help you today?"),
            (TurnRole.CUSTOMER, "My earbuds order says delivered but I never got it."),
            (TurnRole.ASSISTANT, "I can help. Can you confirm your registered phone number?"),
            (TurnRole.CUSTOMER, f"It ends in {' '.join(sameer.phone[-4:])}."),
            (TurnRole.TOOL, '{"tool": "get_order", "order_number": "VX-3107"}'),
            (TurnRole.ASSISTANT, "VX-3107 shows delivered two days ago. I've raised it with "
                                 "our delivery team; a specialist will call within 24 hours."),
        ],
    )  # fmt: skip
    b.ticket(
        customer=sameer,
        order=o,
        conversation=conv,
        category=TicketCategory.MISSING_DELIVERY,
        status=TicketStatus.OPEN,
        priority=TicketPriority.HIGH,
        summary="Carrier shows delivered (left at front door); customer says not received.",
        created_at=conv.ended_at,
    )

    # Return window expired by 1 day, plus a companion with 1 day left.
    neha = b.customer("Neha Gupta", "neha.gupta@example.com")
    _delivered_order(b, neha, "VX-4410", 31, [(_p("JNS-IND-32"), 1)])
    _delivered_order(b, neha, "VX-4411", 29, [(_p("TSH-BLK-M"), 2)])

    # Already refunded (fully), and partially refunded.
    vikram = b.customer("Vikram Singh", "vikram.singh@example.com")
    o = _delivered_order(b, vikram, "VX-5120", 14, [(_p("SPK-BT-10"), 1)])
    ret = b.return_(
        o,
        o.items[0],
        reason=ReturnReason.DAMAGED,
        status=ReturnStatus.COMPLETED,
        requested_at=A - _days(12),
        resolved_at=A - _days(8),
    )
    b.refund(
        o,
        o.total_paise,
        reason=RefundReason.RETURN,
        status=RefundStatus.SUCCEEDED,
        requested_at=A - _days(8),
        processed_at=A - _days(8) + _hours(2),
        return_=ret,
    )
    o = _delivered_order(b, vikram, "VX-5121", 10, [(_p("KTL-ELC-15"), 1), (_p("CKW-NS-24"), 1)])
    ret = b.return_(
        o,
        o.items[0],
        reason=ReturnReason.NOT_AS_DESCRIBED,
        status=ReturnStatus.COMPLETED,
        requested_at=A - _days(8),
        resolved_at=A - _days(5),
    )
    b.refund(
        o,
        o.items[0].line_total_paise,
        reason=RefundReason.RETURN,
        status=RefundStatus.SUCCEEDED,
        requested_at=A - _days(5),
        processed_at=A - _days(5) + _hours(2),
        return_=ret,
    )

    # Cancelled orders: one refunded, one with the refund still pending.
    ananya = b.customer("Ananya Rao", "ananya.rao@example.com")
    for number, placed_days_ago, refund_status in (
        ("VX-6031", 9, RefundStatus.SUCCEEDED),
        ("VX-6032", 2, RefundStatus.PENDING),
    ):
        placed_at = A - _days(placed_days_ago)
        o = b.order(
            ananya,
            number,
            placed_at,
            [(_p("SAR-SLK-01"), 1)] if number == "VX-6031" else [(_p("PWB-10K"), 1)],
            OrderStatus.CANCELLED,
            cancelled_at=placed_at + _hours(2),
        )
        b.refund(
            o,
            o.total_paise,
            reason=RefundReason.CANCELLATION,
            status=refund_status,
            requested_at=placed_at + _hours(2),
            processed_at=placed_at + _days(1) if refund_status == RefundStatus.SUCCEEDED else None,
        )

    # Sound-alike order numbers, same customer.
    rahul = b.customer("Rahul Verma", "rahul.verma@example.com")
    _delivered_order(b, rahul, "VX-1508", 8, [(_p("PWB-10K"), 1)])
    _delivered_order(b, rahul, "VX-1580", 6, [(_p("BTL-STL-1"), 2)])

    # Sound-alike order numbers across different customers.
    rohan = b.customer("Rohan Iyer", "rohan.iyer@example.com")
    _delivered_order(b, rohan, "VX-1315", 7, [(_p("SAR-SLK-01"), 1)])
    priya_a = b.customer("Priya Sharma", "priya.sharma@example.com")
    o = b.order(priya_a, "VX-1350", A - _days(3), [(_p("WCH-SMT-02"), 1)], OrderStatus.SHIPPED)
    b.ship(o, o.items, shipped_at=A - _days(2), status=ShipmentStatus.IN_TRANSIT, eta=A + _days(1))

    # Duplicate full name: a second, unrelated Priya Sharma.
    priya_b = b.customer("Priya Sharma", "priya.sharma2@example.com")
    _delivered_order(b, priya_b, "VX-9012", 15, [(_p("BKS-NOV-01"), 3)])

    # High-value order above the auto-refund cap, still inside the return window.
    arjun = b.customer("Arjun Malhotra", "arjun.malhotra@example.com")
    _delivered_order(b, arjun, "VX-7302", 5, [(LAPTOP, 1)])

    # Shipment stuck in transit well past its ETA.
    meera = b.customer("Meera Pillai", "meera.pillai@example.com")
    o = b.order(meera, "VX-8015", A - _days(10), [(_p("SPK-BT-10"), 1)], OrderStatus.SHIPPED)
    b.ship(o, o.items, shipped_at=A - _days(9), status=ShipmentStatus.IN_TRANSIT, eta=A - _days(4))

    # Unverified caller asking about someone else's order.
    b.conversation(
        customer=None,
        started_at=A - _hours(3),
        verified=False,
        outcome=ConversationOutcome.ABANDONED,
        turns=[
            (TurnRole.ASSISTANT, "Thanks for calling. How can I help you today?"),
            (TurnRole.CUSTOMER, "Hi, this is Priya Sharma. Where is my order thirteen fifty?"),
            (TurnRole.ASSISTANT, "I can check. Can you confirm the phone number on your account?"),
            (TurnRole.CUSTOMER, "I don't have it with me. Just tell me the delivery address."),
            (TurnRole.ASSISTANT, "Sorry, I can't share order details until I verify your account."),
        ],
    )  # fmt: skip


def _pick_generic_numbers(rng: random.Random, taken: set[str], count: int) -> list[str]:
    """Order numbers that are not spoken-confusable with any other number in the dataset."""
    blocked = set(taken)
    for number in taken:
        blocked |= spoken_confusables(number)
    pool = [f"VX-{n}" for n in range(1000, 10000)]
    rng.shuffle(pool)
    chosen: list[str] = []
    for number in pool:
        if number in blocked:
            continue
        chosen.append(number)
        blocked.add(number)
        blocked |= spoken_confusables(number)
        if len(chosen) == count:
            break
    return sorted(chosen)


def _generic_orders(b: _Builder, customers: list[Customer], count: int) -> None:
    rng = b.rng
    taken = {o.order_number for o in b.data.orders}
    numbers = _pick_generic_numbers(rng, taken, count)
    ages = sorted((rng.uniform(0.1, 120) for _ in range(count)), reverse=True)
    products = list(CATALOG.values())

    for number, age in zip(numbers, ages, strict=True):
        customer = rng.choice(customers)
        placed_at = (ANCHOR - _days(age)).replace(second=0, microsecond=0)
        n_lines = rng.choices((1, 2, 3), weights=(60, 30, 10))[0]
        lines = [
            (p, rng.choices((1, 2), weights=(85, 15))[0]) for p in rng.sample(products, n_lines)
        ]

        if age > 1 and rng.random() < 0.05:
            o = b.order(
                customer,
                number,
                placed_at,
                lines,
                OrderStatus.CANCELLED,
                cancelled_at=placed_at + _hours(3),
            )
            b.refund(
                o,
                o.total_paise,
                reason=RefundReason.CANCELLATION,
                status=RefundStatus.SUCCEEDED,
                requested_at=placed_at + _hours(3),
                processed_at=placed_at + _days(1),
            )
            continue

        transit_days = rng.randint(3, 6)
        shipped_at = placed_at + _days(1)
        delivered_at = placed_at + _days(transit_days) + _hours(rng.randint(0, 8))
        eta = placed_at + _days(transit_days) + _hours(12)

        if shipped_at > ANCHOR:
            b.order(customer, number, placed_at, lines, OrderStatus.PLACED)
            continue
        if delivered_at > ANCHOR:
            o = b.order(customer, number, placed_at, lines, OrderStatus.SHIPPED)
            b.ship(o, o.items, shipped_at=shipped_at, status=ShipmentStatus.IN_TRANSIT, eta=eta)
            continue

        o = b.order(customer, number, placed_at, lines, OrderStatus.DELIVERED)
        b.ship(
            o,
            o.items,
            shipped_at=shipped_at,
            status=ShipmentStatus.DELIVERED,
            eta=eta,
            delivered_at=delivered_at,
        )
        roll = rng.random()
        returned_at = delivered_at + _days(rng.randint(2, 10))
        if roll < 0.06 and returned_at + _days(4) <= ANCHOR:
            item = o.items[0]
            ret = b.return_(
                o,
                item,
                reason=rng.choice(list(ReturnReason)),
                status=ReturnStatus.COMPLETED,
                requested_at=returned_at,
                resolved_at=returned_at + _days(4),
            )
            b.refund(
                o,
                item.line_total_paise,
                reason=RefundReason.RETURN,
                status=RefundStatus.SUCCEEDED,
                requested_at=returned_at + _days(4),
                processed_at=returned_at + _days(4) + _hours(2),
                return_=ret,
            )
        elif roll > 0.96:
            b.ticket(
                customer=customer,
                order=o,
                category=TicketCategory.ORDER_STATUS,
                status=TicketStatus.RESOLVED,
                priority=TicketPriority.LOW,
                summary="Asked for delivery date; shared tracking update.",
                created_at=shipped_at + _hours(12),
            )


def build_dataset(seed: int = SEED) -> Dataset:
    b = _Builder(random.Random(seed))  # noqa: S311 - reproducible mock data, not crypto
    _edge_cases(b)
    edge_order_count = len(b.data.orders)
    generic_customers = [
        b.customer(f"{b.rng.choice(FIRST_NAMES)} {b.rng.choice(LAST_NAMES)}")
        for _ in range(TOTAL_CUSTOMERS - len(b.data.customers))
    ]
    _generic_orders(b, generic_customers, TOTAL_ORDERS - edge_order_count)
    return b.data


def write_dataset(session: Session, data: Dataset, *, reset: bool = False) -> None:
    if reset:
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    elif session.scalar(select(func.count()).select_from(Customer)):
        raise RuntimeError("Database already has customers; rerun with --reset to replace them.")
    # Orders, shipments, items, returns and refunds cascade from customers.
    session.add_all(data.customers)
    session.add_all(data.conversations)
    session.add_all(data.tickets)
    session.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="truncate domain tables first")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.reset and settings.app_env == "production":
        print("Refusing to --reset when APP_ENV=production.", file=sys.stderr)
        return 2

    data = build_dataset()
    # Built before writing: committed objects expire, and lazy="raise" blocks reloading them.
    summary = (
        f"Seeded {len(data.customers)} customers, {len(data.orders)} orders, "
        f"{len(data.tickets)} tickets, {len(data.conversations)} conversations "
        f"into {settings.postgres_db!r}."
    )
    engine = create_engine(settings.database_url())
    try:
        with Session(engine) as session, session.begin():
            write_dataset(session, data, reset=args.reset)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
