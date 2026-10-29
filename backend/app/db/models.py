from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

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

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {datetime: DateTime(timezone=True)}  # noqa: RUF012 - SQLAlchemy API


def str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    # VARCHAR + CHECK rather than a native PG enum: adding a value later is a plain
    # constraint swap instead of ALTER TYPE, and Python still gets enum members back.
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


def pk() -> Mapped[int]:
    return mapped_column(BigInteger, Identity(), primary_key=True)


def created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = pk()
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    phone: Mapped[str] = mapped_column(String(20), unique=True)
    postal_code: Mapped[str] = mapped_column(String(6))
    created_at: Mapped[datetime] = created_at()

    orders: Mapped[list["Order"]] = relationship(back_populates="customer", lazy="raise")

    __table_args__ = (
        CheckConstraint("email = lower(email)", name="email_lowercase"),
        CheckConstraint("phone ~ '^\\+91[6-9][0-9]{9}$'", name="phone_e164_in"),
        CheckConstraint("postal_code ~ '^[1-9][0-9]{5}$'", name="postal_code_format"),
    )


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = pk()
    order_number: Mapped[str] = mapped_column(String(16), unique=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(str_enum(OrderStatus, "order_status"))
    currency: Mapped[str] = mapped_column(String(3), server_default="INR")
    subtotal_paise: Mapped[int] = mapped_column(BigInteger)
    shipping_fee_paise: Mapped[int] = mapped_column(BigInteger)
    total_paise: Mapped[int] = mapped_column(BigInteger)
    placed_at: Mapped[datetime]
    cancelled_at: Mapped[datetime | None]
    # Set on replacement orders created by a reshipment.
    replaces_order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), index=True)

    customer: Mapped[Customer] = relationship(back_populates="orders", lazy="raise")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", lazy="raise", order_by="OrderItem.id"
    )
    shipments: Mapped[list["Shipment"]] = relationship(
        back_populates="order", lazy="raise", order_by="Shipment.id"
    )
    returns: Mapped[list["Return"]] = relationship(
        back_populates="order", lazy="raise", order_by="Return.id"
    )
    refunds: Mapped[list["Refund"]] = relationship(
        back_populates="order", lazy="raise", order_by="Refund.id"
    )

    __table_args__ = (
        CheckConstraint("order_number ~ '^VX-[0-9]{4}$'", name="order_number_format"),
        CheckConstraint("currency = 'INR'", name="currency_inr"),
        CheckConstraint(
            "subtotal_paise >= 0 AND shipping_fee_paise >= 0", name="amounts_non_negative"
        ),
        CheckConstraint("total_paise = subtotal_paise + shipping_fee_paise", name="total_sum"),
        CheckConstraint(
            "(status = 'cancelled') = (cancelled_at IS NOT NULL)", name="cancelled_at_matches"
        ),
    )


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[int] = pk()
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    carrier: Mapped[str] = mapped_column(String(40))
    tracking_number: Mapped[str] = mapped_column(String(40), unique=True)
    status: Mapped[ShipmentStatus] = mapped_column(str_enum(ShipmentStatus, "shipment_status"))
    shipped_at: Mapped[datetime | None]
    estimated_delivery_at: Mapped[datetime | None]
    delivered_at: Mapped[datetime | None]

    order: Mapped[Order] = relationship(back_populates="shipments", lazy="raise")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="shipment", lazy="raise", order_by="OrderItem.id"
    )
    events: Mapped[list["ShipmentEvent"]] = relationship(
        back_populates="shipment", lazy="raise", order_by="ShipmentEvent.occurred_at"
    )

    __table_args__ = (
        CheckConstraint(
            "(status = 'delivered') = (delivered_at IS NOT NULL)", name="delivered_at_matches"
        ),
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = pk()
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    # NULL until the item is packed into a shipment; lets one order span several shipments.
    shipment_id: Mapped[int | None] = mapped_column(
        ForeignKey("shipments.id", ondelete="SET NULL"), index=True
    )
    sku: Mapped[str] = mapped_column(String(32))
    product_name: Mapped[str] = mapped_column(String(200))
    quantity: Mapped[int]
    unit_price_paise: Mapped[int] = mapped_column(BigInteger)

    order: Mapped[Order] = relationship(back_populates="items", lazy="raise")
    shipment: Mapped[Shipment | None] = relationship(back_populates="items", lazy="raise")

    __table_args__ = (
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint("unit_price_paise >= 0", name="unit_price_non_negative"),
    )

    @property
    def line_total_paise(self) -> int:
        return self.quantity * self.unit_price_paise


class ShipmentEvent(Base):
    __tablename__ = "shipment_events"

    id: Mapped[int] = pk()
    shipment_id: Mapped[int] = mapped_column(ForeignKey("shipments.id", ondelete="CASCADE"))
    code: Mapped[ShipmentEventCode] = mapped_column(
        str_enum(ShipmentEventCode, "shipment_event_code")
    )
    occurred_at: Mapped[datetime]
    location: Mapped[str | None] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(String(300))

    shipment: Mapped[Shipment] = relationship(back_populates="events", lazy="raise")

    __table_args__ = (Index(None, "shipment_id", "occurred_at"),)


class Return(Base):
    __tablename__ = "returns"

    id: Mapped[int] = pk()
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    order_item_id: Mapped[int | None] = mapped_column(ForeignKey("order_items.id"))
    quantity: Mapped[int]
    reason: Mapped[ReturnReason] = mapped_column(str_enum(ReturnReason, "return_reason"))
    status: Mapped[ReturnStatus] = mapped_column(str_enum(ReturnStatus, "return_status"))
    notes: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), unique=True)
    requested_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None]

    order: Mapped[Order] = relationship(back_populates="returns", lazy="raise")
    item: Mapped[OrderItem | None] = relationship(lazy="raise")
    refunds: Mapped[list["Refund"]] = relationship(back_populates="return_", lazy="raise")

    __table_args__ = (CheckConstraint("quantity > 0", name="quantity_positive"),)


class Refund(Base):
    __tablename__ = "refunds"

    id: Mapped[int] = pk()
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    return_id: Mapped[int | None] = mapped_column(ForeignKey("returns.id"))
    amount_paise: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[RefundStatus] = mapped_column(str_enum(RefundStatus, "refund_status"))
    reason: Mapped[RefundReason] = mapped_column(str_enum(RefundReason, "refund_reason"))
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    requested_at: Mapped[datetime]
    processed_at: Mapped[datetime | None]

    order: Mapped[Order] = relationship(back_populates="refunds", lazy="raise")
    return_: Mapped[Return | None] = relationship(back_populates="refunds", lazy="raise")

    __table_args__ = (CheckConstraint("amount_paise > 0", name="amount_positive"),)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = pk()
    # NULL until (unless) the caller is verified.
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"), index=True)
    channel: Mapped[ConversationChannel] = mapped_column(
        str_enum(ConversationChannel, "conversation_channel")
    )
    started_at: Mapped[datetime]
    verified_at: Mapped[datetime | None]
    ended_at: Mapped[datetime | None]
    outcome: Mapped[ConversationOutcome | None] = mapped_column(
        str_enum(ConversationOutcome, "conversation_outcome")
    )

    customer: Mapped[Customer | None] = relationship(lazy="raise")
    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation", lazy="raise", order_by="ConversationTurn.turn_index"
    )

    __table_args__ = (
        CheckConstraint(
            "verified_at IS NULL OR customer_id IS NOT NULL", name="verified_has_customer"
        ),
    )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[int] = pk()
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    turn_index: Mapped[int]
    role: Mapped[TurnRole] = mapped_column(str_enum(TurnRole, "turn_role"))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]

    conversation: Mapped[Conversation] = relationship(back_populates="turns", lazy="raise")

    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_index"),
        CheckConstraint("turn_index >= 0", name="turn_index_non_negative"),
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = pk()
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), index=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    category: Mapped[TicketCategory] = mapped_column(str_enum(TicketCategory, "ticket_category"))
    status: Mapped[TicketStatus] = mapped_column(str_enum(TicketStatus, "ticket_status"))
    priority: Mapped[TicketPriority] = mapped_column(str_enum(TicketPriority, "ticket_priority"))
    summary: Mapped[str] = mapped_column(Text)
    # Structured handoff payload for escalations (reason, customer context, actions attempted).
    details: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    customer: Mapped[Customer | None] = relationship(lazy="raise")
    order: Mapped[Order | None] = relationship(lazy="raise")
    conversation: Mapped[Conversation | None] = relationship(lazy="raise")


class ToolInvocation(Base):
    """Idempotency ledger for write tools: one row per key, holding the original result."""

    __tablename__ = "tool_invocations"

    id: Mapped[int] = pk()
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    tool_name: Mapped[str] = mapped_column(String(64))
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"))
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    # sha256 of the canonical arguments; a reused key with different arguments is rejected.
    request_hash: Mapped[str] = mapped_column(String(64))
    response: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = created_at()
