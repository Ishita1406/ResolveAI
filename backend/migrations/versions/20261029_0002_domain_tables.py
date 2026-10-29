"""Domain tables: customers, orders, shipments, returns, refunds, tickets, conversations.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("phone", sa.String(length=20), nullable=False),
        sa.Column("postal_code", sa.String(length=6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "phone ~ '^\\+91[6-9][0-9]{9}$'", name=op.f("ck_customers_phone_e164_in")
        ),
        sa.CheckConstraint(
            "postal_code ~ '^[1-9][0-9]{5}$'", name=op.f("ck_customers_postal_code_format")
        ),
        sa.CheckConstraint("email = lower(email)", name=op.f("ck_customers_email_lowercase")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_customers")),
        sa.UniqueConstraint("email", name=op.f("uq_customers_email")),
        sa.UniqueConstraint("phone", name=op.f("uq_customers_phone")),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "channel",
            sa.Enum(
                "voice",
                "text",
                name="conversation_channel",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "outcome",
            sa.Enum(
                "resolved",
                "escalated",
                "abandoned",
                name="conversation_outcome",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=True,
        ),
        sa.CheckConstraint(
            "verified_at IS NULL OR customer_id IS NOT NULL",
            name=op.f("ck_conversations_verified_has_customer"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_conversations_customer_id_customers")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_index(
        op.f("ix_conversations_customer_id"), "conversations", ["customer_id"], unique=False
    )
    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("order_number", sa.String(length=16), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "placed",
                "shipped",
                "partially_shipped",
                "delivered",
                "cancelled",
                name="order_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("currency", sa.String(length=3), server_default="INR", nullable=False),
        sa.Column("subtotal_paise", sa.BigInteger(), nullable=False),
        sa.Column("shipping_fee_paise", sa.BigInteger(), nullable=False),
        sa.Column("total_paise", sa.BigInteger(), nullable=False),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'cancelled') = (cancelled_at IS NOT NULL)",
            name=op.f("ck_orders_cancelled_at_matches"),
        ),
        sa.CheckConstraint("currency = 'INR'", name=op.f("ck_orders_currency_inr")),
        sa.CheckConstraint(
            "order_number ~ '^VX-[0-9]{4}$'", name=op.f("ck_orders_order_number_format")
        ),
        sa.CheckConstraint(
            "subtotal_paise >= 0 AND shipping_fee_paise >= 0",
            name=op.f("ck_orders_amounts_non_negative"),
        ),
        sa.CheckConstraint(
            "total_paise = subtotal_paise + shipping_fee_paise", name=op.f("ck_orders_total_sum")
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_orders_customer_id_customers")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_orders")),
        sa.UniqueConstraint("order_number", name=op.f("uq_orders_order_number")),
    )
    op.create_index(op.f("ix_orders_customer_id"), "orders", ["customer_id"], unique=False)
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.Enum(
                "customer",
                "assistant",
                "tool",
                "system",
                name="turn_role",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "turn_index >= 0", name=op.f("ck_conversation_turns_turn_index_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_conversation_turns_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_turns")),
        sa.UniqueConstraint(
            "conversation_id",
            "turn_index",
            name=op.f("uq_conversation_turns_conversation_id_turn_index"),
        ),
    )
    op.create_table(
        "shipments",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("carrier", sa.String(length=40), nullable=False),
        sa.Column("tracking_number", sa.String(length=40), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "label_created",
                "in_transit",
                "out_for_delivery",
                "delivered",
                "lost",
                "returned_to_sender",
                name="shipment_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("shipped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("estimated_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'delivered') = (delivered_at IS NOT NULL)",
            name=op.f("ck_shipments_delivered_at_matches"),
        ),
        sa.ForeignKeyConstraint(
            ["order_id"],
            ["orders.id"],
            name=op.f("fk_shipments_order_id_orders"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shipments")),
        sa.UniqueConstraint("tracking_number", name=op.f("uq_shipments_tracking_number")),
    )
    op.create_index(op.f("ix_shipments_order_id"), "shipments", ["order_id"], unique=False)
    op.create_table(
        "tickets",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=True),
        sa.Column("order_id", sa.BigInteger(), nullable=True),
        sa.Column("conversation_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "category",
            sa.Enum(
                "missing_delivery",
                "damaged_product",
                "return_refund",
                "order_status",
                "other",
                name="ticket_category",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "open",
                "pending_customer",
                "escalated",
                "resolved",
                name="ticket_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "priority",
            sa.Enum(
                "low",
                "normal",
                "high",
                name="ticket_priority",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_tickets_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"], ["customers.id"], name=op.f("fk_tickets_customer_id_customers")
        ),
        sa.ForeignKeyConstraint(
            ["order_id"], ["orders.id"], name=op.f("fk_tickets_order_id_orders")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tickets")),
    )
    op.create_index(op.f("ix_tickets_customer_id"), "tickets", ["customer_id"], unique=False)
    op.create_index(op.f("ix_tickets_order_id"), "tickets", ["order_id"], unique=False)
    op.create_table(
        "order_items",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("shipment_id", sa.BigInteger(), nullable=True),
        sa.Column("sku", sa.String(length=32), nullable=False),
        sa.Column("product_name", sa.String(length=200), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("unit_price_paise", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("quantity > 0", name=op.f("ck_order_items_quantity_positive")),
        sa.CheckConstraint(
            "unit_price_paise >= 0", name=op.f("ck_order_items_unit_price_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["order_id"],
            ["orders.id"],
            name=op.f("fk_order_items_order_id_orders"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["shipment_id"],
            ["shipments.id"],
            name=op.f("fk_order_items_shipment_id_shipments"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_order_items")),
    )
    op.create_index(op.f("ix_order_items_order_id"), "order_items", ["order_id"], unique=False)
    op.create_index(
        op.f("ix_order_items_shipment_id"), "order_items", ["shipment_id"], unique=False
    )
    op.create_table(
        "shipment_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("shipment_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "code",
            sa.Enum(
                "label_created",
                "picked_up",
                "in_transit",
                "out_for_delivery",
                "delivered",
                "delivery_attempted",
                "exception",
                name="shipment_event_code",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location", sa.String(length=120), nullable=True),
        sa.Column("description", sa.String(length=300), nullable=False),
        sa.ForeignKeyConstraint(
            ["shipment_id"],
            ["shipments.id"],
            name=op.f("fk_shipment_events_shipment_id_shipments"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_shipment_events")),
    )
    op.create_index(
        op.f("ix_shipment_events_shipment_id_occurred_at"),
        "shipment_events",
        ["shipment_id", "occurred_at"],
        unique=False,
    )
    op.create_table(
        "returns",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("order_item_id", sa.BigInteger(), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column(
            "reason",
            sa.Enum(
                "damaged",
                "wrong_item",
                "not_as_described",
                "no_longer_needed",
                name="return_reason",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "requested",
                "label_issued",
                "received",
                "rejected",
                "completed",
                name="return_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=64), nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("quantity > 0", name=op.f("ck_returns_quantity_positive")),
        sa.ForeignKeyConstraint(
            ["order_id"], ["orders.id"], name=op.f("fk_returns_order_id_orders")
        ),
        sa.ForeignKeyConstraint(
            ["order_item_id"], ["order_items.id"], name=op.f("fk_returns_order_item_id_order_items")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_returns")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_returns_idempotency_key")),
    )
    op.create_index(op.f("ix_returns_order_id"), "returns", ["order_id"], unique=False)
    op.create_table(
        "refunds",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("return_id", sa.BigInteger(), nullable=True),
        sa.Column("amount_paise", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "succeeded",
                "failed",
                name="refund_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.Enum(
                "return",
                "cancellation",
                "missing_delivery",
                "goodwill",
                name="refund_reason",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("amount_paise > 0", name=op.f("ck_refunds_amount_positive")),
        sa.ForeignKeyConstraint(
            ["order_id"], ["orders.id"], name=op.f("fk_refunds_order_id_orders")
        ),
        sa.ForeignKeyConstraint(
            ["return_id"], ["returns.id"], name=op.f("fk_refunds_return_id_returns")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_refunds")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_refunds_idempotency_key")),
    )
    op.create_index(op.f("ix_refunds_order_id"), "refunds", ["order_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_refunds_order_id"), table_name="refunds")
    op.drop_table("refunds")
    op.drop_index(op.f("ix_returns_order_id"), table_name="returns")
    op.drop_table("returns")
    op.drop_index(op.f("ix_shipment_events_shipment_id_occurred_at"), table_name="shipment_events")
    op.drop_table("shipment_events")
    op.drop_index(op.f("ix_order_items_shipment_id"), table_name="order_items")
    op.drop_index(op.f("ix_order_items_order_id"), table_name="order_items")
    op.drop_table("order_items")
    op.drop_index(op.f("ix_tickets_order_id"), table_name="tickets")
    op.drop_index(op.f("ix_tickets_customer_id"), table_name="tickets")
    op.drop_table("tickets")
    op.drop_index(op.f("ix_shipments_order_id"), table_name="shipments")
    op.drop_table("shipments")
    op.drop_table("conversation_turns")
    op.drop_index(op.f("ix_orders_customer_id"), table_name="orders")
    op.drop_table("orders")
    op.drop_index(op.f("ix_conversations_customer_id"), table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("customers")
