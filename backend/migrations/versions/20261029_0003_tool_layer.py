"""Tool layer: idempotency ledger, ticket handoff details, replacement-order link.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_invocations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=True),
        sa.Column("conversation_id", sa.BigInteger(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_tool_invocations_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_tool_invocations_customer_id_customers"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_invocations")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_tool_invocations_idempotency_key")),
    )
    op.add_column("orders", sa.Column("replaces_order_id", sa.BigInteger(), nullable=True))
    op.create_index(
        op.f("ix_orders_replaces_order_id"), "orders", ["replaces_order_id"], unique=False
    )
    op.create_foreign_key(
        op.f("fk_orders_replaces_order_id_orders"),
        "orders",
        "orders",
        ["replaces_order_id"],
        ["id"],
    )
    op.add_column(
        "tickets", sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("tickets", "details")
    op.drop_constraint(op.f("fk_orders_replaces_order_id_orders"), "orders", type_="foreignkey")
    op.drop_index(op.f("ix_orders_replaces_order_id"), table_name="orders")
    op.drop_column("orders", "replaces_order_id")
    op.drop_table("tool_invocations")
