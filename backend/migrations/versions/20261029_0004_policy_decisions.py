"""Policy decision audit log.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "policy_decisions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("tool_name", sa.String(length=64), nullable=False),
        sa.Column("action_id", sa.String(length=40), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("rule", sa.String(length=80), nullable=False),
        sa.Column("reason", sa.String(length=40), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("amount_paise", sa.BigInteger(), nullable=True),
        sa.Column("policy_version", sa.String(length=40), nullable=False),
        sa.Column("facts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("customer_id", sa.BigInteger(), nullable=True),
        sa.Column("conversation_id", sa.BigInteger(), nullable=True),
        sa.CheckConstraint(
            "stage IN ('proposal', 'execution')", name=op.f("ck_policy_decisions_stage_known")
        ),
        sa.CheckConstraint(
            "verdict IN ('approve', 'reject', 'require_human')",
            name=op.f("ck_policy_decisions_verdict_known"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_policy_decisions_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["customer_id"],
            ["customers.id"],
            name=op.f("fk_policy_decisions_customer_id_customers"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_decisions")),
    )
    op.create_index(
        op.f("ix_policy_decisions_action_id"), "policy_decisions", ["action_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_policy_decisions_action_id"), table_name="policy_decisions")
    op.drop_table("policy_decisions")
