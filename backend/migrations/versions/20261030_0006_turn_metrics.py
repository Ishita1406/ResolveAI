"""Per-turn timings, usage and cost (observability).

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "turn_metrics",
        sa.Column("turn_id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.BigInteger(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("turn_index", sa.Integer(), nullable=True),
        sa.Column("channel", sa.String(length=16), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column("model", sa.String(length=80), nullable=True),
        sa.Column("prompt_version", sa.String(length=40), nullable=True),
        sa.Column("llm_calls", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Double(), nullable=True),
        sa.Column("llm_first_token_ms", sa.Double(), nullable=True),
        sa.Column("llm_ms", sa.Double(), nullable=True),
        sa.Column("tool_calls", sa.Integer(), nullable=True),
        sa.Column("tool_ms", sa.Double(), nullable=True),
        sa.Column("agent_ms", sa.Double(), nullable=True),
        sa.Column("vad_ms", sa.Double(), nullable=True),
        sa.Column("stt_ms", sa.Double(), nullable=True),
        sa.Column("agent_first_text_ms", sa.Double(), nullable=True),
        sa.Column("first_sentence_ms", sa.Double(), nullable=True),
        sa.Column("tts_first_audio_ms", sa.Double(), nullable=True),
        sa.Column("time_to_first_audio_ms", sa.Double(), nullable=True),
        sa.Column("filler", sa.Boolean(), nullable=True),
        sa.Column("interrupted", sa.Boolean(), nullable=True),
        sa.Column("trace_id", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_turn_metrics_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("turn_id", name=op.f("pk_turn_metrics")),
    )
    op.create_index(
        op.f("ix_turn_metrics_conversation_id"), "turn_metrics", ["conversation_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_turn_metrics_conversation_id"), table_name="turn_metrics")
    op.drop_table("turn_metrics")
