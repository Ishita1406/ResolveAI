"""Policy document chunks for hybrid retrieval (full-text + pgvector).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Copied rather than imported: a migration must not change when the model module does.
EMBEDDING_DIM = 384
TSV = (
    "setweight(to_tsvector('english', section), 'A') || "
    "setweight(to_tsvector('english', content), 'B') || "
    "setweight(to_tsvector('english', title), 'C')"
)


def upgrade() -> None:
    op.create_table(
        "policy_chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("document", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("section", sa.String(length=200), nullable=False),
        sa.Column("section_id", sa.String(length=200), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("embedding_model", sa.String(length=100), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=EMBEDDING_DIM), nullable=False),
        sa.Column("tsv", postgresql.TSVECTOR(), sa.Computed(TSV, persisted=True), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_policy_chunks")),
        sa.UniqueConstraint(
            "document", "section_id", name=op.f("uq_policy_chunks_document_section_id")
        ),
    )
    op.create_index("ix_policy_chunks_tsv", "policy_chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_policy_chunks_embedding",
        "policy_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_policy_chunks_embedding", table_name="policy_chunks")
    op.drop_index("ix_policy_chunks_tsv", table_name="policy_chunks")
    op.drop_table("policy_chunks")
