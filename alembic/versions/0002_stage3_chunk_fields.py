"""Extend chunks for Stage 3 retrieval and deterministic identity (ADR-006, ADR-036).

Adds the ancestry, chunking-version, and embedding-version columns that Task 3.1
and Task 3.2 require, plus the uniqueness constraint that acts as the final
idempotency backstop behind the deterministic UUIDv5 chunk IDs.

Revision ID: 0002_stage3_chunk_fields
Revises: 0001_initial_schema
Create Date: 2026-08-17 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_stage3_chunk_fields"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Existing chunk rows (if any) predate deterministic IDs and carry no
    # document_id, so they cannot be migrated forward meaningfully. Chunks are
    # derived data rebuilt from PostgreSQL elements (ADR-002), so clearing them
    # is safe and cheaper than backfilling.
    op.execute("DELETE FROM chunks")

    op.add_column("chunks", sa.Column("document_id", sa.Uuid(), nullable=False))
    op.add_column(
        "chunks",
        sa.Column("chunk_type", sa.String(length=32), nullable=False, server_default="mixed"),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "chunking_version", sa.String(length=64), nullable=False, server_default="fixed-v1"
        ),
    )
    op.add_column(
        "chunks", sa.Column("section_path", sa.JSON(), nullable=False, server_default="[]")
    )
    op.add_column("chunks", sa.Column("page_span", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("chunks", sa.Column("embedding_version", sa.String(length=64), nullable=True))

    op.create_foreign_key(
        "fk_chunks_document_id",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index("ix_chunks_chunking_version", "chunks", ["chunking_version"])
    op.create_index("ix_chunks_primary_page_number", "chunks", ["primary_page_number"])
    op.create_index("ix_chunks_embedding_version", "chunks", ["embedding_version"])
    op.create_unique_constraint(
        "uq_chunks_version_chunking_index",
        "chunks",
        ["version_id", "chunking_version", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_chunks_version_chunking_index", "chunks", type_="unique")
    op.drop_index("ix_chunks_embedding_version", table_name="chunks")
    op.drop_index("ix_chunks_primary_page_number", table_name="chunks")
    op.drop_index("ix_chunks_chunking_version", table_name="chunks")
    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_constraint("fk_chunks_document_id", "chunks", type_="foreignkey")

    op.drop_column("chunks", "embedding_version")
    op.drop_column("chunks", "page_span")
    op.drop_column("chunks", "section_path")
    op.drop_column("chunks", "chunking_version")
    op.drop_column("chunks", "chunk_type")
    op.drop_column("chunks", "document_id")
