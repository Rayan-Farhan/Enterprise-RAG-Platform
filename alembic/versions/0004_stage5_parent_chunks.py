"""Add the parent-child chunk link for Stage 5 retrieval (Task 5.2, ADR-006).

The hierarchical strategy emits a section chunk plus the leaves that hang from
it. Retrieval matches a leaf and expands into its parent for generation context,
so the link has to survive persistence rather than living only in the candidate
list.

``ondelete="SET NULL"`` rather than CASCADE: deleting a section chunk should
orphan its leaves, not delete them. The leaves carry element IDs the golden
dataset points at, and cascading would silently remove evidence the evaluation
harness expects to find.

Revision ID: 0004_stage5_parent_chunks
Revises: 0003_stage4_experiment_tracking
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_stage5_parent_chunks"
down_revision = "0003_stage4_experiment_tracking"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add chunks.parent_chunk_id with its self-referential foreign key."""
    op.add_column(
        "chunks",
        sa.Column("parent_chunk_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        "ix_chunks_parent_chunk_id",
        "chunks",
        ["parent_chunk_id"],
    )
    op.create_foreign_key(
        "fk_chunks_parent_chunk_id",
        "chunks",
        "chunks",
        ["parent_chunk_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Drop the parent link. Existing chunks are unaffected."""
    op.drop_constraint("fk_chunks_parent_chunk_id", "chunks", type_="foreignkey")
    op.drop_index("ix_chunks_parent_chunk_id", table_name="chunks")
    op.drop_column("chunks", "parent_chunk_id")
