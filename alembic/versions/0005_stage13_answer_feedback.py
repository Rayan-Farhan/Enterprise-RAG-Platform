"""Answer records and structured feedback for the Stage 13 feedback loop (ADR-050).

Every answer served through the API is persisted with its evidence set and the
configuration that produced it, and feedback points at that record. Deleting an
answer record deletes its feedback (CASCADE): feedback about an answer nobody can
reconstruct is not reviewable.

Revision ID: 0005_stage13_answer_feedback
Revises: 0004_stage5_parent_chunks
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005_stage13_answer_feedback"
down_revision = "0004_stage5_parent_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "answer_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("support", sa.String(16), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("rejected", sa.Boolean(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("retrieval_config", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(32), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("prompt_versions", sa.JSON(), nullable=False),
        sa.Column("degradations", sa.JSON(), nullable=False),
        sa.Column("total_latency_ms", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "answer_feedback",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "answer_id",
            sa.Uuid(),
            sa.ForeignKey("answer_records.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("helpful", sa.Boolean(), nullable=False),
        sa.Column("answer_correct", sa.Boolean(), nullable=True),
        sa.Column("answer_complete", sa.Boolean(), nullable=True),
        sa.Column("citations_correct", sa.Boolean(), nullable=True),
        sa.Column("source_authoritative", sa.Boolean(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("candidate_question_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_answer_feedback_answer_id", "answer_feedback", ["answer_id"])
    op.create_index("ix_answer_feedback_status", "answer_feedback", ["status"])
    op.create_index(
        "ix_answer_feedback_status_created", "answer_feedback", ["status", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_answer_feedback_status_created", table_name="answer_feedback")
    op.drop_index("ix_answer_feedback_status", table_name="answer_feedback")
    op.drop_index("ix_answer_feedback_answer_id", table_name="answer_feedback")
    op.drop_table("answer_feedback")
    op.drop_table("answer_records")
