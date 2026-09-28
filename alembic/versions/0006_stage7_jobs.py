"""Durable job records for the async ingestion plane (Task 7.2, master §30).

Deleting a document or version deletes its jobs (CASCADE): a job's history is
only meaningful while the thing it processed exists.

Revision ID: 0006_stage7_jobs
Revises: 0005_stage13_answer_feedback
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0006_stage7_jobs"
down_revision = "0005_stage13_answer_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "version_id",
            sa.Uuid(),
            sa.ForeignKey("document_versions.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("queue", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("worker", sa.String(255), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("progress_message", sa.String(255), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_document_created", "jobs", ["document_id", "created_at"])
    op.create_index("ix_jobs_version_id", "jobs", ["version_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_version_id", table_name="jobs")
    op.drop_index("ix_jobs_document_created", table_name="jobs")
    op.drop_table("jobs")
