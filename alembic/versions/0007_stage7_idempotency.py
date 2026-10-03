"""Idempotency backstops: canonical uniqueness, one live job per step, idempotency keys (Task 7.4).

Revision ID: 0007_stage7_idempotency
Revises: 0006_stage7_jobs
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0007_stage7_idempotency"
down_revision = "0006_stage7_jobs"
branch_labels = None
depends_on = None

ACTIVE_JOB_PREDICATE = "status IN ('queued', 'running')"


def upgrade() -> None:
    op.create_unique_constraint("uq_pages_version_page", "pages", ["version_id", "page_number"])
    op.create_unique_constraint(
        "uq_elements_version_element", "elements", ["version_id", "element_id"]
    )
    op.create_index(
        "uq_jobs_active_step",
        "jobs",
        ["version_id", "task_type"],
        unique=True,
        postgresql_where=sa.text(ACTIVE_JOB_PREDICATE),
        sqlite_where=sa.text(ACTIVE_JOB_PREDICATE),
    )
    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("operation", "key", name="uq_idempotency_records_operation_key"),
    )


def downgrade() -> None:
    op.drop_table("idempotency_records")
    op.drop_index("uq_jobs_active_step", table_name="jobs")
    op.drop_constraint("uq_elements_version_element", "elements", type_="unique")
    op.drop_constraint("uq_pages_version_page", "pages", type_="unique")
