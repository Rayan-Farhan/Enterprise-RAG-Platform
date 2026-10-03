"""Replayable jobs: task name, arguments, failure kind, replay lineage (Task 7.5).

Revision ID: 0008_stage7_dead_letters
Revises: 0007_stage7_idempotency
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008_stage7_dead_letters"
down_revision = "0007_stage7_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("task_name", sa.String(128), nullable=True))
    op.add_column("jobs", sa.Column("payload", sa.JSON(), nullable=True))
    op.add_column("jobs", sa.Column("failure_kind", sa.String(32), nullable=True))
    op.add_column(
        "jobs",
        sa.Column(
            "replay_of_id",
            sa.Uuid(),
            sa.ForeignKey("jobs.id", ondelete="SET NULL", name="fk_jobs_replay_of_id"),
            nullable=True,
        ),
    )
    op.create_index("ix_jobs_replay_of_id", "jobs", ["replay_of_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_replay_of_id", table_name="jobs")
    op.drop_constraint("fk_jobs_replay_of_id", "jobs", type_="foreignkey")
    op.drop_column("jobs", "replay_of_id")
    op.drop_column("jobs", "failure_kind")
    op.drop_column("jobs", "payload")
    op.drop_column("jobs", "task_name")
