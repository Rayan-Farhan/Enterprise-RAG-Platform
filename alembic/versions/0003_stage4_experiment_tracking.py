"""Add Stage 4 experiment tracking tables (ADR-029, Task 4.4).

Three tables: the run (configuration snapshot plus aggregate metrics), the
per-question result, and the Layer 3 human verdict. Human verdicts are a
separate table rather than columns on the result row precisely so that a review
never overwrites an automatic score — the comparison between the two is the
point of Layer 3.

Revision ID: 0003_stage4_experiment_tracking
Revises: 0002_stage3_chunk_fields
Create Date: 2026-08-18 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_stage4_experiment_tracking"
down_revision: str | None = "0002_stage3_chunk_fields"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dataset_split", sa.String(length=16), nullable=False),
        sa.Column("dataset_version", sa.String(length=32), nullable=False),
        sa.Column("dataset_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("git_commit", sa.String(length=64), nullable=True),
        sa.Column("config_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("embedding_version", sa.String(length=64), nullable=True),
        sa.Column("chunking_version", sa.String(length=64), nullable=True),
        sa.Column("generator_provider", sa.String(length=32), nullable=True),
        sa.Column("generator_model", sa.String(length=128), nullable=True),
        sa.Column("generator_model_version", sa.String(length=128), nullable=True),
        sa.Column("judge_provider", sa.String(length=32), nullable=True),
        sa.Column("judge_model", sa.String(length=128), nullable=True),
        sa.Column("judge_model_version", sa.String(length=128), nullable=True),
        sa.Column("prompt_versions", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("prompt_hashes", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("metrics_by_type", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("system_metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiment_runs_name", "experiment_runs", ["name"])
    op.create_index("ix_experiment_runs_dataset_split", "experiment_runs", ["dataset_split"])
    op.create_index("ix_experiment_runs_git_commit", "experiment_runs", ["git_commit"])
    op.create_index("ix_experiment_runs_embedding_version", "experiment_runs", ["embedding_version"])
    op.create_index("ix_experiment_runs_chunking_version", "experiment_runs", ["chunking_version"])
    op.create_index("ix_experiment_runs_name_split", "experiment_runs", ["name", "dataset_split"])

    op.create_table(
        "experiment_question_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=128), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(length=32), nullable=False),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False, server_default=""),
        sa.Column("abstained", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rejected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("support", sa.String(length=32), nullable=True),
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("retrieved_element_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("context_element_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("citation_markers", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("cited_element_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("retrieval_metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("deterministic_metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("judge_metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("judge_raw_scores", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("judge_stdev", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("judge_reasoning", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("judge_errors", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("generation_latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("token_counts", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("expected_to_fail_until_stage", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["experiment_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("run_id", "question_id", name="uq_question_results_run_question"),
    )
    op.create_index("ix_question_results_run_id", "experiment_question_results", ["run_id"])
    op.create_index(
        "ix_question_results_question_id", "experiment_question_results", ["question_id"]
    )
    op.create_index(
        "ix_question_results_question_type", "experiment_question_results", ["question_type"]
    )
    op.create_index(
        "ix_question_results_run_type", "experiment_question_results", ["run_id", "question_type"]
    )

    op.create_table(
        "experiment_human_verdicts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=128), nullable=False),
        sa.Column("reviewer", sa.String(length=128), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("is_faithful", sa.Boolean(), nullable=True),
        sa.Column("citations_are_correct", sa.Boolean(), nullable=True),
        sa.Column("abstention_was_right", sa.Boolean(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=True),
        sa.Column("comment", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["run_id"], ["experiment_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "run_id", "question_id", "reviewer", name="uq_human_verdicts_run_question_reviewer"
        ),
    )
    op.create_index("ix_human_verdicts_run_id", "experiment_human_verdicts", ["run_id"])
    op.create_index(
        "ix_human_verdicts_question_id", "experiment_human_verdicts", ["question_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_human_verdicts_question_id", table_name="experiment_human_verdicts")
    op.drop_index("ix_human_verdicts_run_id", table_name="experiment_human_verdicts")
    op.drop_table("experiment_human_verdicts")

    op.drop_index("ix_question_results_run_type", table_name="experiment_question_results")
    op.drop_index("ix_question_results_question_type", table_name="experiment_question_results")
    op.drop_index("ix_question_results_question_id", table_name="experiment_question_results")
    op.drop_index("ix_question_results_run_id", table_name="experiment_question_results")
    op.drop_table("experiment_question_results")

    op.drop_index("ix_experiment_runs_name_split", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_chunking_version", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_embedding_version", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_git_commit", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_dataset_split", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_name", table_name="experiment_runs")
    op.drop_table("experiment_runs")
