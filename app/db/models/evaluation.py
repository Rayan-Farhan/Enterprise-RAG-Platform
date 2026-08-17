"""Experiment tracking tables (Task 4.4, ADR-029).

Results live in PostgreSQL rather than only in committed JSON because the
questions Stages 5-10 ask are cross-run queries — "which question types did the
reranker help?", "when did abstention accuracy start falling?" — and grepping a
directory of JSON files answers neither.

The committed files under ``evaluation/results/`` remain the durable, reviewable
record; the tables are the queryable index over them.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin


class ExperimentRun(Base, TimestampMixin):
    """One named pipeline configuration measured against one dataset split."""

    __tablename__ = "experiment_runs"
    __table_args__ = (Index("ix_experiment_runs_name_split", "name", "dataset_split"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset_split: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Every setting that could change the numbers, captured at run start",
    )
    embedding_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chunking_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    generator_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    generator_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    generator_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    judge_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    judge_model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    prompt_hashes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_by_type: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    system_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    question_results: Mapped[list["QuestionResult"]] = relationship(
        "QuestionResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    human_verdicts: Mapped[list["HumanReviewVerdict"]] = relationship(
        "HumanReviewVerdict",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ExperimentRun(name={self.name!r}, split={self.dataset_split}, n={self.dataset_size})>"


class QuestionResult(Base, TimestampMixin):
    """Per-question outcome within an experiment run.

    Stored per question, not only in aggregate, because "the average moved 3%" is
    never an actionable finding — the useful question is always *which* questions
    moved, and that cannot be recovered from a mean after the fact.
    """

    __tablename__ = "experiment_question_results"
    __table_args__ = (
        UniqueConstraint("run_id", "question_id", name="uq_question_results_run_question"),
        Index("ix_question_results_run_type", "run_id", "question_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    question_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)

    answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    rejected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    support: Mapped[str | None] = mapped_column(String(32), nullable=True)

    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    retrieved_element_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    context_element_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    citation_markers: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    cited_element_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)

    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    deterministic_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    judge_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    judge_raw_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    judge_stdev: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    judge_reasoning: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    judge_errors: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    retrieval_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    generation_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    token_counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_to_fail_until_stage: Mapped[int | None] = mapped_column(Integer, nullable=True)

    run: Mapped["ExperimentRun"] = relationship("ExperimentRun", back_populates="question_results")

    def __repr__(self) -> str:
        return f"<QuestionResult(run_id={self.run_id}, question_id={self.question_id!r})>"


class HumanReviewVerdict(Base, TimestampMixin):
    """Layer 3 verdict, stored alongside — never replacing — the automatic scores."""

    __tablename__ = "experiment_human_verdicts"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "question_id", "reviewer", name="uq_human_verdicts_run_question_reviewer"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_faithful: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    citations_are_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    abstention_was_right: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="", nullable=False)

    run: Mapped["ExperimentRun"] = relationship("ExperimentRun", back_populates="human_verdicts")

    def __repr__(self) -> str:
        return f"<HumanReviewVerdict(question_id={self.question_id!r}, reviewer={self.reviewer!r})>"
