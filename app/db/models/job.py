"""Durable job records for the async ingestion plane (Task 7.2, master §30).

Celery moves messages; this table is the record of work. One row is one task
execution request - a pipeline step for one document version, a cleanup sweep,
an evaluation run - and it outlives the message that carried it. The API reads
job state from here, never from the broker.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, TimestampMixin


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})
ACTIVE_JOB_PREDICATE = "status IN ('queued', 'running')"


class FailureKind(StrEnum):
    """Why a job ended up dead-lettered (Task 7.5)."""

    PERMANENT = "permanent"  # an error retrying cannot fix
    RETRIES_EXHAUSTED = "retries_exhausted"  # transient, but it kept happening
    DELIVERY_LIMIT = "delivery_limit"  # a poison message: delivered too often to trust
    UNPUBLISHED = "unpublished"  # never reached the broker


class JobType(StrEnum):
    """The locked ingestion chain (Task 7.3, ADR-018), plus diagnostics."""

    PARSE_DOCUMENT = "parse_document"
    EXTRACT_PAGES = "extract_pages"
    OCR_PAGES = "ocr_pages"
    NORMALIZE_DOCUMENT = "normalize_document"
    CHUNK_DOCUMENT = "chunk_document"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    INDEX_OPENSEARCH = "index_opensearch"
    INDEX_QDRANT = "index_qdrant"
    VALIDATE_INDEX = "validate_index"
    PUBLISH_VERSION = "publish_version"
    DIAGNOSTIC = "diagnostic"


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_document_created", "document_id", "created_at"),
        # One live job per step of a version (master §31 "unique job
        # constraints"): a retried request or a replay racing the chain cannot
        # start a second copy of work already queued or running.
        Index(
            "uq_jobs_active_step",
            "version_id",
            "task_type",
            unique=True,
            postgresql_where=text(ACTIVE_JOB_PREDICATE),
            sqlite_where=text(ACTIVE_JOB_PREDICATE),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Nullable: cleanup sweeps and evaluation runs are jobs with no document.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True
    )
    version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Null until parsing has created the version this job belongs to",
    )
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    queue: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=JobStatus.QUEUED.value, nullable=False, index=True
    )
    attempt: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Starts so far; above 1 means a retry or a redelivery after a worker was lost",
    )
    worker: Mapped[str | None] = mapped_column(String(255), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, doc="0.0 to 1.0")
    progress_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_name: Mapped[str | None] = mapped_column(
        String(128), nullable=True, doc="The Celery task that runs this job; what a replay re-sends"
    )
    payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, doc="The task's arguments, kept so a dead letter can be replayed"
    )
    failure_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    replay_of_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="The dead-lettered job this one replays",
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Set when a running job is asked to stop; it stops at its next checkpoint",
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.task_type}, status={self.status})>"
