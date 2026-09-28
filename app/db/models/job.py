"""Durable job records for the async ingestion plane (Task 7.2, master §30).

Celery moves messages; this table is the record of work. One row is one task
execution request - a pipeline step for one document version, a cleanup sweep,
an evaluation run - and it outlives the message that carried it. The API reads
job state from here, never from the broker.
"""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, TimestampMixin


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


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
    __table_args__ = (Index("ix_jobs_document_created", "document_id", "created_at"),)

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
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Set when a running job is asked to stop; it stops at its next checkpoint",
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.task_type}, status={self.status})>"
