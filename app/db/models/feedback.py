"""Answer records and structured user feedback (Stage 13, ADR-050, master §55).

Feedback is only useful if the answer it judges can be reconstructed: which
question, which evidence the model read, which prompt and model produced it,
and what the retriever did. So every answer served through the API is persisted
as an ``AnswerRecord`` *before* anyone rates it, and feedback points at that
record rather than carrying a copy of the answer.

Stage 12 (observability) will extend this into full request tracing; the record
here holds what the feedback loop needs today — the evidence set and the
configuration that produced it.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin


class FeedbackStatus(StrEnum):
    """Where a piece of feedback is in the review queue (Task 13.4)."""

    NEW = "new"
    PROMOTED = "promoted"  # turned into a golden-dataset candidate by a reviewer
    DISMISSED = "dismissed"  # reviewed; no evaluation case warranted


class AnswerRecord(Base, TimestampMixin):
    """One answer as served, with everything needed to reconstruct it."""

    __tablename__ = "answer_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    support: Mapped[str] = mapped_column(String(16), nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    rejected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False, doc="Citations returned to the user"
    )
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="Every chunk the model read: marker, ids, title, page, section, text",
    )
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_versions: Mapped[dict[str, str]] = mapped_column(JSON, default=dict, nullable=False)
    degradations: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    feedback: Mapped[list["AnswerFeedback"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan", passive_deletes=True
    )


class AnswerFeedback(Base, TimestampMixin):
    """Structured feedback on one answer: more than a thumbs up or down."""

    __tablename__ = "answer_feedback"
    __table_args__ = (Index("ix_answer_feedback_status_created", "status", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    answer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("answer_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False, doc="Thumbs up (true) / down")
    # None = the user did not say. Distinct from False, which is a judgement.
    answer_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    answer_complete: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    citations_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    source_authoritative: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True, doc="What went wrong?")

    status: Mapped[str] = mapped_column(
        String(16), default=FeedbackStatus.NEW.value, nullable=False, index=True
    )
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    candidate_question_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, doc="Golden-dataset candidate created on promotion"
    )

    answer: Mapped[AnswerRecord] = relationship(back_populates="feedback")
