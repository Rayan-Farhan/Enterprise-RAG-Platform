"""Feedback and review-queue API contracts (Tasks 13.3–13.4)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.schemas import DatasetSplit, Difficulty, QuestionType


class EvidencePassage(BaseModel):
    """One passage the model was given, under the marker it saw."""

    marker: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    document_title: str | None = None
    version_number: int | None = None
    page_number: int
    page_span: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    text: str


class FeedbackCreate(BaseModel):
    """Structured feedback: a rating plus the specific judgements behind it.

    Each judgement is optional — ``null`` means "not assessed", which is not the
    same as ``false``.
    """

    answer_id: uuid.UUID
    helpful: bool
    answer_correct: bool | None = None
    answer_complete: bool | None = None
    citations_correct: bool | None = None
    source_authoritative: bool | None = None
    comment: str | None = Field(default=None, max_length=4000)


class AnswerSummary(BaseModel):
    """The answer a piece of feedback is about, as the reviewer needs to see it."""

    id: uuid.UUID
    query: str
    answer: str
    support: str
    abstained: bool
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidencePassage] = Field(default_factory=list)
    provider: str | None = None
    model_name: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    answer_id: uuid.UUID
    helpful: bool
    answer_correct: bool | None = None
    answer_complete: bool | None = None
    citations_correct: bool | None = None
    source_authoritative: bool | None = None
    comment: str | None = None
    status: str
    reviewer_note: str | None = None
    reviewed_at: datetime | None = None
    candidate_question_id: str | None = None
    created_at: datetime
    answer: AnswerSummary


class FeedbackListResponse(BaseModel):
    items: list[FeedbackResponse]
    total: int


class DismissRequest(BaseModel):
    reviewer_note: str | None = Field(default=None, max_length=4000)


class PromoteRequest(BaseModel):
    """What the reviewer decides when turning feedback into an evaluation case."""

    question_type: QuestionType
    acceptable_answer: str = Field(min_length=1, max_length=4000)
    evidence_markers: list[str] = Field(
        default_factory=list, description="Markers of the passages that answer the question"
    )
    must_abstain: bool = False
    must_contain: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    split: DatasetSplit = DatasetSplit.DEV
    question: str | None = Field(
        default=None, min_length=5, description="Reworded question; defaults to the user's"
    )
    reviewer_note: str | None = Field(default=None, max_length=4000)
