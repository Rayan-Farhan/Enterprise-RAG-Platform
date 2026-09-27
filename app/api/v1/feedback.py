"""Feedback and review-queue router (Tasks 13.3–13.4, master §55)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.feedback import (
    AnswerSummary,
    DismissRequest,
    FeedbackCreate,
    FeedbackListResponse,
    FeedbackResponse,
    PromoteRequest,
)
from app.db.models.feedback import AnswerFeedback, AnswerRecord, FeedbackStatus
from app.db.session import get_db_session
from app.feedback.service import FeedbackService, get_feedback_service

router = APIRouter(tags=["Feedback"])


def answer_summary(record: AnswerRecord) -> AnswerSummary:
    return AnswerSummary.model_validate(record, from_attributes=True)


def _to_response(feedback: AnswerFeedback) -> FeedbackResponse:
    return FeedbackResponse(
        id=feedback.id,
        answer_id=feedback.answer_id,
        helpful=feedback.helpful,
        answer_correct=feedback.answer_correct,
        answer_complete=feedback.answer_complete,
        citations_correct=feedback.citations_correct,
        source_authoritative=feedback.source_authoritative,
        comment=feedback.comment,
        status=feedback.status,
        reviewer_note=feedback.reviewer_note,
        reviewed_at=feedback.reviewed_at,
        candidate_question_id=feedback.candidate_question_id,
        created_at=feedback.created_at,
        answer=answer_summary(feedback.answer),
    )


@router.get("/answers/{answer_id}", response_model=AnswerSummary, summary="A recorded answer")
async def get_answer(
    answer_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> AnswerSummary:
    return answer_summary(await service.get_answer(session, answer_id))


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Rate an answer",
)
async def submit_feedback(
    body: FeedbackCreate,
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    fields = body.model_dump(exclude={"answer_id"})
    if fields["comment"] is not None:
        fields["comment"] = fields["comment"].strip() or None
    return _to_response(await service.submit(session, body.answer_id, **fields))


@router.get("/feedback", response_model=FeedbackListResponse, summary="The review queue")
async def list_feedback(
    status_filter: FeedbackStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackListResponse:
    items, total = await service.list_feedback(session, status_filter, limit, offset)
    return FeedbackListResponse(items=[_to_response(f) for f in items], total=total)


@router.post(
    "/feedback/{feedback_id}/dismiss",
    response_model=FeedbackResponse,
    summary="Close feedback without creating an evaluation case",
)
async def dismiss_feedback(
    feedback_id: uuid.UUID,
    body: DismissRequest,
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    return _to_response(await service.dismiss(session, feedback_id, body.reviewer_note))


@router.post(
    "/feedback/{feedback_id}/promote",
    response_model=FeedbackResponse,
    summary="Promote feedback into a golden-dataset candidate",
)
async def promote_feedback(
    feedback_id: uuid.UUID,
    body: PromoteRequest,
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    return _to_response(await service.promote(session, feedback_id, **body.model_dump()))


@router.post(
    "/feedback/{feedback_id}/reopen",
    response_model=FeedbackResponse,
    summary="Return reviewed feedback to the queue, withdrawing a pending candidate",
)
async def reopen_feedback(
    feedback_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    return _to_response(await service.reopen(session, feedback_id))
