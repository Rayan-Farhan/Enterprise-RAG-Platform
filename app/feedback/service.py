"""Answer persistence and the feedback review loop (Tasks 13.3–13.4, master §55).

The loop is: every served answer is recorded with the evidence the model read;
a user rates it; a reviewer works the queue and either dismisses the feedback or
promotes it into a golden-dataset candidate (:mod:`app.evaluation.candidates`),
which a maintainer then accepts into a split.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ConflictException, NotFoundException
from app.core.logging import get_logger
from app.db.models.feedback import AnswerFeedback, AnswerRecord, FeedbackStatus
from app.evaluation.candidates import CANDIDATES_PATH, add_candidate, remove_candidate
from app.evaluation.dataset import DatasetError
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)
from app.generation.service import AnswerResult
from app.ingestion.chunking.provenance import strip_prefix

logger = get_logger("app.feedback")


def evidence_of(result: AnswerResult) -> list[dict[str, Any]]:
    """The passages the model was given, keyed by the marker it saw.

    ``context_chunk_ids`` is the subset of ``context_chunks`` that fit the token
    budget, in marker order, so the n-th included chunk is evidence ``[n]``.
    """
    included = set(result.context_chunk_ids)
    passages = [c for c in result.context_chunks if c.chunk_id in included]
    return [
        {
            "marker": str(index),
            "chunk_id": str(chunk.chunk_id),
            "document_id": str(chunk.document_id),
            "version_id": str(chunk.version_id),
            "document_title": chunk.document_title,
            "version_number": chunk.version_number,
            "page_number": chunk.page_number,
            "page_span": chunk.page_span,
            "section_path": chunk.section_path,
            "element_ids": chunk.element_ids,
            "text": strip_prefix(chunk.content),
        }
        for index, chunk in enumerate(passages, start=1)
    ]


class FeedbackService:
    """Records answers and moves feedback through the review queue."""

    def __init__(self, candidates_path: Path | None = None) -> None:
        self.candidates_path = candidates_path or CANDIDATES_PATH

    # -- answers ----------------------------------------------------------

    async def record_answer(self, session: AsyncSession, result: AnswerResult) -> uuid.UUID:
        """Persist a served answer and commit, so feedback can reference it at once.

        The commit is explicit: on the streaming path the client receives the
        ``answer_id`` before the request-scoped session would otherwise commit,
        and a rating sent in that window would find no answer to attach to.
        """
        record = AnswerRecord(
            query=result.query,
            answer=result.answer,
            support=str(result.support),
            abstained=result.abstained,
            rejected=result.rejected,
            citations=[c.model_dump(mode="json") for c in result.citations],
            evidence=evidence_of(result),
            retrieval_config=result.retrieval_config,
            provider=result.provider,
            model_name=result.model_name,
            prompt_versions=result.prompt_versions,
            degradations=result.degradations,
            total_latency_ms=round(result.total_latency_ms, 2),
        )
        session.add(record)
        await session.commit()
        return record.id

    async def get_answer(self, session: AsyncSession, answer_id: uuid.UUID) -> AnswerRecord:
        record = await session.get(AnswerRecord, answer_id)
        if record is None:
            raise NotFoundException(f"Answer {answer_id} not found")
        return record

    # -- feedback ---------------------------------------------------------

    async def submit(
        self, session: AsyncSession, answer_id: uuid.UUID, **fields: Any
    ) -> AnswerFeedback:
        await self.get_answer(session, answer_id)
        feedback = AnswerFeedback(answer_id=answer_id, **fields)
        session.add(feedback)
        await session.commit()
        logger.info("feedback_submitted", answer_id=str(answer_id), helpful=feedback.helpful)
        return await self.get_feedback(session, feedback.id)

    async def get_feedback(self, session: AsyncSession, feedback_id: uuid.UUID) -> AnswerFeedback:
        stmt = (
            select(AnswerFeedback)
            .options(selectinload(AnswerFeedback.answer))
            .where(AnswerFeedback.id == feedback_id)
            .execution_options(populate_existing=True)
        )
        feedback = (await session.execute(stmt)).scalar_one_or_none()
        if feedback is None:
            raise NotFoundException(f"Feedback {feedback_id} not found")
        return feedback

    async def list_feedback(
        self,
        session: AsyncSession,
        status: FeedbackStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AnswerFeedback], int]:
        """The review queue, oldest first so nothing starves at the bottom."""
        stmt = select(AnswerFeedback).options(selectinload(AnswerFeedback.answer))
        count = select(func.count()).select_from(AnswerFeedback)
        if status is not None:
            stmt = stmt.where(AnswerFeedback.status == status.value)
            count = count.where(AnswerFeedback.status == status.value)
        stmt = stmt.order_by(AnswerFeedback.created_at.asc()).limit(limit).offset(offset)
        items = list((await session.execute(stmt)).scalars().all())
        total = (await session.execute(count)).scalar_one()
        return items, total

    async def dismiss(
        self, session: AsyncSession, feedback_id: uuid.UUID, reviewer_note: str | None
    ) -> AnswerFeedback:
        feedback = await self._open_for_review(session, feedback_id)
        feedback.status = FeedbackStatus.DISMISSED.value
        feedback.reviewer_note = reviewer_note
        feedback.reviewed_at = datetime.now(UTC)
        await session.commit()
        return await self.get_feedback(session, feedback_id)

    async def reopen(self, session: AsyncSession, feedback_id: uuid.UUID) -> AnswerFeedback:
        """Undo a review. A promoted candidate is withdrawn if still pending."""
        feedback = await self.get_feedback(session, feedback_id)
        if feedback.candidate_question_id:
            remove_candidate(feedback.candidate_question_id, self.candidates_path)
        feedback.status = FeedbackStatus.NEW.value
        feedback.candidate_question_id = None
        feedback.reviewed_at = None
        await session.commit()
        return await self.get_feedback(session, feedback_id)

    async def promote(
        self,
        session: AsyncSession,
        feedback_id: uuid.UUID,
        *,
        question_type: QuestionType,
        acceptable_answer: str,
        evidence_markers: list[str],
        must_abstain: bool = False,
        must_contain: list[str] | None = None,
        difficulty: Difficulty = Difficulty.MEDIUM,
        split: DatasetSplit = DatasetSplit.DEV,
        question: str | None = None,
        reviewer_note: str | None = None,
    ) -> AnswerFeedback:
        """Turn reviewed feedback into a golden-dataset candidate.

        The reviewer supplies what the model got wrong — the reference answer and
        which of the passages it read actually answer the question. Evidence is
        recorded at element granularity, like the rest of the dataset (ADR-036),
        so the candidate survives a re-chunk.
        """
        feedback = await self._open_for_review(session, feedback_id)
        answer = feedback.answer

        by_marker = {e["marker"]: e for e in answer.evidence}
        unknown = [m for m in evidence_markers if m not in by_marker]
        if unknown:
            raise DatasetError(
                message=f"Evidence markers not in this answer: {', '.join(unknown)}",
                details={"markers": unknown, "available": sorted(by_marker)},
            )
        expected = [
            ExpectedEvidence(
                document_id=uuid.UUID(e["document_id"]),
                version_id=uuid.UUID(e["version_id"]),
                element_ids=e["element_ids"],
                page_numbers=e.get("page_span") or [e["page_number"]],
                section_path=e.get("section_path") or [],
                document_title=e.get("document_title"),
            )
            for e in (by_marker[m] for m in evidence_markers)
        ]

        question_id = f"{split.value}-feedback-{feedback.id.hex[:8]}"
        try:
            candidate = GoldenQuestion(
                question_id=question_id,
                question=question or answer.query,
                question_type=question_type,
                difficulty=difficulty,
                split=split,
                expected_evidence=expected,
                acceptable_answer=acceptable_answer,
                required_citations=0 if must_abstain else min(len(expected), 1),
                must_contain=must_contain or [],
                must_abstain=must_abstain,
                source="feedback",
                notes=_provenance_note(feedback, reviewer_note),
            )
        except ValidationError as exc:
            raise DatasetError(
                message=f"Candidate is not a valid golden question: {exc.errors()[0]['msg']}",
                details={"question_id": question_id},
            ) from exc

        add_candidate(candidate, self.candidates_path)

        feedback.status = FeedbackStatus.PROMOTED.value
        feedback.reviewer_note = reviewer_note
        feedback.reviewed_at = datetime.now(UTC)
        feedback.candidate_question_id = question_id
        await session.commit()
        logger.info("feedback_promoted", feedback_id=str(feedback_id), question_id=question_id)
        return await self.get_feedback(session, feedback_id)

    async def _open_for_review(
        self, session: AsyncSession, feedback_id: uuid.UUID
    ) -> AnswerFeedback:
        feedback = await self.get_feedback(session, feedback_id)
        if feedback.status != FeedbackStatus.NEW.value:
            raise ConflictException(
                f"Feedback {feedback_id} was already reviewed ({feedback.status}); reopen it first"
            )
        return feedback


def _provenance_note(feedback: AnswerFeedback, reviewer_note: str | None) -> str:
    """Where the candidate came from, so a bad question can be traced to its answer."""
    parts = [f"From feedback {feedback.id} on answer {feedback.answer_id}."]
    if feedback.comment:
        parts.append(f"User: {feedback.comment.strip()}")
    if reviewer_note:
        parts.append(f"Reviewer: {reviewer_note.strip()}")
    return " ".join(parts)


def get_feedback_service() -> FeedbackService:
    """FastAPI dependency; overridden in tests to redirect the candidates file."""
    return FeedbackService()
