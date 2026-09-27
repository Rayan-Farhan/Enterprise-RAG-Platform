"""Chat router with SSE streaming and a non-streaming variant (Task 3.6).

Every answer is recorded (Task 13.3) and returned with its ``answer_id`` and the
evidence passages the model read, which is what the UI's evidence panel and
feedback form attach to.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.chat import (
    AnswerMetadataResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
)
from app.core.logging import get_logger
from app.db.session import get_db_session
from app.feedback.service import FeedbackService, evidence_of, get_feedback_service
from app.generation.service import (
    AnswerResult,
    GenerationService,
    answer_events,
    get_generation_service,
)

logger = get_logger("app.api.chat")
router = APIRouter(prefix="/chat", tags=["Chat"])


async def _record(
    feedback: FeedbackService, session: AsyncSession, result: AnswerResult
) -> uuid.UUID | None:
    """Record the answer; a recording failure costs the feedback link, not the answer."""
    try:
        return await feedback.record_answer(session, result)
    except Exception as exc:  # noqa: BLE001 - the user still gets their answer
        logger.exception("answer_record_failed", exc_info=exc)
        await session.rollback()
        result.degradations.append("answer_not_recorded")
        return None


def _to_response(result: AnswerResult, answer_id: uuid.UUID | None = None) -> ChatResponse:
    """Map the domain answer onto the API contract."""
    return ChatResponse(
        query=result.query,
        answer=result.answer,
        support=str(result.support),
        abstained=result.abstained,
        citations=[CitationResponse(**citation.model_dump()) for citation in result.citations],
        evidence=evidence_of(result),  # type: ignore[arg-type]
        answer_id=answer_id,
        metadata=AnswerMetadataResponse(
            provider=result.provider,
            model_name=result.model_name,
            model_version=result.model_version,
            prompt_versions=result.prompt_versions,
            prompt_hashes=result.prompt_hashes,
            retrieval_config=result.retrieval_config,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            token_counts=result.token_counts,
            retrieval_latency_ms=round(result.retrieval_latency_ms, 2),
            generation_latency_ms=round(result.generation_latency_ms, 2),
            total_latency_ms=round(result.total_latency_ms, 2),
            evidence_tokens=result.evidence_tokens,
            degradations=result.degradations,
        ),
    )


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "",
    response_model=ChatResponse,
    response_model_exclude_none=False,
    summary="Ask a question and receive a grounded, cited answer",
)
async def chat_completion(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session),
    service: GenerationService = Depends(get_generation_service),
    feedback: FeedbackService = Depends(get_feedback_service),
) -> ChatResponse | StreamingResponse:
    """Answer a question against the HR corpus.

    With ``stream=true`` the response is an SSE stream emitting `metadata`, then
    `token`, then `citations`, then `done`. The streaming and non-streaming paths
    run the identical pipeline and return the same content; `metadata` carries the
    ``answer_id`` and ``evidence``, and `done` repeats the ``answer_id``.
    """
    filters = request.filters.to_filters() if request.filters else None

    if not request.stream:
        result = await service.answer(
            query=request.query,
            session=session,
            top_k=request.top_k,
            filters=filters,
        )
        return _to_response(result, await _record(feedback, session, result))

    async def event_stream() -> AsyncIterator[str]:
        try:
            result = await service.answer(
                query=request.query,
                session=session,
                top_k=request.top_k,
                filters=filters,
            )
            answer_id = await _record(feedback, session, result)
            extra: dict[str, Any] = {
                "answer_id": str(answer_id) if answer_id else None,
                "evidence": evidence_of(result),
            }
            for event, payload in answer_events(result, extra):
                yield _sse(event, payload)
        except Exception as exc:  # noqa: BLE001 - stream must terminate cleanly
            # Headers are already sent, so an error cannot become an HTTP status.
            # It is delivered as a terminal event instead of a truncated stream.
            logger.exception("chat_stream_failed", exc_info=exc)
            yield _sse(
                "error",
                {
                    "message": "Answer generation failed before completion.",
                    "code": "GENERATION_ERROR",
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
