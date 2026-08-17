"""Search router exposing raw retrieval for debugging and evaluation (Task 3.6)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.chat import (
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
)
from app.core.logging import get_logger
from app.db.session import get_db_session
from app.retrieval.dense import DenseRetriever, get_dense_retriever

logger = get_logger("app.api.search")
router = APIRouter(prefix="/search", tags=["Search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Execute raw retrieval without generation",
)
async def execute_search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db_session),
    retriever: DenseRetriever = Depends(get_dense_retriever),
) -> SearchResponse:
    """Return retrieved chunks with scores and provenance, bypassing generation.

    This is the introspection surface Stage 4's evaluation harness measures
    retrieval quality through, independent of the generator.
    """
    result = await retriever.retrieve(
        query=request.query,
        session=session,
        top_k=request.top_k,
        filters=request.filters.to_filters() if request.filters else None,
        min_score=request.min_score,
    )

    return SearchResponse(
        query=result.query,
        hits=[
            SearchHitResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                document_title=chunk.document_title,
                score=chunk.score,
                rank=chunk.rank,
                channel=chunk.channel,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
                token_count=chunk.token_count,
                page_number=chunk.page_number,
                page_span=chunk.page_span,
                section_path=chunk.section_path,
                element_ids=chunk.element_ids,
                bounding_box=chunk.bounding_box,
                content=chunk.content,
            )
            for chunk in result.chunks
        ],
        total_candidates=result.total_candidates,
        latency_ms=round(result.latency_ms, 2),
        embedding_version=result.embedding_version,
        retrieval_config=result.retrieval_config,
    )
