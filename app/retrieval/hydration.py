"""Rehydrate ranked hits from PostgreSQL (ADR-002).

Every retrieval channel ranks against a derived index — Qdrant for dense,
OpenSearch for BM25 — and both are rebuilt from PostgreSQL, never the other way
round. So every channel turns its ranked chunk IDs into results the same way:
by reading the authoritative rows, in rank order, with the channel's own score.
One implementation keeps the channels' ``RetrievedChunk`` output identical in
shape, which is what Stage 6 fusion (Task 6.4) relies on.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.repositories.chunk_repo import ChunkRepository
from app.retrieval.schemas import RetrievedChunk

logger = get_logger("app.retrieval.hydration")


async def hydrate_ranked(
    session: AsyncSession,
    ranked: list[tuple[uuid.UUID, float]],
    channel: str,
) -> list[RetrievedChunk]:
    """Replace index hits with PostgreSQL records, preserving rank order.

    ``ranked`` is ``(chunk_id, score)`` best first. A hit with no row in
    PostgreSQL means the index is ahead of the database — a real inconsistency,
    so it is logged rather than silently dropped.
    """
    if not ranked:
        return []

    scores = dict(ranked)
    ordered_ids = [chunk_id for chunk_id, _ in ranked]
    found = {
        chunk.id: chunk for chunk in await ChunkRepository(session).get_many_by_ids(ordered_ids)
    }

    missing = [cid for cid in ordered_ids if cid not in found]
    if missing:
        logger.warning(
            "index_hits_missing_in_postgres",
            channel=channel,
            count=len(missing),
            chunk_ids=[str(c) for c in missing[:10]],
        )

    present = (cid for cid in ordered_ids if cid in found)
    return [
        chunk_to_retrieved(found[chunk_id], scores[chunk_id], rank, channel)
        for rank, chunk_id in enumerate(present, start=1)
    ]


def chunk_to_retrieved(chunk: Chunk, score: float, rank: int, channel: str) -> RetrievedChunk:
    """Build the channel-agnostic retrieval result from a chunk row."""
    version = chunk.version
    record = getattr(version, "metadata_record", None) if version else None

    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        version_id=chunk.version_id,
        content=chunk.content,
        score=score,
        channel=channel,
        rank=rank,
        chunk_index=chunk.chunk_index,
        chunk_type=chunk.chunk_type,
        token_count=chunk.token_count,
        page_number=chunk.primary_page_number,
        page_span=list(chunk.page_span or []),
        section_path=list(chunk.section_path or []),
        element_ids=list(chunk.element_ids or []),
        bounding_box=chunk.bounding_box,
        document_title=chunk.document.title if chunk.document else None,
        version_number=version.version_number if version else None,
        metadata={
            "department": getattr(record, "department", None),
            "policy_type": getattr(record, "policy_type", None),
            "policy_status": getattr(record, "policy_status", None),
            "effective_from": (
                version.effective_from.isoformat()
                if version is not None and version.effective_from is not None
                else None
            ),
            # Task 5.2 reads this to expand a matched leaf into its section.
            # Carried as metadata rather than a typed field so the retrieval
            # contract does not assume a hierarchy that only one chunking
            # strategy produces.
            "parent_chunk_id": (str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None),
        },
    )
