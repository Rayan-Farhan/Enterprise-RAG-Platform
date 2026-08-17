"""Dense vector retrieval (Task 3.3, ADR-007).

Stage 6 replaces this as the *only* channel by fusing it with BM25 and neural
sparse, but the ``RetrievedChunk`` contract it returns is what fusion consumes,
so that contract is deliberately channel-agnostic already.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.repositories.chunk_repo import ChunkRepository
from app.retrieval.embedding import EmbeddingService, get_embedding_service
from app.retrieval.schemas import (
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)
from app.retrieval.vector_store import QdrantVectorStore, VectorHit, get_vector_store

logger = get_logger("app.retrieval.dense")


class DenseRetriever:
    """Embeds a query and returns the nearest chunks with provenance attached."""

    channel = "dense"

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-K chunks for a query.

        When a ``session`` is supplied, hits are rehydrated from PostgreSQL so the
        returned content is authoritative (ADR-002). Without one, the vector
        payload is used directly — acceptable for debugging via ``/search``, but
        never for generation.
        """
        started = time.perf_counter()
        k = top_k or self.settings.RETRIEVAL_TOP_K
        threshold = self.settings.RETRIEVAL_MIN_SCORE if min_score is None else min_score

        query_vector = await self.embeddings.embed_query(query)

        hits = self.vector_store.search(
            query_vector=query_vector,
            limit=k,
            filters=filters,
            min_score=threshold,
            chunking_version=self.settings.CHUNKING_VERSION,
            embedding_version=self.settings.effective_embedding_version,
        )

        chunks = (
            await self._hydrate(session, hits)
            if session is not None
            else [self._from_payload(hit, rank) for rank, hit in enumerate(hits, start=1)]
        )

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "dense_retrieval_complete",
            query_chars=len(query),
            hits=len(chunks),
            top_k=k,
            min_score=threshold,
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_candidates=len(hits),
            latency_ms=latency_ms,
            embedding_version=self.settings.effective_embedding_version,
            retrieval_config=self.config_snapshot(k, threshold, filters),
        )

    def config_snapshot(
        self,
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None,
    ) -> dict[str, object]:
        """The retrieval configuration to record on every answer (Stage 3 exit gate)."""
        return {
            "channels": [self.channel],
            "top_k": top_k,
            "min_score": min_score,
            "chunking_version": self.settings.CHUNKING_VERSION,
            "chunk_size_tokens": self.settings.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": self.settings.CHUNK_OVERLAP_TOKENS,
            "embedding_version": self.settings.effective_embedding_version,
            "reranking_enabled": self.settings.ENABLE_RERANKING,
            "filters": filters.model_dump(mode="json", exclude_defaults=True) if filters else {},
        }

    async def _hydrate(
        self,
        session: AsyncSession,
        hits: list[VectorHit],
    ) -> list[RetrievedChunk]:
        """Replace vector payloads with authoritative PostgreSQL records."""
        if not hits:
            return []

        ordered_ids: list[uuid.UUID] = []
        scores: dict[uuid.UUID, float] = {}
        for hit in hits:
            raw_id = hit.payload.get("chunk_id")
            if not raw_id:
                continue
            try:
                chunk_id = uuid.UUID(str(raw_id))
            except ValueError:
                logger.warning("dense_hit_invalid_chunk_id", point_id=hit.point_id)
                continue
            ordered_ids.append(chunk_id)
            scores[chunk_id] = hit.score

        repo = ChunkRepository(session)
        found = {chunk.id: chunk for chunk in await repo.get_many_by_ids(ordered_ids)}

        # A hit with no row in PostgreSQL means the index is ahead of the database
        # — a real inconsistency worth logging rather than silently dropping.
        missing = [cid for cid in ordered_ids if cid not in found]
        if missing:
            logger.warning(
                "dense_hits_missing_in_postgres",
                count=len(missing),
                chunk_ids=[str(c) for c in missing[:10]],
            )

        results: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(
            (cid for cid in ordered_ids if cid in found), start=1
        ):
            results.append(
                self._from_model(found[chunk_id], scores[chunk_id], rank)
            )
        return results

    def _from_model(self, chunk: Chunk, score: float, rank: int) -> RetrievedChunk:
        version = chunk.version
        record = getattr(version, "metadata_record", None) if version else None

        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            content=chunk.content,
            score=score,
            channel=self.channel,
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
            },
        )

    def _from_payload(self, hit: VectorHit, rank: int) -> RetrievedChunk:
        """Build a result straight from the vector payload (debug path only)."""
        payload = hit.payload
        return RetrievedChunk(
            chunk_id=uuid.UUID(str(payload["chunk_id"])),
            document_id=uuid.UUID(str(payload["document_id"])),
            version_id=uuid.UUID(str(payload["version_id"])),
            content=str(payload.get("content", "")),
            score=hit.score,
            channel=self.channel,
            rank=rank,
            chunk_index=int(payload.get("chunk_index", 0)),
            chunk_type=str(payload.get("chunk_type", "mixed")),
            token_count=int(payload.get("token_count", 0)),
            page_number=int(payload.get("page_number", 1)),
            page_span=list(payload.get("page_span") or []),
            section_path=list(payload.get("section_path") or []),
            element_ids=list(payload.get("element_ids") or []),
            bounding_box=payload.get("bounding_box"),
            document_title=payload.get("document_title"),
            metadata={
                "department": payload.get("department"),
                "policy_type": payload.get("policy_type"),
                "policy_status": payload.get("policy_status"),
            },
        )


_dense_retriever: DenseRetriever | None = None


def get_dense_retriever() -> DenseRetriever:
    """Return the singleton DenseRetriever."""
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = DenseRetriever()
    return _dense_retriever
