"""BM25 lexical retrieval (Task 6.1, ADR-007/008).

The second retrieval channel. It returns the same ``RetrievedChunk`` contract as
dense retrieval, rehydrated from PostgreSQL the same way, so Task 6.4 fusion can
combine the two without knowing which channel produced a hit.

BM25 scores are unbounded and only comparable within one query, so the dense
``RETRIEVAL_MIN_SCORE`` cosine floor does not apply here. The channel's
relevance cut is its rank: it returns the top K.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.hydration import hydrate_ranked
from app.retrieval.lexical_store import LexicalHit, OpenSearchLexicalStore, get_lexical_store
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk

logger = get_logger("app.retrieval.lexical")


class LexicalRetriever:
    """Runs a BM25 query and returns the top chunks with provenance attached."""

    channel = "bm25"

    def __init__(
        self,
        lexical_store: OpenSearchLexicalStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Explicit settings get a store built from them (an index name override
        # must reach the store), otherwise the process-wide singleton.
        self.lexical_store = lexical_store or (
            OpenSearchLexicalStore(settings=settings) if settings else get_lexical_store()
        )

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-K chunks for a query by BM25.

        ``min_score`` is accepted for interface parity with the dense channel and
        ignored: a BM25 score has no fixed scale to put a floor on.
        """
        started = time.perf_counter()
        k = top_k or self.settings.RETRIEVAL_TOP_K

        # The OpenSearch client is synchronous. A worker thread keeps the event
        # loop free, which is what lets fusion (Task 6.4) run channels in parallel.
        hits = await asyncio.to_thread(
            self.lexical_store.search,
            query=query,
            limit=k,
            filters=filters,
            chunking_version=self.settings.CHUNKING_VERSION,
        )

        chunks = (
            await hydrate_ranked(session, self._ranked(hits), self.channel)
            if session is not None
            else [self._from_payload(hit, rank) for rank, hit in enumerate(hits, start=1)]
        )

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            f"{self.channel}_retrieval_complete",
            query_chars=len(query),
            hits=len(chunks),
            top_k=k,
            latency_ms=round(latency_ms, 2),
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_candidates=len(hits),
            latency_ms=latency_ms,
            embedding_version=None,
            retrieval_config=self.config_snapshot(k, filters),
        )

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        """Chunks the filter admits under the current chunking version."""
        return await asyncio.to_thread(
            self.lexical_store.count,
            chunking_version=self.settings.CHUNKING_VERSION,
            filters=filters,
        )

    def config_snapshot(
        self,
        top_k: int,
        filters: RetrievalFilters | None,
    ) -> dict[str, object]:
        """The retrieval configuration recorded on every answer."""
        return {
            "channels": [self.channel],
            "top_k": top_k,
            "min_score": None,
            "chunking_version": self.settings.CHUNKING_VERSION,
            "chunk_size_tokens": self.settings.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": self.settings.CHUNK_OVERLAP_TOKENS,
            "lexical_index": self.settings.OPENSEARCH_INDEX_NAME,
            "reranking_enabled": self.settings.ENABLE_RERANKING,
            "filters": filters.model_dump(mode="json", exclude_defaults=True) if filters else {},
        }

    @staticmethod
    def _ranked(hits: list[LexicalHit]) -> list[tuple[uuid.UUID, float]]:
        ranked: list[tuple[uuid.UUID, float]] = []
        for hit in hits:
            try:
                ranked.append((uuid.UUID(hit.chunk_id), hit.score))
            except ValueError:
                logger.warning("index_hit_invalid_chunk_id", chunk_id=hit.chunk_id)
        return ranked

    def _from_payload(self, hit: LexicalHit, rank: int) -> RetrievedChunk:
        """Build a result straight from the index document (debug path only).

        The lexical index stores only what BM25 and filtering need, so element
        IDs, page spans and bounding boxes are absent here. Generation always
        passes a session and gets the full PostgreSQL record instead.
        """
        payload = hit.payload
        return RetrievedChunk(
            chunk_id=uuid.UUID(hit.chunk_id),
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
            document_title=payload.get("document_title"),
            metadata={
                "department": payload.get("department"),
                "policy_type": payload.get("policy_type"),
                "policy_status": payload.get("policy_status"),
            },
        )


_lexical_retriever: LexicalRetriever | None = None


def get_lexical_retriever() -> LexicalRetriever:
    """Return the singleton LexicalRetriever."""
    global _lexical_retriever
    if _lexical_retriever is None:
        _lexical_retriever = LexicalRetriever()
    return _lexical_retriever
