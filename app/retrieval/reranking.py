"""Cross-encoder reranking over the retrieved pool (Task 6.5, ADR-011).

Retrieval channels score a query and a chunk independently (vectors, term
weights); a reranker reads them together, which is slower but orders far better.
So the retriever casts a wide net — ``RERANK_CANDIDATES`` chunks from whichever
channel or fusion is configured — and the reranker picks the final top K.

It goes through the model gateway (ADR-046): the Jina reranker under the hosted
profile, TEI-served BGE/Qwen rerankers under the local profile. Nothing here
knows which.

Master §16 warns against assuming a reranker is always worth its cost, so the
trace records its latency and how far it moved each chunk, and it sits behind
``ENABLE_RERANKING``. A reranker failure (quota, outage) degrades to the
retriever's own order rather than failing retrieval.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.schemas import RetrievalFilters, RetrievalResult

if TYPE_CHECKING:
    from app.models.gateway import ModelGateway
    from app.retrieval.channels import Retriever

logger = get_logger("app.retrieval.reranking")


class RerankingRetriever:
    """Wraps any retriever: widen the pool, rerank it, keep the top K."""

    def __init__(
        self,
        inner: Retriever,
        gateway: ModelGateway | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.inner = inner
        self.settings = settings or inner.settings or get_settings()
        if gateway is None:
            from app.models.gateway import build_model_gateway, get_model_gateway

            gateway = (
                build_model_gateway(self.settings) if settings is not None else get_model_gateway()
            )
        self.gateway = gateway

    @property
    def channel(self) -> str:
        return self.inner.channel

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        return await self.inner.candidate_count(filters)

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        k = top_k or self.settings.RETRIEVAL_TOP_K
        pool = max(self.settings.RERANK_CANDIDATES, k)
        result = await self.inner.retrieve(query, session, pool, filters, min_score)
        candidates = result.chunks

        trace: dict[str, Any] = {
            "model": None,
            "candidates": len(candidates),
            "top_k": k,
            "latency_ms": 0.0,
            "status": "skipped",
            "moves": [],
        }
        if len(candidates) <= 1:
            return self._finish(result, candidates[:k], trace)

        started = time.perf_counter()
        try:
            reranked = await self.gateway.rerank(
                query=query,
                documents=[chunk.content for chunk in candidates],
                top_k=k,
            )
        except Exception as exc:  # noqa: BLE001 - the retriever's own order still stands
            trace.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            logger.warning("rerank_failed_keeping_retrieval_order", error=str(exc))
            return self._finish(result, candidates[:k], trace)

        ordered = []
        for new_rank, scored in enumerate(reranked.results[:k], start=1):
            chunk = candidates[scored.index]
            ordered.append(
                chunk.model_copy(
                    update={
                        "score": scored.score,
                        "rank": new_rank,
                        "metadata": {
                            **chunk.metadata,
                            "retrieval_rank": chunk.rank,
                            "retrieval_score": chunk.score,
                        },
                    }
                )
            )
            trace["moves"].append([chunk.rank, new_rank])

        trace.update(
            model=reranked.metadata.model_name,
            status="ok",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        logger.info(
            "rerank_complete",
            model=trace["model"],
            candidates=len(candidates),
            kept=len(ordered),
            latency_ms=trace["latency_ms"],
        )
        return self._finish(result, ordered, trace)

    @staticmethod
    def _finish(
        result: RetrievalResult, chunks: list[Any], trace: dict[str, Any]
    ) -> RetrievalResult:
        result.chunks = chunks
        result.latency_ms += float(trace["latency_ms"])
        result.retrieval_config = {
            **result.retrieval_config,
            "top_k": trace["top_k"],
            "reranking_enabled": True,
            "reranking": trace,
        }
        return result
