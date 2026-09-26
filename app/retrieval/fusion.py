"""Hybrid retrieval by rank fusion (Task 6.4, ADR-013).

Each active channel — dense, BM25, neural sparse — ranks its own candidate pool
of ``RETRIEVAL_CANDIDATE_LIMIT`` chunks. Fusion combines the pools into one
ranking, and only the fused top-K is rehydrated from PostgreSQL.

Two methods, selected by ``FUSION_METHOD``:

* ``rrf`` — Reciprocal Rank Fusion: ``score = Σ weight_c / (k + rank_c)`` over
  the channels that returned the chunk. It uses ranks only, so a cosine
  similarity and an unbounded BM25 score combine without calibration. This is
  the default because the channels' scores live on incomparable scales.
* ``weighted`` — ``Σ weight_c × normalised_score_c``, with each channel's scores
  min-max normalised within the query. Kept so the matrix can compare it.

Channel loss degrades instead of failing: a channel that errors (OpenSearch
down, sparse model not deployed) is recorded in the trace and fusion continues
with the rest. Only losing every channel is an error.

Filters are passed to every channel unchanged, so metadata narrowing (Task 6.3)
wrapped around this retriever is pushed into all engines at once.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.hydration import hydrate_ranked
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk

if TYPE_CHECKING:
    from app.retrieval.channels import Retriever

logger = get_logger("app.retrieval.fusion")


@dataclass
class FusedCandidate:
    chunk_id: uuid.UUID
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    best: RetrievedChunk | None = None  # the channel copy, used without a session


def reciprocal_rank_fusion(
    rankings: Mapping[str, list[RetrievedChunk]],
    k: int,
    weights: Mapping[str, float],
) -> list[FusedCandidate]:
    """Fuse ranked lists by RRF. Ties break on the best single-channel rank."""
    fused: dict[uuid.UUID, FusedCandidate] = {}
    for channel, chunks in rankings.items():
        weight = weights.get(channel, 1.0)
        for rank, chunk in enumerate(chunks, start=1):
            candidate = fused.setdefault(chunk.chunk_id, FusedCandidate(chunk.chunk_id, best=chunk))
            candidate.score += weight / (k + rank)
            candidate.ranks[channel] = rank
    return sorted(fused.values(), key=lambda c: (-c.score, min(c.ranks.values())))


def weighted_score_fusion(
    rankings: Mapping[str, list[RetrievedChunk]],
    weights: Mapping[str, float],
) -> list[FusedCandidate]:
    """Fuse by weighted, per-channel min-max normalised scores."""
    fused: dict[uuid.UUID, FusedCandidate] = {}
    for channel, chunks in rankings.items():
        if not chunks:
            continue
        weight = weights.get(channel, 1.0)
        scores = [c.score for c in chunks]
        low, high = min(scores), max(scores)
        span = high - low
        for rank, chunk in enumerate(chunks, start=1):
            normalised = (chunk.score - low) / span if span > 0 else 1.0
            candidate = fused.setdefault(chunk.chunk_id, FusedCandidate(chunk.chunk_id, best=chunk))
            candidate.score += weight * normalised
            candidate.ranks[channel] = rank
    return sorted(fused.values(), key=lambda c: (-c.score, min(c.ranks.values())))


class FusionRetriever:
    """Queries every active channel and fuses their rankings."""

    channel = "hybrid"

    def __init__(
        self,
        channels: dict[str, Retriever] | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.channels = channels if channels is not None else self._build_channels()
        if not self.channels:
            raise ValueError("Hybrid retrieval needs at least one active channel")

    def _build_channels(self) -> dict[str, Retriever]:
        from app.retrieval.channels import build_channel

        names = list(dict.fromkeys(self.settings.HYBRID_CHANNELS))
        if "sparse" in names and not self.settings.ENABLE_NEURAL_SPARSE:
            logger.info("fusion_channel_disabled", channel="sparse", reason="ENABLE_NEURAL_SPARSE")
            names.remove("sparse")
        return {name: build_channel(name, self.settings) for name in names}

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        """Every channel indexes the same chunks, so any live one gives the pool."""
        for retriever in self.channels.values():
            try:
                return await retriever.candidate_count(filters)
            except Exception as exc:  # noqa: BLE001 - try the next channel
                logger.warning("fusion_candidate_count_failed", error=str(exc))
        return 0

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        k = top_k or self.settings.RETRIEVAL_TOP_K
        pool = max(self.settings.RETRIEVAL_CANDIDATE_LIMIT, k)

        names = list(self.channels)
        outcomes = await asyncio.gather(
            *(
                # No session: each channel returns IDs, ranks and scores from its
                # index; only the fused top-K is read from PostgreSQL, once.
                self.channels[name].retrieve(query, None, pool, filters, min_score)
                for name in names
            ),
            return_exceptions=True,
        )

        rankings: dict[str, list[RetrievedChunk]] = {}
        channel_trace: dict[str, dict[str, Any]] = {}
        for name, outcome in zip(names, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                channel_trace[name] = {
                    "status": "failed",
                    "error": f"{type(outcome).__name__}: {outcome}",
                }
                logger.warning("fusion_channel_failed", channel=name, error=str(outcome))
                continue
            rankings[name] = outcome.chunks
            channel_trace[name] = {
                "status": "ok",
                "hits": len(outcome.chunks),
                "latency_ms": round(outcome.latency_ms, 2),
            }

        if not rankings:
            raise RuntimeError(f"Every hybrid channel failed: {channel_trace}")

        weights = self.settings.FUSION_WEIGHTS
        fused = (
            reciprocal_rank_fusion(rankings, self.settings.FUSION_RRF_K, weights)
            if self.settings.FUSION_METHOD == "rrf"
            else weighted_score_fusion(rankings, weights)
        )
        top = fused[:k]

        if session is not None:
            chunks = await hydrate_ranked(
                session, [(c.chunk_id, c.score) for c in top], self.channel
            )
        else:
            chunks = [
                c.best.model_copy(update={"score": c.score, "rank": rank, "channel": self.channel})
                for rank, c in enumerate(top, start=1)
                if c.best is not None
            ]
        ranks = {c.chunk_id: c.ranks for c in top}
        for chunk in chunks:
            chunk.metadata = {**chunk.metadata, "fusion_ranks": ranks.get(chunk.chunk_id, {})}

        latency_ms = (time.perf_counter() - started) * 1000
        trace = {
            "method": self.settings.FUSION_METHOD,
            "rrf_k": self.settings.FUSION_RRF_K,
            "weights": {name: weights.get(name, 1.0) for name in names},
            "pool_per_channel": pool,
            "channels": channel_trace,
            "fused_candidates": len(fused),
            "in_every_channel": sum(1 for c in top if len(c.ranks) == len(rankings)),
        }
        logger.info(
            "hybrid_retrieval_complete",
            channels={n: t["status"] for n, t in channel_trace.items()},
            fused_candidates=len(fused),
            hits=len(chunks),
            latency_ms=round(latency_ms, 2),
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_candidates=len(fused),
            latency_ms=latency_ms,
            embedding_version=self.settings.effective_embedding_version
            if "dense" in rankings
            else None,
            retrieval_config=self.config_snapshot(k, filters, trace),
        )

    def config_snapshot(
        self,
        top_k: int,
        filters: RetrievalFilters | None,
        trace: dict[str, Any],
    ) -> dict[str, object]:
        return {
            "channels": [f"hybrid:{name}" for name in self.channels],
            "top_k": top_k,
            "min_score": None,
            "chunking_version": self.settings.CHUNKING_VERSION,
            "chunk_size_tokens": self.settings.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": self.settings.CHUNK_OVERLAP_TOKENS,
            "embedding_version": self.settings.effective_embedding_version,
            "reranking_enabled": self.settings.ENABLE_RERANKING,
            "filters": filters.model_dump(mode="json", exclude_defaults=True) if filters else {},
            "fusion": trace,
        }
