"""Late-interaction reordering over the retrieved pool (Task 6.6, ADR-012).

An architectural capability, deliberately off (``ENABLE_LATE_INTERACTION``). A
single dense vector compresses a whole chunk; late interaction keeps a vector
per token (ColBERT) and scores a query against a chunk by MaxSim — each query
token takes its best-matching chunk token, and the maxima are summed. It is
more precise than one vector and cheaper than a cross-encoder, which is why it
sits where blueprint §17 puts it:

    dense + sparse -> candidate pool -> late interaction -> final ranking

Qdrant stores the token matrices as multivectors with the MAX_SIM comparator,
so scoring runs inside the engine, restricted to the candidates' own points.
Routing — which questions should use it — stays experimental per ADR-012.

The collection is derived from PostgreSQL (ADR-002) and keyed on a point ID
derived from the chunk ID and the model, so re-indexing overwrites. Chunks
must be indexed with ``LateInteractionIndexer`` before the stage can score
them; a pool with any unindexed candidate keeps its retrieval order rather
than mixing scored and unscored chunks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.repositories.chunk_repo import ChunkRepository
from app.ingestion.chunking.base import compute_point_id
from app.retrieval.schemas import RetrievalFilters, RetrievalResult

if TYPE_CHECKING:
    from app.models.gateway import ModelGateway
    from app.retrieval.channels import Retriever

logger = get_logger("app.retrieval.late_interaction")

_INDEX_BATCH = 8


def late_interaction_version(settings: AppSettings) -> str:
    """Identifies the token-vector space; part of every point ID."""
    return f"{settings.LATE_INTERACTION_MODEL}-d{settings.LATE_INTERACTION_DIMENSIONS}"


class LateInteractionStore:
    """The Qdrant multivector collection holding per-token chunk vectors."""

    def __init__(self, client: QdrantClient | None = None, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.collection_name = self.settings.LATE_INTERACTION_COLLECTION
        self._client = client

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.QDRANT_HOST,
                port=self.settings.QDRANT_PORT,
                api_key=self.settings.QDRANT_API_KEY or None,
                timeout=30,
            )
        return self._client

    def point_id(self, chunk_id: uuid.UUID) -> str:
        return compute_point_id(chunk_id, late_interaction_version(self.settings))

    def ensure_collection(self) -> bool:
        if self.client.collection_exists(self.collection_name):
            return False
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.settings.LATE_INTERACTION_DIMENSIONS,
                distance=qmodels.Distance.COSINE,
                multivector_config=qmodels.MultiVectorConfig(
                    comparator=qmodels.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        for field_name in ("chunking_version", "version_id"):
            self.client.create_payload_index(
                self.collection_name, field_name, qmodels.PayloadSchemaType.KEYWORD
            )
        logger.info("late_interaction_collection_created", collection=self.collection_name)
        return True

    def upsert(self, points: list[tuple[str, list[list[float]], dict[str, Any]]]) -> int:
        if not points:
            return 0
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(id=pid, vector=vectors, payload=payload)
                for pid, vectors, payload in points
            ],
            wait=True,
        )
        return len(points)

    def existing_ids(self, point_ids: list[str]) -> set[str]:
        if not point_ids or not self.client.collection_exists(self.collection_name):
            return set()
        found = self.client.retrieve(
            self.collection_name, ids=point_ids, with_payload=False, with_vectors=False
        )
        return {str(point.id) for point in found}

    def maxsim(
        self, query_vectors: list[list[float]], point_ids: list[str], limit: int
    ) -> list[tuple[str, float]]:
        """Score the given points against the query by MaxSim, best first."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vectors,
            query_filter=qmodels.Filter(must=[qmodels.HasIdCondition(has_id=point_ids)]),
            limit=limit,
            with_payload=False,
        )
        return [(str(point.id), float(point.score or 0.0)) for point in response.points]


@dataclass
class LateInteractionIndexingResult:
    version_id: uuid.UUID
    encoded: int
    skipped: int


class LateInteractionIndexer:
    """Encodes a version's chunks as token matrices, skipping those already present."""

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        store: LateInteractionStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or LateInteractionStore(settings=self.settings)
        self.gateway = gateway or _gateway_for(self.settings, explicit=settings is not None)

    async def index_version(
        self, session: AsyncSession, version_id: uuid.UUID, force: bool = False
    ) -> LateInteractionIndexingResult:
        chunks = await ChunkRepository(session).list_by_version(
            version_id, self.settings.CHUNKING_VERSION
        )
        self.store.ensure_collection()
        ids = {chunk.id: self.store.point_id(chunk.id) for chunk in chunks}
        present = set() if force else self.store.existing_ids(list(ids.values()))
        pending = [chunk for chunk in chunks if ids[chunk.id] not in present]

        for start in range(0, len(pending), _INDEX_BATCH):
            batch = pending[start : start + _INDEX_BATCH]
            encoded = await self.gateway.embed_multivector(
                [chunk.content for chunk in batch], input_type="document"
            )
            self.store.upsert(
                [
                    (
                        ids[chunk.id],
                        item.vectors,
                        {
                            "chunk_id": str(chunk.id),
                            "version_id": str(chunk.version_id),
                            "chunking_version": chunk.chunking_version,
                        },
                    )
                    for chunk, item in zip(batch, encoded.embeddings, strict=True)
                ]
            )
        return LateInteractionIndexingResult(
            version_id=version_id, encoded=len(pending), skipped=len(present)
        )


class LateInteractionRetriever:
    """Wraps any retriever: take its pool, reorder it by MaxSim, keep the top K."""

    def __init__(
        self,
        inner: Retriever,
        gateway: ModelGateway | None = None,
        store: LateInteractionStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.inner = inner
        self.settings = settings or inner.settings or get_settings()
        self.store = store or LateInteractionStore(settings=self.settings)
        self.gateway = gateway or _gateway_for(self.settings, explicit=settings is not None)

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
        pool = max(self.settings.LATE_INTERACTION_CANDIDATES, k)
        result = await self.inner.retrieve(query, session, pool, filters, min_score)
        candidates = result.chunks
        trace: dict[str, Any] = {
            "model": self.settings.LATE_INTERACTION_MODEL,
            "candidates": len(candidates),
            "status": "skipped",
            "latency_ms": 0.0,
            "moves": [],
        }
        if len(candidates) <= 1:
            return self._finish(result, candidates[:k], trace)

        started = time.perf_counter()
        try:
            by_point = {self.store.point_id(chunk.chunk_id): chunk for chunk in candidates}
            present = self.store.existing_ids(list(by_point))
            if len(present) < len(by_point):
                trace.update(status="incomplete_index", unindexed=len(by_point) - len(present))
                logger.warning("late_interaction_pool_not_indexed", unindexed=trace["unindexed"])
                return self._finish(result, candidates[:k], trace)

            encoded = await self.gateway.embed_multivector([query], input_type="query")
            scored = self.store.maxsim(encoded.embeddings[0].vectors, list(by_point), limit=k)
        except Exception as exc:  # noqa: BLE001 - the retriever's own order still stands
            trace.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            logger.warning("late_interaction_failed_keeping_retrieval_order", error=str(exc))
            return self._finish(result, candidates[:k], trace)

        ordered = []
        for new_rank, (point_id, score) in enumerate(scored, start=1):
            chunk = by_point[point_id]
            ordered.append(
                chunk.model_copy(
                    update={
                        "score": score,
                        "rank": new_rank,
                        "metadata": {**chunk.metadata, "pre_late_interaction_rank": chunk.rank},
                    }
                )
            )
            trace["moves"].append([chunk.rank, new_rank])
        trace.update(status="ok", latency_ms=round((time.perf_counter() - started) * 1000, 2))
        return self._finish(result, ordered, trace)

    @staticmethod
    def _finish(
        result: RetrievalResult, chunks: list[Any], trace: dict[str, Any]
    ) -> RetrievalResult:
        result.chunks = chunks
        result.latency_ms += float(trace["latency_ms"])
        result.retrieval_config = {**result.retrieval_config, "late_interaction": trace}
        return result


def _gateway_for(settings: AppSettings, explicit: bool) -> ModelGateway:
    from app.models.gateway import build_model_gateway, get_model_gateway

    return build_model_gateway(settings) if explicit else get_model_gateway()
