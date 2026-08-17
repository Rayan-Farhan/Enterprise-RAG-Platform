"""Chunk indexing pipeline: chunk -> embed -> upsert (Task 3.2, ADR-036).

Written as small composable steps because Stage 7 wraps each as a Celery task
without rewriting the logic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.exceptions import NotFoundException
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.repositories.chunk_repo import ChunkRepository
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.chunking.base import compute_point_id
from app.retrieval.embedding import EmbeddingService, get_embedding_service
from app.retrieval.schemas import ChunkPayload
from app.retrieval.vector_store import QdrantVectorStore, VectorPoint, get_vector_store

logger = get_logger("app.retrieval.indexer")


@dataclass
class IndexingResult:
    """Outcome of indexing one document version."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int
    points_upserted: int
    embedding_version: str
    provider: str = "unknown"
    dimensions: int = 0
    embedding_latency_ms: float = 0.0
    rate_limit_waits: int = 0

    @property
    def is_noop(self) -> bool:
        """True when everything was already indexed — the idempotency signal."""
        return self.chunks_embedded == 0


class ChunkIndexer:
    """Embeds a version's chunks and upserts them into the vector store."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    async def index_version(
        self,
        session: AsyncSession,
        version_id: uuid.UUID,
        force: bool = False,
    ) -> IndexingResult:
        """Index a document version's chunks.

        Only chunks lacking a current vector are embedded, so re-running is cheap
        and produces zero duplicate points. ``force`` re-embeds everything, which
        is what Stage 5/6 experiments need when a parameter changes without a
        version bump.
        """
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        version = await doc_repo.get_version_by_id(version_id)
        if version is None:
            raise NotFoundException(f"Document version '{version_id}' was not found")

        chunking_version = self.settings.CHUNKING_VERSION
        embedding_version = self.settings.effective_embedding_version

        all_chunks = await chunk_repo.list_by_version(version_id, chunking_version)
        pending = (
            all_chunks
            if force
            else await chunk_repo.list_unindexed(
                version_id=version_id,
                chunking_version=chunking_version,
                embedding_version=embedding_version,
            )
        )

        log = logger.bind(
            version_id=str(version_id),
            total=len(all_chunks),
            pending=len(pending),
            force=force,
        )

        result = IndexingResult(
            document_id=version.document_id,
            version_id=version_id,
            chunks_total=len(all_chunks),
            chunks_embedded=0,
            chunks_skipped=len(all_chunks) - len(pending),
            points_upserted=0,
            embedding_version=embedding_version,
        )

        if not pending:
            log.info("indexing_noop_all_chunks_current")
            return result

        embedded = await self.embeddings.embed_texts([c.content for c in pending])
        if len(embedded.vectors) != len(pending):
            raise ValueError(
                f"Embedding count mismatch: {len(embedded.vectors)} vectors "
                f"for {len(pending)} chunks"
            )


        # The collection is created with the dimensionality the provider actually
        # returned, not the configured guess, so a provider default change is a
        # loud failure at upsert rather than silent truncation.
        self.vector_store.ensure_collection(dimensions=embedded.dimensions)

        metadata = self._metadata_payload(version)
        document_title = version.document.title if version.document else None

        points: list[VectorPoint] = []
        point_ids: dict[uuid.UUID, str] = {}

        for chunk, vector in zip(pending, embedded.vectors, strict=True):
            point_id = compute_point_id(chunk.id, embedding_version)
            point_ids[chunk.id] = point_id
            points.append(
                VectorPoint(
                    point_id=point_id,
                    vector=vector,
                    payload=self._build_payload(
                        chunk=chunk,
                        embedding_version=embedding_version,
                        document_title=document_title,
                        metadata=metadata,
                    ),
                )
            )

        result.points_upserted = self.vector_store.upsert(points)
        await chunk_repo.mark_indexed(
            chunk_ids=[c.id for c in pending],
            embedding_version=embedding_version,
            point_ids=point_ids,
        )

        result.chunks_embedded = len(pending)
        result.provider = embedded.provider
        result.dimensions = embedded.dimensions
        result.embedding_latency_ms = embedded.total_latency_ms
        result.rate_limit_waits = embedded.rate_limit_waits

        log.info(
            "indexing_complete",
            embedded=result.chunks_embedded,
            upserted=result.points_upserted,
            dimensions=result.dimensions,
            provider=result.provider,
        )
        return result

    @staticmethod
    def _metadata_payload(version: object) -> dict[str, str | None]:
        """Flatten HR metadata from the version record for payload filtering."""
        record = getattr(version, "metadata_record", None)
        fields = (
            "department",
            "policy_type",
            "policy_status",
            "country",
            "employee_type",
            "grade",
            "confidentiality",
            "audience",
        )
        if record is None:
            return dict.fromkeys(fields)
        return {name: getattr(record, name, None) for name in fields}

    @staticmethod
    def _build_payload(
        chunk: Chunk,
        embedding_version: str,
        document_title: str | None,
        metadata: dict[str, str | None],
    ) -> ChunkPayload:
        return ChunkPayload(
            chunk_id=str(chunk.id),
            document_id=str(chunk.document_id),
            version_id=str(chunk.version_id),
            chunk_index=chunk.chunk_index,
            chunk_type=chunk.chunk_type,
            chunking_version=chunk.chunking_version,
            embedding_version=embedding_version,
            content=chunk.content,
            token_count=chunk.token_count,
            element_ids=list(chunk.element_ids or []),
            section_path=list(chunk.section_path or []),
            page_number=chunk.primary_page_number,
            page_span=list(chunk.page_span or []),
            bounding_box=chunk.bounding_box,
            document_title=document_title,
            **metadata,  # type: ignore[arg-type]
        )


_chunk_indexer: ChunkIndexer | None = None


def get_chunk_indexer() -> ChunkIndexer:
    """Return the singleton ChunkIndexer."""
    global _chunk_indexer
    if _chunk_indexer is None:
        _chunk_indexer = ChunkIndexer()
    return _chunk_indexer
