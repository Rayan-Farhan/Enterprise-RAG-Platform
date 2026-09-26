"""Propagate a version's metadata into every retrieval index (Task 6.3).

Metadata is authoritative in PostgreSQL (ADR-002) and copied into each index so
filters can run inside the engine (master §13). When it changes after indexing,
the copies must follow or filtering silently uses stale values. This rewrites
the metadata fields in place — Qdrant payloads, BM25 and neural sparse
documents — without re-embedding or re-encoding anything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.exceptions import NotFoundException
from app.core.logging import get_logger
from app.db.repositories.document_repo import DocumentRepository
from app.retrieval.indexer import ChunkIndexer
from app.retrieval.lexical_store import OpenSearchLexicalStore, get_lexical_store
from app.retrieval.sparse_store import OpenSearchSparseStore, get_sparse_store
from app.retrieval.vector_store import QdrantVectorStore, get_vector_store

logger = get_logger("app.retrieval.metadata_sync")


@dataclass
class MetadataSyncResult:
    version_id: uuid.UUID
    fields: dict[str, str | None]
    lexical_updated: int = 0
    sparse_updated: int = 0


class MetadataSync:
    """Pushes a version's current metadata into Qdrant and OpenSearch."""

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        lexical_store: OpenSearchLexicalStore | None = None,
        sparse_store: OpenSearchSparseStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.lexical_store = lexical_store or get_lexical_store()
        self.sparse_store = sparse_store or get_sparse_store()

    async def sync_version(
        self, session: AsyncSession, version_id: uuid.UUID
    ) -> MetadataSyncResult:
        version = await DocumentRepository(session).get_version_by_id(version_id)
        if version is None:
            raise NotFoundException(f"Document version '{version_id}' was not found")

        fields = ChunkIndexer._metadata_payload(version)
        result = MetadataSyncResult(version_id=version_id, fields=fields)

        self.vector_store.set_version_payload(str(version_id), fields)
        if self.settings.ENABLE_LEXICAL_INDEXING:
            result.lexical_updated = self.lexical_store.update_version_fields(
                str(version_id), fields
            )
        if self.settings.ENABLE_NEURAL_SPARSE:
            result.sparse_updated = self.sparse_store.update_version_fields(str(version_id), fields)

        logger.info(
            "metadata_synced",
            version_id=str(version_id),
            lexical_updated=result.lexical_updated,
            sparse_updated=result.sparse_updated,
        )
        return result
