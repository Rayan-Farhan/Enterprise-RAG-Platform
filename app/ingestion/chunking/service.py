"""Chunking orchestration and idempotent persistence (Task 3.1, ADR-036)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.exceptions import NotFoundException
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.db.models.version import DocumentVersion
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.chunking.base import ChunkCandidate, ChunkingContext, ChunkingStrategy
from app.ingestion.chunking.contextual import ContextualChunker
from app.ingestion.chunking.fixed_size import FixedSizeChunker
from app.ingestion.chunking.hierarchical import HierarchicalChunker
from app.ingestion.chunking.structure_aware import StructureAwareChunker

logger = get_logger("app.ingestion.chunking")


@dataclass
class ChunkingResult:
    """Outcome of chunking one document version."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    chunking_version: str
    strategy: str
    chunks_created: int
    chunks_updated: int
    chunks_removed: int
    total_chunks: int
    total_tokens: int

    @property
    def is_noop(self) -> bool:
        """True when re-chunking changed nothing — the idempotency signal."""
        return self.chunks_created == 0 and self.chunks_removed == 0


#: Every strategy takes the same three construction arguments, so the experiment
#: matrix in Task 5.3 can sweep size and overlap across all of them uniformly.
STRATEGIES: dict[str, Callable[..., ChunkingStrategy]] = {
    "fixed": FixedSizeChunker,
    "structure_aware": StructureAwareChunker,
    "hierarchical": HierarchicalChunker,
    "contextual": ContextualChunker,
}


def build_strategy(settings: AppSettings | None = None) -> ChunkingStrategy:
    """Construct the configured chunking strategy.

    The config field is a ``Literal``, so an unknown name fails at settings
    validation rather than here; the lookup below therefore cannot legitimately
    miss, and a KeyError would mean the Literal and this table drifted apart.
    """
    cfg = settings or get_settings()
    strategy: ChunkingStrategy = STRATEGIES[cfg.CHUNKING_STRATEGY](
        chunk_size_tokens=cfg.CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=cfg.CHUNK_OVERLAP_TOKENS,
        chunking_version=cfg.CHUNKING_VERSION,
    )
    return strategy


class ChunkingService:
    """Chunks a persisted document version and stores chunks idempotently."""

    def __init__(
        self,
        strategy: ChunkingStrategy | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.strategy = strategy or build_strategy(self.settings)

    async def chunk_version(
        self,
        session: AsyncSession,
        version_id: uuid.UUID,
    ) -> ChunkingResult:
        """Chunk a document version, replacing any stale chunks for this strategy."""
        repo = DocumentRepository(session)
        version = await repo.get_version_by_id(version_id)
        if version is None:
            raise NotFoundException(f"Document version '{version_id}' was not found")

        elements = await repo.get_all_elements_by_version(
            version_id=version_id,
            include_boilerplate=False,
        )
        log = logger.bind(
            version_id=str(version_id),
            document_id=str(version.document_id),
            elements=len(elements),
            strategy=self.strategy.strategy_name,
        )

        document = await repo.get_by_id(version.document_id)
        candidates = self.strategy.chunk(elements, self._context(document, version))
        chunking_version = self.strategy.chunking_version

        existing = await self._load_existing(session, version_id, chunking_version)
        desired_ids: set[uuid.UUID] = set()
        created = 0
        updated = 0

        for candidate in candidates:
            chunk_id = candidate.identity(version_id, chunking_version)
            desired_ids.add(chunk_id)

            current = existing.get(chunk_id)
            if current is None:
                session.add(
                    self._to_model(
                        chunk_id=chunk_id,
                        document_id=version.document_id,
                        version_id=version_id,
                        chunking_version=chunking_version,
                        candidate=candidate,
                    )
                )
                created += 1
            elif current.content != candidate.content:
                # Same identity, different content means the element text changed
                # underneath a stable ID. Refresh the payload and drop the stale
                # vector pointer so the indexer re-embeds it.
                current.content = candidate.content
                current.token_count = candidate.token_count
                current.section_path = candidate.section_path
                current.embedding_id = None
                current.embedding_version = None
                updated += 1

        # Chunks that no longer exist for this strategy are removed. Scoping the
        # delete by chunking_version keeps other strategies' chunks intact, which
        # Stage 5 relies on when comparing strategies side by side.
        stale_ids = [cid for cid in existing if cid not in desired_ids]
        if stale_ids:
            await session.execute(delete(Chunk).where(Chunk.id.in_(stale_ids)))

        await session.flush()

        result = ChunkingResult(
            document_id=version.document_id,
            version_id=version_id,
            chunking_version=chunking_version,
            strategy=self.strategy.strategy_name,
            chunks_created=created,
            chunks_updated=updated,
            chunks_removed=len(stale_ids),
            total_chunks=len(candidates),
            total_tokens=sum(c.token_count for c in candidates),
        )

        log.info(
            "chunking_complete",
            created=created,
            updated=updated,
            removed=len(stale_ids),
            total=len(candidates),
        )
        return result

    @staticmethod
    def _context(document: Document | None, version: DocumentVersion) -> ChunkingContext:
        """Assemble the document-level facts a strategy may prefix onto chunks.

        Only fields the canonical model actually carries are included. A prefix
        of "Effective Date: None" is noise in every embedding, so absent values
        are omitted rather than stringified.
        """
        metadata: dict[str, Any] = {}
        if version.effective_from is not None:
            metadata["effective_date"] = version.effective_from.date().isoformat()
        if version.authority:
            metadata["policy_id"] = version.authority
        if version.version_number:
            metadata["version_label"] = f"v{version.version_number}"

        return ChunkingContext(
            document_title=(document.title if document else ""),
            document_id=version.document_id,
            version_id=version.id,
            metadata=metadata,
        )

    async def _load_existing(
        self,
        session: AsyncSession,
        version_id: uuid.UUID,
        chunking_version: str,
    ) -> dict[uuid.UUID, Chunk]:
        stmt = select(Chunk).where(
            Chunk.version_id == version_id,
            Chunk.chunking_version == chunking_version,
        )
        result = await session.execute(stmt)
        return {chunk.id: chunk for chunk in result.scalars().all()}

    @staticmethod
    def _to_model(
        chunk_id: uuid.UUID,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
        chunking_version: str,
        candidate: ChunkCandidate,
    ) -> Chunk:
        return Chunk(
            id=chunk_id,
            document_id=document_id,
            version_id=version_id,
            chunk_index=candidate.chunk_index,
            chunk_type=str(candidate.chunk_type),
            chunking_version=chunking_version,
            content=candidate.content,
            token_count=candidate.token_count,
            element_ids=candidate.element_ids,
            section_path=candidate.section_path,
            primary_page_number=candidate.primary_page_number,
            page_span=candidate.page_span,
            bounding_box=candidate.bounding_box,
        )


_chunking_service: ChunkingService | None = None


def get_chunking_service() -> ChunkingService:
    """Return the singleton ChunkingService."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service
