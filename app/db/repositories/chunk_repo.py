"""Chunk repository backing indexing and retrieval hydration (Stage 3)."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.chunk import Chunk
from app.db.models.version import DocumentVersion


class ChunkRepository:
    """Async repository for derived chunk records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_by_version(
        self,
        version_id: uuid.UUID,
        chunking_version: str | None = None,
    ) -> list[Chunk]:
        """Return all chunks for a version in chunk order."""
        stmt = select(Chunk).where(Chunk.version_id == version_id)
        if chunking_version is not None:
            stmt = stmt.where(Chunk.chunking_version == chunking_version)
        result = await self.session.execute(stmt.order_by(Chunk.chunk_index.asc()))
        return list(result.scalars().all())

    async def list_unindexed(
        self,
        version_id: uuid.UUID,
        chunking_version: str,
        embedding_version: str,
    ) -> list[Chunk]:
        """Return chunks lacking a current vector for the active embedding version.

        A chunk indexed under a different embedding version counts as unindexed,
        which is what makes an embedding-model change trigger re-indexing without
        a manual purge.
        """
        stmt = (
            select(Chunk)
            .where(
                Chunk.version_id == version_id,
                Chunk.chunking_version == chunking_version,
            )
            .where(
                (Chunk.embedding_id.is_(None))
                | (Chunk.embedding_version.is_(None))
                | (Chunk.embedding_version != embedding_version)
            )
            .order_by(Chunk.chunk_index.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_many_by_ids(self, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
        """Fetch chunks by ID with their version and document eagerly loaded.

        Retrieval returns vector-store payloads; this rehydrates the authoritative
        PostgreSQL record, because ADR-002 makes the vector store derived and
        never a source of truth.
        """
        if not chunk_ids:
            return []
        stmt = (
            select(Chunk)
            .options(
                selectinload(Chunk.version).selectinload(DocumentVersion.metadata_record),
                selectinload(Chunk.document),
            )
            .where(Chunk.id.in_(chunk_ids))
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_version(self, version_id: uuid.UUID) -> int:
        """Count chunks belonging to a version."""
        stmt = select(func.count()).select_from(Chunk).where(Chunk.version_id == version_id)
        result = await self.session.execute(stmt)
        return result.scalar_one() or 0

    async def mark_indexed(
        self,
        chunk_ids: list[uuid.UUID],
        embedding_version: str,
        point_ids: dict[uuid.UUID, str],
    ) -> None:
        """Record which vector point now represents each chunk."""
        if not chunk_ids:
            return
        chunks = await self.session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
        for chunk in chunks.scalars().all():
            chunk.embedding_id = point_ids.get(chunk.id)
            chunk.embedding_version = embedding_version
        await self.session.flush()
