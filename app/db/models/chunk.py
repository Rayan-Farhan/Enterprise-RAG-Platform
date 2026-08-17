"""Chunk model prepared for chunking and retrieval (ADR-006, ADR-036, Stage 3 & 5)."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.document import Document
    from app.db.models.version import DocumentVersion


class Chunk(Base, TimestampMixin):
    """Derived retrieval unit composed of one or more canonical elements.

    Chunk identity is deterministic, not random: ``id`` is a UUIDv5 derived from
    ``(version_id, element_ids, chunk_index, chunking_version)``. Re-chunking an
    unchanged version therefore produces byte-identical IDs, which is the
    idempotency foundation required by ADR-036 and reused by Stage 7's Celery
    replay semantics.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "chunking_version",
            "chunk_index",
            name="uq_chunks_version_chunking_index",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        doc="Deterministic UUIDv5 chunk identity (never randomly generated)",
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        doc="0-indexed sequence of this chunk within the document version",
    )
    chunk_type: Mapped[str] = mapped_column(
        String(32),
        default="mixed",
        nullable=False,
        index=True,
        doc="Chunk composition type (paragraph, section, table, figure, list, mixed) per ADR-006",
    )
    chunking_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="Chunking strategy version participating in the deterministic chunk ID",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Text payload indexed for dense and sparse search",
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    element_ids: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="List of source canonical element_ids comprising this chunk",
    )
    section_path: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="Heading ancestry from document root to this chunk, outermost first",
    )
    primary_page_number: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        index=True,
    )
    page_span: Mapped[list[int]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
        doc="All page numbers this chunk draws content from",
    )
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Union bounding box across source elements",
    )
    embedding_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Deterministic vector store point ID in Qdrant",
    )
    embedding_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Embedding model version that produced the indexed vector",
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<Chunk(id={self.id}, version_id={self.version_id}, "
            f"idx={self.chunk_index}, tokens={self.token_count})>"
        )
