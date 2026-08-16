"""Chunk model prepared for chunking and retrieval (ADR-006, Stage 3 & 5)."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.version import DocumentVersion


class Chunk(Base, TimestampMixin):
    """Derived retrieval unit composed of one or more canonical elements."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
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
    primary_page_number: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        index=True,
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

    # Relationships
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, version_id={self.version_id}, idx={self.chunk_index}, tokens={self.token_count})>"
