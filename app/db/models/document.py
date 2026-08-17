"""Document root model (ADR-002, ADR-005, Master Plan §8)."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.chunk import Chunk
    from app.db.models.version import DocumentVersion


class Document(Base, TimestampMixin):
    """Canonical document root representing a distinct logical or physical source document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        doc="Optional external system identifier",
    )
    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
    )
    mime_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )
    file_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        doc="SHA-256 hash of original file content for exact duplicate detection",
    )
    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        doc="MinIO/S3 object storage key (e.g., original/<hash>.pdf)",
    )
    duplicate_group_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        nullable=True,
        index=True,
        doc="Shared cluster identifier for near/semantic duplicates (Master Plan §10)",
    )
    source_priority: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        doc="Authority priority ranking (higher value = higher priority)",
    )

    # Relationships
    versions: Mapped[list["DocumentVersion"]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}', hash='{self.file_hash[:8]}...')>"
