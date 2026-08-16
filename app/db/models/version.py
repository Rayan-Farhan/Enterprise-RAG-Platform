"""DocumentVersion model (ADR-005, ADR-037, Master Plan §11)."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.chunk import Chunk
    from app.db.models.document import Document
    from app.db.models.element import Element
    from app.db.models.metadata import DocumentMetadata
    from app.db.models.page import Page


class VersionStatus(StrEnum):
    """Document lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class DocumentVersion(Base, TimestampMixin):
    """A specific immutable or versioned snapshot of a document."""

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=VersionStatus.ACTIVE.value,
        nullable=False,
        index=True,
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Temporal validity starting timestamp (ADR-037, ADR-038)",
    )
    effective_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Temporal validity expiration timestamp (ADR-037, ADR-038)",
    )
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
        doc="Reference to prior version this version replaces",
    )
    authority: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        doc="Issuing authority or committee (e.g., Global HR Compliance)",
    )
    total_pages: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    total_elements: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    parser_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Name of the parser that produced this version (e.g., docling, pymupdf)",
    )
    parsing_duration_ms: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="versions")
    pages: Mapped[list["Page"]] = relationship(
        "Page",
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )
    elements: Mapped[list["Element"]] = relationship(
        "Element",
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="Element.sequence_index",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="version",
        cascade="all, delete-orphan",
    )
    metadata_record: Mapped["DocumentMetadata | None"] = relationship(
        "DocumentMetadata",
        back_populates="version",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<DocumentVersion(id={self.id}, doc_id={self.document_id}, v={self.version_number}, status='{self.status}')>"
