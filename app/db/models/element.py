"""Canonical Element model (ADR-005, Master Plan §8, §10)."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.page import Page
    from app.db.models.version import DocumentVersion


class Element(Base, TimestampMixin):
    """Atomic structural unit of a document preserving layout, coordinates, and provenance."""

    __tablename__ = "elements"

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
    page_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        doc="1-indexed physical page number",
    )
    element_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        doc="Stable element identifier generated during ingestion",
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        doc="Hierarchical parent element identifier (e.g. parent section heading)",
    )
    element_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        doc="Canonical type (heading, paragraph, table, figure, image, list, formula, header, footer)",
    )
    sequence_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        doc="Natural reading sequence index within the document",
    )
    text_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Extracted textual content or markdown representation",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hash of this element's text_content",
    )
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Physical coordinates on page: {x0, y0, x1, y1, unit}",
    )
    table_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Structured table payload: {num_rows, num_cols, headers, cells, markdown}",
    )
    asset_storage_key: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        doc="MinIO/S3 object storage key for image or cropped table/figure asset",
    )
    source_uri: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        doc="Original URI or path from which this element was extracted",
    )
    is_boilerplate: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Flag indicating recurring boilerplate (header/footer/legal notice)",
    )
    boilerplate_reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        doc="Reason why this element was flagged as boilerplate",
    )
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Extensible metadata (e.g., heading level, font styling)",
    )

    # Relationships
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="elements")
    page: Mapped["Page"] = relationship("Page", back_populates="elements")

    def __repr__(self) -> str:
        return f"<Element(id={self.id}, type='{self.element_type}', page={self.page_number}, seq={self.sequence_index})>"
