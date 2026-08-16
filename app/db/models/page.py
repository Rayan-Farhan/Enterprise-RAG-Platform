"""Page representation model (ADR-005, Master Plan §8)."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.element import Element
    from app.db.models.version import DocumentVersion


class Page(Base, TimestampMixin):
    """Page-level representation preserving document layout and visual coordinates."""

    __tablename__ = "pages"

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
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        doc="1-indexed physical page number",
    )
    width: Mapped[float] = mapped_column(
        Float,
        default=595.0,
        nullable=False,
        doc="Page width in points or standard coordinates",
    )
    height: Mapped[float] = mapped_column(
        Float,
        default=842.0,
        nullable=False,
        doc="Page height in points or standard coordinates",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hash of all text content extracted on this page",
    )
    page_image_key: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        doc="MinIO/S3 object storage key for rendered page image (e.g., pages/<vid>/page_1.png)",
    )

    # Relationships
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="pages")
    elements: Mapped[list["Element"]] = relationship(
        "Element",
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="Element.sequence_index",
    )

    def __repr__(self) -> str:
        return f"<Page(version_id={self.version_id}, page_num={self.page_number})>"
