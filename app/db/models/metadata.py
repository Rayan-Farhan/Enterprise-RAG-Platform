"""Enterprise HR Metadata model (Master Plan §13)."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.version import DocumentVersion


class DocumentMetadata(Base, TimestampMixin):
    """Enterprise policy and document metadata used for structured filtering in retrieval."""

    __tablename__ = "document_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Common HR & Policy indexed fields (Master Plan §13)
    department: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="e.g., Engineering, Human Resources, Finance, Legal",
    )
    policy_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="e.g., Leave, Compensation, Travel, Security, Conduct",
    )
    policy_status: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="e.g., Active, Under Review, Proposed, Deprecated",
    )
    country: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="ISO country code or name (e.g., US, UK, Global)",
    )
    location: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="Office location or Remote (e.g., New York, London, Remote)",
    )
    employee_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="e.g., Full-time, Part-time, Contractor, Intern",
    )
    grade: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="Job level or band (e.g., L3+, Executive, All)",
    )
    confidentiality: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        doc="e.g., Public, Internal, Confidential, Highly Confidential",
    )
    audience: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        doc="e.g., All Employees, Managers, People Leaders",
    )

    # Dynamic custom attributes stored as JSON/JSONB
    custom_attributes: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Arbitrary extensible key-value metadata attributes",
    )

    # Relationships
    version: Mapped["DocumentVersion"] = relationship("DocumentVersion", back_populates="metadata_record")

    def __repr__(self) -> str:
        return f"<DocumentMetadata(version_id={self.version_id}, dept='{self.department}', policy='{self.policy_type}')>"
