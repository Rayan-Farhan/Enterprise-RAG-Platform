"""Pydantic schemas for document ingestion and retrieval APIs (ADR-030, ADR-035)."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadataInput(BaseModel):
    """Input HR and enterprise metadata fields (Master Plan §13)."""

    department: str | None = Field(default=None, description="Department name (e.g., Engineering, HR)")
    policy_type: str | None = Field(default=None, description="Policy category (e.g., Leave, Security)")
    policy_status: str | None = Field(default="active", description="Policy lifecycle status")
    country: str | None = Field(default=None, description="Applicable country code (e.g., US, UK, Global)")
    location: str | None = Field(default=None, description="Office location")
    employee_type: str | None = Field(default=None, description="e.g., Full-time, Contractor")
    grade: str | None = Field(default=None, description="e.g., L3+, Executive, All")
    confidentiality: str | None = Field(default="internal", description="Access level / confidentiality tier")
    audience: str | None = Field(default=None, description="Target audience")
    effective_from: datetime | None = Field(default=None, description="Effective starting timestamp")
    effective_until: datetime | None = Field(default=None, description="Effective expiration timestamp")
    authority: str | None = Field(default=None, description="Issuing body or authority")
    external_id: str | None = Field(default=None, description="External document reference ID")
    custom_attributes: dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom attributes")


class DocumentIngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    filename: str
    file_hash: str
    storage_key: str
    total_pages: int
    total_elements: int
    is_duplicate: bool
    created_at: datetime
    message: str


class ElementResponse(BaseModel):
    """Canonical structural element details."""

    id: uuid.UUID
    element_id: str
    parent_id: str | None
    element_type: str
    sequence_index: int
    page_number: int
    text_content: str
    content_hash: str
    bounding_box: dict[str, Any] | None
    table_data: dict[str, Any] | None
    asset_storage_key: str | None
    source_uri: str | None
    is_boilerplate: bool
    boilerplate_reason: str | None


class DocumentMetadataResponse(BaseModel):
    """Metadata response for a document version."""

    department: str | None
    policy_type: str | None
    policy_status: str | None
    country: str | None
    location: str | None
    employee_type: str | None
    grade: str | None
    confidentiality: str | None
    audience: str | None
    custom_attributes: dict[str, Any] = Field(default_factory=dict)


class DocumentVersionResponse(BaseModel):
    """Summary of a document version."""

    id: uuid.UUID
    version_number: int
    status: str
    total_pages: int
    total_elements: int
    parser_name: str
    parsing_duration_ms: float
    effective_from: datetime | None
    effective_until: datetime | None
    authority: str | None
    metadata: DocumentMetadataResponse | None = None
    created_at: datetime


class DocumentDetailResponse(BaseModel):
    """Detailed view of a document with versions and storage references."""

    id: uuid.UUID
    external_id: str | None
    title: str
    mime_type: str
    file_size_bytes: int
    file_hash: str
    storage_key: str
    source_priority: int
    versions: list[DocumentVersionResponse]
    created_at: datetime
    updated_at: datetime


class DocumentListItem(BaseModel):
    """Summary view for document listings."""

    id: uuid.UUID
    title: str
    mime_type: str
    file_size_bytes: int
    file_hash: str
    storage_key: str
    latest_version: int
    total_pages: int
    total_elements: int
    department: str | None = None
    policy_type: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: list[DocumentListItem]
    total: int
    limit: int
    offset: int
