"""Retrieval and citation data contracts (Stage 3, ADR-026)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ChunkPayload(BaseModel):
    """Payload stored alongside every vector point.

    Filterable fields are flat and typed so both Qdrant (Stage 3) and OpenSearch
    (Stage 6) can index them. The ACL fields are written as placeholders now so
    Stage 8 can enforce pre-query filtering without a full re-index.
    """

    chunk_id: str
    document_id: str
    version_id: str
    chunk_index: int
    chunk_type: str = "mixed"
    chunking_version: str
    embedding_version: str
    content: str
    token_count: int = 0
    element_ids: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    page_number: int = 1
    page_span: list[int] = Field(default_factory=list)
    bounding_box: dict[str, Any] | None = None
    document_title: str | None = None

    # HR metadata (master §13) — filterable
    department: str | None = None
    policy_type: str | None = None
    policy_status: str | None = None
    country: str | None = None
    employee_type: str | None = None
    grade: str | None = None
    confidentiality: str | None = None
    audience: str | None = None

    # ACL placeholders — populated and enforced in Stage 8 (ADR-023)
    tenant_id: str | None = None
    department_id: str | None = None
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    classification: str = "internal"


class RetrievalFilters(BaseModel):
    """Metadata constraints pushed into the vector store query, never applied after."""

    document_ids: list[uuid.UUID] = Field(default_factory=list)
    version_ids: list[uuid.UUID] = Field(default_factory=list)
    department: str | None = None
    policy_type: str | None = None
    policy_status: str | None = None
    country: str | None = None
    employee_type: str | None = None
    grade: str | None = None
    page_number: int | None = None

    def is_empty(self) -> bool:
        """True when no constraint is set."""
        return not any(
            [
                self.document_ids,
                self.version_ids,
                self.department,
                self.policy_type,
                self.policy_status,
                self.country,
                self.employee_type,
                self.grade,
                self.page_number is not None,
            ]
        )


class Citation(BaseModel):
    """A resolvable pointer from an answer back to canonical evidence (ADR-026).

    Every field down to ``bounding_box`` is populated in Stage 3 even though the
    box is only consumed in Stage 9/13, because backfilling provenance after the
    index is built is exactly what ADR-005 exists to prevent.
    """

    marker: str = Field(description="Inline marker used in the answer, e.g. [1]")
    document_id: uuid.UUID
    version_id: uuid.UUID
    chunk_id: uuid.UUID
    document_title: str | None = None
    version_number: int | None = None
    page_number: int
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    bounding_box: dict[str, Any] | None = None
    quote: str | None = Field(
        default=None, description="Short supporting excerpt from the cited chunk"
    )

    @property
    def section_label(self) -> str:
        """Human-readable section path."""
        return " > ".join(p for p in self.section_path if p) or "(untitled section)"


class RetrievedChunk(BaseModel):
    """A chunk returned by a retrieval channel, with score and provenance."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    content: str
    score: float
    channel: str = Field(default="dense", description="Retrieval channel that produced this hit")
    rank: int = 0
    chunk_index: int = 0
    chunk_type: str = "mixed"
    token_count: int = 0
    page_number: int = 1
    page_span: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    bounding_box: dict[str, Any] | None = None
    document_title: str | None = None
    version_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_citation(self, marker: str, quote_chars: int = 240) -> Citation:
        """Build the citation object for this chunk."""
        quote = self.content.strip()
        if len(quote) > quote_chars:
            quote = quote[:quote_chars].rstrip() + "…"

        return Citation(
            marker=marker,
            document_id=self.document_id,
            version_id=self.version_id,
            chunk_id=self.chunk_id,
            document_title=self.document_title,
            version_number=self.version_number,
            page_number=self.page_number,
            section_path=self.section_path,
            element_ids=self.element_ids,
            bounding_box=self.bounding_box,
            quote=quote,
        )


class RetrievalResult(BaseModel):
    """The outcome of one retrieval call, including the config that produced it."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_candidates: int = 0
    latency_ms: float = 0.0
    embedding_version: str | None = None
    retrieval_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks
