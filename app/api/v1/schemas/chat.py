"""Chat and search API contracts (Task 3.6, ADR-035)."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from app.retrieval.schemas import RetrievalFilters


class MetadataFilterInput(BaseModel):
    """HR metadata constraints pushed into the retrieval query."""

    document_ids: list[uuid.UUID] = Field(default_factory=list)
    version_ids: list[uuid.UUID] = Field(default_factory=list)
    department: str | None = None
    policy_type: str | None = None
    policy_status: str | None = None
    country: str | None = None
    employee_type: str | None = None
    grade: str | None = None
    page_number: int | None = Field(default=None, ge=1)

    def to_filters(self) -> RetrievalFilters:
        return RetrievalFilters(**self.model_dump())


class ChatRequest(BaseModel):
    """A question to answer against the HR corpus."""

    query: str = Field(min_length=1, max_length=4000, description="Natural language question")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Override retrieval top-K")
    stream: bool = Field(default=False, description="Return an SSE stream instead of JSON")
    filters: MetadataFilterInput | None = None


class CitationResponse(BaseModel):
    """A resolvable citation returned with an answer."""

    marker: str
    document_id: uuid.UUID
    version_id: uuid.UUID
    chunk_id: uuid.UUID
    document_title: str | None = None
    version_number: int | None = None
    page_number: int
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    bounding_box: dict[str, Any] | None = None
    quote: str | None = None


class AnswerMetadataResponse(BaseModel):
    """Reproducibility metadata recorded with every answer (Stage 3 exit gate)."""

    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    retrieved_chunk_ids: list[uuid.UUID] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    evidence_tokens: int = 0
    degradations: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """A grounded answer with citations and support state."""

    query: str
    answer: str
    support: str = Field(description="grounded | partial | insufficient")
    abstained: bool
    citations: list[CitationResponse] = Field(default_factory=list)
    metadata: AnswerMetadataResponse


class SearchRequest(BaseModel):
    """Raw retrieval request for debugging and evaluation (no generation)."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: MetadataFilterInput | None = None


class SearchHitResponse(BaseModel):
    """One retrieved chunk with its score and provenance."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    version_id: uuid.UUID
    document_title: str | None = None
    score: float
    rank: int
    channel: str
    chunk_index: int
    chunk_type: str
    token_count: int
    page_number: int
    page_span: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    bounding_box: dict[str, Any] | None = None
    content: str


class SearchResponse(BaseModel):
    """Raw retrieval results plus the configuration that produced them."""

    query: str
    hits: list[SearchHitResponse] = Field(default_factory=list)
    total_candidates: int = 0
    latency_ms: float = 0.0
    embedding_version: str | None = None
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
