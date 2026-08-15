"""Data schemas for Model Gateway requests and responses (ADR-046)."""

from typing import Any

from pydantic import BaseModel, Field


class TokenCounts(BaseModel):
    """Token usage counts for model executions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelMetadata(BaseModel):
    """Execution metadata returned with every model gateway call."""

    provider: str = Field(description="Inference provider (gemini, groq, jina, vllm, tei)")
    model_name: str = Field(description="Model identifier")
    model_version: str | None = Field(
        default=None, description="Explicit model version if applicable"
    )
    prompt_version: str | None = Field(
        default=None, description="Version of the prompt template used"
    )
    token_counts: TokenCounts = Field(default_factory=TokenCounts)
    latency_ms: float = Field(description="Execution latency in milliseconds")
    details: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Output from text or vision model generation."""

    text: str
    metadata: ModelMetadata
    finish_reason: str | None = "stop"


class EmbeddingResult(BaseModel):
    """Single dense vector embedding result."""

    embedding: list[float]
    index: int = 0


class EmbeddingsResponse(BaseModel):
    """Batched embeddings response."""

    embeddings: list[EmbeddingResult]
    metadata: ModelMetadata


class ScoredDocument(BaseModel):
    """A document scored by a reranker."""

    index: int
    text: str
    score: float


class RerankResult(BaseModel):
    """Reranking output containing ordered scored documents."""

    results: list[ScoredDocument]
    metadata: ModelMetadata


class ImagePayload(BaseModel):
    """Image representation for vision models."""

    image_bytes: bytes | None = None
    image_url: str | None = None
    mime_type: str = "image/png"
