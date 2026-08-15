"""Model Gateway package (ADR-046)."""

from app.models.gateway import (
    HostedModelGateway,
    LocalModelGateway,
    ModelGateway,
    get_model_gateway,
)
from app.models.schemas import (
    EmbeddingResult,
    EmbeddingsResponse,
    GenerationResult,
    ImagePayload,
    ModelMetadata,
    RerankResult,
    ScoredDocument,
    TokenCounts,
)

__all__ = [
    "ModelGateway",
    "HostedModelGateway",
    "LocalModelGateway",
    "get_model_gateway",
    "ModelMetadata",
    "TokenCounts",
    "GenerationResult",
    "EmbeddingResult",
    "EmbeddingsResponse",
    "ScoredDocument",
    "RerankResult",
    "ImagePayload",
]
