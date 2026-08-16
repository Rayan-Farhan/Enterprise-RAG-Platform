"""Document ingestion, parsing, canonical models, and chunking (ADR-004, ADR-005, ADR-006)."""

from app.ingestion.adapters.canonical_adapter import CanonicalAdapter
from app.ingestion.dedup import (
    BoilerplateDetector,
    compute_file_sha256,
    compute_simhash,
    simhash_similarity,
)
from app.ingestion.service import IngestionResult, IngestionService, get_ingestion_service

__all__ = [
    "CanonicalAdapter",
    "BoilerplateDetector",
    "compute_file_sha256",
    "compute_simhash",
    "simhash_similarity",
    "IngestionResult",
    "IngestionService",
    "get_ingestion_service",
]
