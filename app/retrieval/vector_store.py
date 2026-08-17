"""Qdrant vector store wrapper (Task 3.2, ADR-007).

Qdrant is a *derived* store (ADR-002): everything here can be rebuilt from
PostgreSQL, and nothing here is authoritative. Point IDs are deterministic so a
repeated index run upserts in place instead of creating duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.schemas import ChunkPayload, RetrievalFilters

logger = get_logger("app.retrieval.vector_store")

# Payload fields that need an index to be filterable at scale. Keyword indexes
# are created up front, including the Stage 8 ACL fields, so enabling ACL
# filtering later is a query change and not a re-index.
_KEYWORD_INDEX_FIELDS = (
    "document_id",
    "version_id",
    "chunking_version",
    "embedding_version",
    "department",
    "policy_type",
    "policy_status",
    "country",
    "employee_type",
    "grade",
    "confidentiality",
    "audience",
    "tenant_id",
    "department_id",
    "allowed_roles",
    "allowed_users",
    "classification",
)
_INTEGER_INDEX_FIELDS = ("page_number", "chunk_index")


@dataclass
class VectorPoint:
    """A vector plus its filterable payload, ready to upsert."""

    point_id: str
    vector: list[float]
    payload: ChunkPayload


@dataclass
class VectorHit:
    """A scored match returned by the vector store."""

    point_id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    """Thin, testable wrapper over the Qdrant collection holding chunk vectors."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = self.settings.QDRANT_COLLECTION_NAME
        self._client = client

    @property
    def client(self) -> QdrantClient:
        """Lazily construct the Qdrant client.

        Construction is deferred so importing this module (and therefore the app)
        never requires a reachable Qdrant.
        """
        if self._client is None:
            self._client = QdrantClient(
                host=self.settings.QDRANT_HOST,
                port=self.settings.QDRANT_PORT,
                api_key=self.settings.QDRANT_API_KEY or None,
                timeout=30,
            )
        return self._client

    def ensure_collection(self, dimensions: int | None = None) -> bool:
        """Create the collection and payload indexes if absent. Returns True if created."""
        size = dimensions or self.settings.EMBEDDING_DIMENSIONS

        if self.client.collection_exists(self.collection_name):
            logger.debug("qdrant_collection_present", collection=self.collection_name)
            return False

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        for field_name in _KEYWORD_INDEX_FIELDS:
            self._create_index(field_name, qmodels.PayloadSchemaType.KEYWORD)
        for field_name in _INTEGER_INDEX_FIELDS:
            self._create_index(field_name, qmodels.PayloadSchemaType.INTEGER)

        logger.info(
            "qdrant_collection_created",
            collection=self.collection_name,
            dimensions=size,
        )
        return True

    def _create_index(self, field_name: str, schema: Any) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:  # noqa: BLE001 - index may already exist
            logger.debug("qdrant_payload_index_skipped", field=field_name, error=str(exc))

    def upsert(self, points: list[VectorPoint]) -> int:
        """Upsert points by deterministic ID. Repeated calls create no duplicates."""
        if not points:
            return 0

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=point.point_id,
                    vector=point.vector,
                    payload=point.payload.model_dump(mode="json"),
                )
                for point in points
            ],
            wait=True,
        )
        logger.info("qdrant_upsert_complete", points=len(points))
        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
        chunking_version: str | None = None,
        embedding_version: str | None = None,
    ) -> list[VectorHit]:
        """Run a filtered vector search. Filters are applied inside the query."""
        query_filter = self.build_filter(
            filters=filters,
            chunking_version=chunking_version,
            embedding_version=embedding_version,
        )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=min_score,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001 - client raises a bare UnexpectedResponse
            # An absent collection means nothing has been indexed yet. That is a
            # normal state for a fresh deployment, not an internal error: the
            # caller should see zero results and abstain, not a 500.
            if self._is_missing_collection(exc):
                logger.warning(
                    "qdrant_collection_absent_returning_no_results",
                    collection=self.collection_name,
                )
                return []
            raise

        return [
            VectorHit(
                point_id=str(point.id),
                score=float(point.score or 0.0),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    @staticmethod
    def _is_missing_collection(exc: Exception) -> bool:
        """Detect Qdrant's 'collection doesn't exist' response.

        The client raises a generic `UnexpectedResponse` rather than a typed error,
        so the 404 plus the message is the only reliable signal.
        """
        message = str(exc).lower()
        return "doesn't exist" in message or ("404" in message and "collection" in message)

    @staticmethod
    def build_filter(
        filters: RetrievalFilters | None = None,
        chunking_version: str | None = None,
        embedding_version: str | None = None,
    ) -> qmodels.Filter | None:
        """Translate metadata constraints into a Qdrant filter."""
        conditions: list[qmodels.Condition] = []

        def match_any(field_name: str, values: list[str]) -> None:
            conditions.append(
                qmodels.FieldCondition(
                    key=field_name,
                    match=qmodels.MatchAny(any=values),
                )
            )

        def match_value(field_name: str, value: str | int) -> None:
            conditions.append(
                qmodels.FieldCondition(
                    key=field_name,
                    match=qmodels.MatchValue(value=value),
                )
            )

        if chunking_version:
            match_value("chunking_version", chunking_version)
        if embedding_version:
            match_value("embedding_version", embedding_version)

        if filters is not None:
            if filters.document_ids:
                match_any("document_id", [str(v) for v in filters.document_ids])
            if filters.version_ids:
                match_any("version_id", [str(v) for v in filters.version_ids])
            for field_name in (
                "department",
                "policy_type",
                "policy_status",
                "country",
                "employee_type",
                "grade",
            ):
                value = getattr(filters, field_name)
                if value:
                    match_value(field_name, value)
            if filters.page_number is not None:
                match_value("page_number", filters.page_number)

        return qmodels.Filter(must=conditions) if conditions else None

    def count(self, exact: bool = True) -> int:
        """Count points currently in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(self.collection_name, exact=exact).count)

    def delete_by_version(self, version_id: str) -> None:
        """Remove all points belonging to a document version."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="version_id",
                            match=qmodels.MatchValue(value=version_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("qdrant_version_points_deleted", version_id=version_id)

    def health_check(self) -> bool:
        """Return True when Qdrant answers a trivial request."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001 - health probe must not raise
            logger.warning("qdrant_health_check_failed", error=str(exc))
            return False


_vector_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """Return the singleton QdrantVectorStore."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store
