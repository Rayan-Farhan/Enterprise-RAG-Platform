"""Unit tests for the Qdrant vector store wrapper (Task 3.2, ADR-007)."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client.http import models as qmodels

from app.core.config import AppSettings
from app.retrieval.schemas import ChunkPayload, RetrievalFilters
from app.retrieval.vector_store import QdrantVectorStore, VectorPoint

DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
VER_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, clamped to [0, 1] to mimic Qdrant's COSINE distance."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return max(0.0, min(1.0, dot / norm)) if norm else 0.0


class FakeQdrantClient:
    """In-memory stand-in recording calls and simulating upsert-by-ID semantics."""

    def __init__(self, exists: bool = False) -> None:
        self._exists = exists
        self.points: dict[str, dict] = {}
        self.created_collections: list[str] = []
        self.payload_indexes: list[str] = []
        self.last_query: dict | None = None
        self.deleted_filters: list[object] = []

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.created_collections.append(collection_name)
        self._exists = True

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: object
    ) -> None:
        self.payload_indexes.append(field_name)

    def upsert(self, collection_name: str, points: list, wait: bool = True) -> None:
        for point in points:
            self.points[str(point.id)] = {"vector": point.vector, "payload": point.payload}

    def query_points(self, **kwargs: object) -> object:
        """Rank stored points by cosine similarity, honouring threshold and limit.

        Real ranking semantics matter here: a double that returned everything
        regardless of `limit` and `score_threshold` would make the retrieval tests
        pass without testing anything.
        """
        self.last_query = dict(kwargs)
        query_vector = list(kwargs.get("query") or [])
        threshold = kwargs.get("score_threshold")
        limit = int(kwargs.get("limit") or 10)

        class Point:
            def __init__(self, pid: str, score: float, payload: dict) -> None:
                self.id = pid
                self.score = score
                self.payload = payload

        class Response:
            def __init__(self, points: list) -> None:
                self.points = points

        scored: list[Point] = []
        for pid, data in self.points.items():
            score = _cosine(query_vector, data["vector"])
            if threshold is not None and score < float(threshold):
                continue
            scored.append(Point(pid, score, data["payload"]))

        scored.sort(key=lambda p: (-p.score, p.id))
        return Response(scored[:limit])

    def count(self, collection_name: str, exact: bool = True) -> object:
        class Result:
            def __init__(self, count: int) -> None:
                self.count = count

        return Result(len(self.points))

    def delete(self, collection_name: str, points_selector: object, wait: bool = True) -> None:
        self.deleted_filters.append(points_selector)

    def get_collections(self) -> object:
        return object()


def make_payload(chunk_id: uuid.UUID, index: int = 0) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=str(chunk_id),
        document_id=str(DOC_ID),
        version_id=str(VER_ID),
        chunk_index=index,
        chunking_version="fixed-v1",
        embedding_version="jina-v3",
        content=f"Chunk content {index}",
        page_number=index + 1,
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    settings = AppSettings(APP_ENV="testing", EMBEDDING_DIMENSIONS=4)
    return QdrantVectorStore(client=FakeQdrantClient(), settings=settings)


class TestCollectionBootstrap:
    def test_creates_collection_and_payload_indexes(self, store: QdrantVectorStore) -> None:
        assert store.ensure_collection(dimensions=8) is True

        client = store.client
        assert isinstance(client, FakeQdrantClient)
        assert client.created_collections == [store.collection_name]
        # ACL fields are indexed now so Stage 8 needs no re-index.
        for acl_field in ("tenant_id", "department_id", "allowed_roles", "classification"):
            assert acl_field in client.payload_indexes
        assert "page_number" in client.payload_indexes

    def test_existing_collection_is_not_recreated(self) -> None:
        store = QdrantVectorStore(
            client=FakeQdrantClient(exists=True),
            settings=AppSettings(APP_ENV="testing"),
        )
        assert store.ensure_collection() is False


class TestIdempotentUpsert:
    def test_repeated_upsert_of_same_point_id_creates_no_duplicates(
        self, store: QdrantVectorStore
    ) -> None:
        store.ensure_collection(dimensions=4)
        chunk_id = uuid.uuid4()
        point = VectorPoint(
            point_id="11111111-1111-1111-1111-111111111111",
            vector=[0.1, 0.2, 0.3, 0.4],
            payload=make_payload(chunk_id),
        )

        assert store.upsert([point]) == 1
        assert store.upsert([point]) == 1
        assert store.upsert([point]) == 1
        assert store.count() == 1

    def test_empty_upsert_is_a_noop(self, store: QdrantVectorStore) -> None:
        assert store.upsert([]) == 0
        assert store.count() == 0

    def test_payload_is_json_serialisable(self, store: QdrantVectorStore) -> None:
        point = VectorPoint(
            point_id="22222222-2222-2222-2222-222222222222",
            vector=[0.1] * 4,
            payload=make_payload(uuid.uuid4()),
        )
        store.upsert([point])
        stored = store.client.points["22222222-2222-2222-2222-222222222222"]["payload"]  # type: ignore[union-attr]

        assert isinstance(stored["document_id"], str)
        assert stored["chunking_version"] == "fixed-v1"
        assert stored["classification"] == "internal"


class TestFilterConstruction:
    def test_no_constraints_yields_no_filter(self) -> None:
        assert QdrantVectorStore.build_filter() is None
        assert QdrantVectorStore.build_filter(filters=RetrievalFilters()) is None

    def test_versions_are_always_constrained_when_supplied(self) -> None:
        built = QdrantVectorStore.build_filter(
            chunking_version="fixed-v1", embedding_version="jina-v3"
        )
        assert built is not None
        keys = {c.key for c in built.must}  # type: ignore[union-attr]
        assert keys == {"chunking_version", "embedding_version"}

    def test_metadata_filters_become_field_conditions(self) -> None:
        built = QdrantVectorStore.build_filter(
            filters=RetrievalFilters(department="HR", policy_type="leave", employee_type="full_time")
        )
        assert built is not None
        keys = {c.key for c in built.must}  # type: ignore[union-attr]
        assert keys == {"department", "policy_type", "employee_type"}

    def test_id_lists_become_match_any(self) -> None:
        built = QdrantVectorStore.build_filter(
            filters=RetrievalFilters(document_ids=[DOC_ID], version_ids=[VER_ID])
        )
        assert built is not None
        conditions = {c.key: c.match for c in built.must}  # type: ignore[union-attr]
        assert isinstance(conditions["document_id"], qmodels.MatchAny)
        assert conditions["document_id"].any == [str(DOC_ID)]

    def test_page_number_zero_is_not_dropped(self) -> None:
        """`page_number=0` must survive; a falsy-but-present value is still a filter."""
        built = QdrantVectorStore.build_filter(filters=RetrievalFilters(page_number=0))
        assert built is not None
        assert {c.key for c in built.must} == {"page_number"}  # type: ignore[union-attr]

    def test_empty_filters_report_themselves_empty(self) -> None:
        assert RetrievalFilters().is_empty()
        assert not RetrievalFilters(department="HR").is_empty()


class TestSearch:
    def test_search_pushes_filters_into_the_query(self, store: QdrantVectorStore) -> None:
        store.upsert(
            [
                VectorPoint(
                    point_id="33333333-3333-3333-3333-333333333333",
                    vector=[0.1] * 4,
                    payload=make_payload(uuid.uuid4()),
                )
            ]
        )

        hits = store.search(
            query_vector=[0.1] * 4,
            limit=5,
            filters=RetrievalFilters(department="HR"),
            min_score=0.3,
            chunking_version="fixed-v1",
            embedding_version="jina-v3",
        )

        query = store.client.last_query  # type: ignore[union-attr]
        assert query is not None
        # Pre-filtering, not post-filtering: the constraint is in the query itself.
        assert query["query_filter"] is not None
        assert query["score_threshold"] == 0.3
        assert query["limit"] == 5
        assert len(hits) == 1
        assert 0.0 <= hits[0].score <= 1.0

    def test_delete_by_version_filters_on_version_id(self, store: QdrantVectorStore) -> None:
        store.delete_by_version(str(VER_ID))
        assert len(store.client.deleted_filters) == 1  # type: ignore[union-attr]

    def test_health_check_true_when_reachable(self, store: QdrantVectorStore) -> None:
        assert store.health_check() is True

    def test_health_check_false_when_unreachable(self) -> None:
        class DeadClient(FakeQdrantClient):
            def get_collections(self) -> object:
                raise ConnectionError("connection refused")

        store = QdrantVectorStore(client=DeadClient(), settings=AppSettings(APP_ENV="testing"))
        assert store.health_check() is False

    def test_count_is_zero_when_collection_absent(self) -> None:
        store = QdrantVectorStore(
            client=FakeQdrantClient(exists=False), settings=AppSettings(APP_ENV="testing")
        )
        assert store.count() == 0
