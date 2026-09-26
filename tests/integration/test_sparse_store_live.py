"""Neural sparse against a real OpenSearch with the models deployed (Task 6.2).

Needs `scripts/setup_neural_sparse.py` to have run: the encoder is ~500 MB and
the first deployment downloads PyTorch, so CI does not provision it. These
tests skip when the models are absent, unless ``RAG_REQUIRE_SPARSE_MODELS=1``.
Each test gets a throwaway index; the pipeline and models are shared.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from app.core.config import AppSettings
from app.retrieval.schemas import ChunkPayload
from app.retrieval.sparse_store import OpenSearchSparseStore, SparseModelNotReadyError

VERSION = "contextual-s256-o32"
TEXTS = {
    "dental": "Late enrollees must satisfy a 365 day benefit waiting period before "
    "major dental services are covered.",
    "military": "Members of the National Guard receive 21 days of paid military leave "
    "each calendar year.",
    "parking": "Parking permits are issued by the campus police department each August.",
}


@pytest.fixture
def store() -> Iterator[OpenSearchSparseStore]:
    settings = AppSettings(
        APP_ENV="testing", SPARSE_INDEX_NAME=f"test_sparse_{uuid.uuid4().hex[:12]}"
    )
    candidate = OpenSearchSparseStore(settings=settings)
    try:
        if not candidate.health_check():
            raise SparseModelNotReadyError("OpenSearch not reachable")
        candidate.ensure_index()
    except SparseModelNotReadyError as exc:
        if os.getenv("RAG_REQUIRE_SPARSE_MODELS") == "1":
            pytest.fail(f"Sparse models required but unavailable: {exc}")
        pytest.skip(f"Neural sparse not provisioned ({exc})")
    yield candidate
    candidate.client.indices.delete(index=candidate.index_name, ignore=[404])


def payloads() -> list[ChunkPayload]:
    return [
        ChunkPayload(
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
            document_id=str(uuid.uuid4()),
            version_id=str(uuid.uuid4()),
            chunk_index=index,
            chunking_version=VERSION,
            embedding_version="none",
            content=text,
        )
        for index, (name, text) in enumerate(TEXTS.items())
    ]


def chunk_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


class TestNeuralSparse:
    def test_the_pipeline_fills_the_sparse_field_on_index(
        self, store: OpenSearchSparseStore
    ) -> None:
        store.upsert(payloads())

        stored = store.client.get(index=store.index_name, id=chunk_id("dental"))["_source"]

        assert stored["content_sparse"]
        assert "dental" in stored["content_sparse"]

    def test_a_query_matches_on_terms_the_chunk_only_implies(
        self, store: OpenSearchSparseStore
    ) -> None:
        """Document expansion: none of these words appear in the dental chunk."""
        store.upsert(payloads())

        hits = store.search("dentist teeth", limit=3, chunking_version=VERSION)

        assert hits[0].chunk_id == chunk_id("dental")

    def test_existing_ids_lets_reindexing_skip_encoding(self, store: OpenSearchSparseStore) -> None:
        store.upsert(payloads()[:2])

        present = store.existing_ids([chunk_id(n) for n in TEXTS])

        assert present == {chunk_id("dental"), chunk_id("military")}
