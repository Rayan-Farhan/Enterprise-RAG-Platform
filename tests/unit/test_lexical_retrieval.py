"""BM25 channel: request building, index projection, and channel selection (Task 6.1)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.retrieval.channels import get_retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.lexical_store import (
    INDEX_SETTINGS,
    LexicalHit,
    build_filter,
    build_query,
    to_document,
)
from app.retrieval.schemas import ChunkPayload, RetrievalFilters

DOC_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


def payload(**overrides: Any) -> ChunkPayload:
    base: dict[str, Any] = {
        "chunk_id": str(uuid.uuid4()),
        "document_id": str(DOC_ID),
        "version_id": str(VERSION_ID),
        "chunk_index": 0,
        "chunking_version": "contextual-s256-o32",
        "embedding_version": "jina-embeddings-v3",
        "content": "Ala. Code § 31-2-13 grants 21 days of paid military leave.",
        "element_ids": ["e1", "e2"],
        "department": "HR",
    }
    base.update(overrides)
    return ChunkPayload(**base)


class TestIndexProjection:
    def test_only_mapped_fields_are_sent(self) -> None:
        """The mapping is strict: an unmapped field would fail the whole bulk request."""
        document = to_document(payload())
        mapped = set(INDEX_SETTINGS["mappings"]["properties"])

        assert set(document) <= mapped
        assert document["content"].startswith("Ala. Code")
        assert "element_ids" not in document  # rehydrated from PostgreSQL instead
        assert "embedding_version" not in document  # BM25 has no embedding

    def test_acl_placeholders_are_indexed_from_day_one(self) -> None:
        """Stage 8 enforces them; mapping them now avoids a re-index then."""
        mapped = INDEX_SETTINGS["mappings"]["properties"]
        for field in ("tenant_id", "allowed_roles", "allowed_users", "classification"):
            assert mapped[field] == {"type": "keyword"}

    def test_content_is_analyzed_as_prose_and_as_codes(self) -> None:
        content = INDEX_SETTINGS["mappings"]["properties"]["content"]
        assert content["analyzer"] == "hr_text"
        assert content["fields"]["exact"]["analyzer"] == "hr_exact"


class TestQueryBuilding:
    def test_scores_prose_and_code_fields_together(self) -> None:
        body = build_query("Ala. Code 31-2-13", limit=8)
        should = body["query"]["bool"]["should"]

        assert body["size"] == 8
        assert should[0]["multi_match"]["fields"] == ["content", "content.exact"]
        assert should[0]["multi_match"]["type"] == "most_fields"
        assert should[1]["match_phrase"]["content"]["query"] == "Ala. Code 31-2-13"

    def test_filters_narrow_in_filter_context_not_scoring(self) -> None:
        """Master §13: metadata narrows the candidate pool; it never changes a score."""
        filters = RetrievalFilters(document_ids=[DOC_ID], department="HR", page_number=4)
        body = build_query("leave", limit=5, filters=filters, chunking_version="v1")
        clauses = body["query"]["bool"]["filter"]

        assert {"term": {"chunking_version": "v1"}} in clauses
        assert {"terms": {"document_id": [str(DOC_ID)]}} in clauses
        assert {"term": {"department": "HR"}} in clauses
        assert {"term": {"page_number": 4}} in clauses
        assert all("filter" not in clause for clause in body["query"]["bool"]["should"])

    def test_no_filters_means_no_clauses(self) -> None:
        assert build_filter() == []


class _FakeStore:
    def __init__(self, hits: list[LexicalHit]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[LexicalHit]:
        self.calls.append(kwargs)
        return self.hits


class TestLexicalRetriever:
    async def test_debug_path_builds_results_from_the_index_document(self) -> None:
        chunk_id = str(uuid.uuid4())
        store = _FakeStore(
            [
                LexicalHit(
                    chunk_id=chunk_id,
                    score=12.5,
                    payload={
                        "document_id": str(DOC_ID),
                        "version_id": str(VERSION_ID),
                        "content": "Section 2.9 Due Process Procedures",
                        "page_number": 7,
                    },
                )
            ]
        )
        settings = AppSettings(APP_ENV="testing", RETRIEVAL_TOP_K=4)
        retriever = LexicalRetriever(lexical_store=store, settings=settings)  # type: ignore[arg-type]

        result = await retriever.retrieve("section 2.9")

        assert store.calls[0]["limit"] == 4
        assert store.calls[0]["chunking_version"] == settings.CHUNKING_VERSION
        assert [str(c.chunk_id) for c in result.chunks] == [chunk_id]
        assert result.chunks[0].channel == "bm25"
        assert result.chunks[0].rank == 1
        assert result.retrieval_config["channels"] == ["bm25"]
        # A BM25 score has no fixed scale, so no cosine floor is recorded or applied.
        assert result.retrieval_config["min_score"] is None

    async def test_min_score_is_ignored_rather_than_misapplied(self) -> None:
        store = _FakeStore([])
        retriever = LexicalRetriever(
            lexical_store=store,  # type: ignore[arg-type]
            settings=AppSettings(APP_ENV="testing"),
        )
        await retriever.retrieve("q", min_score=0.9)

        assert "min_score" not in store.calls[0]


class TestChannelSelection:
    @pytest.mark.parametrize(
        ("mode", "expected"), [("dense", DenseRetriever), ("bm25", LexicalRetriever)]
    )
    def test_retrieval_mode_picks_the_channel(self, mode: str, expected: type) -> None:
        retriever = get_retriever(AppSettings(APP_ENV="testing", RETRIEVAL_MODE=mode))
        assert isinstance(retriever, expected)

    def test_neural_sparse_is_the_production_default(self) -> None:
        """Task 6.7's decision; see the RETRIEVAL_MODE comment in config."""
        settings = AppSettings(APP_ENV="testing")
        assert settings.RETRIEVAL_MODE == "sparse"
        assert settings.ENABLE_NEURAL_SPARSE is True
