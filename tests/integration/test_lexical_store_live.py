"""BM25 against a real OpenSearch (Task 6.1).

The analyzers are the substance of this channel, and only OpenSearch can run
them, so these tests need the Compose service. Each test gets its own throwaway
index. Locally they skip when OpenSearch is not reachable; CI sets
``RAG_REQUIRE_SERVICES=1`` and starts the service, so there a missing
OpenSearch is a failure rather than a silent skip.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from app.core.config import AppSettings
from app.retrieval.lexical_store import OpenSearchLexicalStore
from app.retrieval.schemas import ChunkPayload, RetrievalFilters

CHUNKING_VERSION = "contextual-s256-o32"
DOC_A = uuid.uuid4()
DOC_B = uuid.uuid4()

CORPUS = {
    "military": (
        DOC_A,
        "Document: policy_manual.pdf | Section: Leave > Military Leave\n\n"
        "Active members of the Alabama National Guard receive 21 days of paid "
        "military leave per calendar year under Ala. Code § 31-2-13 (1995).",
    ),
    "dental": (
        DOC_B,
        "Document: dental_booklet.pdf | Section: Contact\n\n"
        "Customer service: call 1-800-292-8868 for claims questions.",
    ),
    "due_process": (
        DOC_A,
        "Document: faculty_handbook.pdf | Section: 2.9 Due Process Procedures\n\n"
        "A faculty member may request review of a termination decision.",
    ),
    "leave_general": (
        DOC_A,
        "Document: policy_manual.pdf | Section: Leave > Annual Leave\n\n"
        "Employees accrue annual leave monthly. Late enrollees must wait 365 days.",
    ),
}


@pytest.fixture
def store() -> Iterator[OpenSearchLexicalStore]:
    settings = AppSettings(
        APP_ENV="testing", OPENSEARCH_INDEX_NAME=f"test_bm25_{uuid.uuid4().hex[:12]}"
    )
    candidate = OpenSearchLexicalStore(settings=settings)
    if not candidate.health_check():
        if os.getenv("RAG_REQUIRE_SERVICES") == "1":
            pytest.fail("OpenSearch is required (RAG_REQUIRE_SERVICES=1) but not reachable")
        pytest.skip("OpenSearch not reachable; start it with `make up`")
    candidate.ensure_index()
    yield candidate
    candidate.client.indices.delete(index=candidate.index_name, ignore=[404])


def payloads(chunking_version: str = CHUNKING_VERSION) -> list[ChunkPayload]:
    return [
        ChunkPayload(
            # Deterministic per (name, version), as real chunk IDs are.
            chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{chunking_version}")),
            document_id=str(document_id),
            version_id=str(document_id),
            chunk_index=index,
            chunking_version=chunking_version,
            embedding_version="none",
            content=content,
            department="HR" if name != "dental" else "Benefits",
        )
        for index, (name, (document_id, content)) in enumerate(CORPUS.items())
    ]


def chunk_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{CHUNKING_VERSION}"))


class TestAnalyzers:
    def test_codes_survive_as_single_tokens_on_the_exact_field(
        self, store: OpenSearchLexicalStore
    ) -> None:
        tokens = store.analyze("Ala. Code § 31-2-13, call 1-800-292-8868; Section 2.9.", "hr_exact")

        assert "31-2-13" in tokens
        assert "1-800-292-8868" in tokens
        assert "2.9" in tokens

    def test_prose_field_stems_and_drops_stop_words(self, store: OpenSearchLexicalStore) -> None:
        tokens = store.analyze("The late enrollees", "hr_text")

        assert "the" not in tokens
        assert "enrollee" in tokens

    def test_table_of_contents_dot_leaders_do_not_become_tokens(
        self, store: OpenSearchLexicalStore
    ) -> None:
        tokens = store.analyze("Calendar Year Deductible ............ 12", "hr_exact")

        assert tokens == ["calendar", "year", "deductible", "12"]


class TestSearch:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("What does Ala. Code 31-2-13 provide?", "military"),
            ("Who answers 1-800-292-8868?", "dental"),
            ("What does section 2.9 cover?", "due_process"),
        ],
    )
    def test_an_exact_code_query_ranks_its_chunk_first(
        self, store: OpenSearchLexicalStore, query: str, expected: str
    ) -> None:
        store.upsert(payloads())

        hits = store.search(query, limit=4, chunking_version=CHUNKING_VERSION)

        assert hits[0].chunk_id == chunk_id(expected)

    def test_reindexing_overwrites_instead_of_duplicating(
        self, store: OpenSearchLexicalStore
    ) -> None:
        store.upsert(payloads())
        store.upsert(payloads())

        assert store.count(chunking_version=CHUNKING_VERSION) == len(CORPUS)

    def test_chunking_versions_do_not_see_each_other(self, store: OpenSearchLexicalStore) -> None:
        store.upsert(payloads())
        store.upsert(payloads(chunking_version="fixed-v2"))

        hits = store.search("military leave", limit=10, chunking_version=CHUNKING_VERSION)

        assert store.count() == 2 * len(CORPUS)
        assert hits
        assert all(hit.payload["chunking_version"] == CHUNKING_VERSION for hit in hits)

    def test_metadata_filters_narrow_the_candidate_pool(
        self, store: OpenSearchLexicalStore
    ) -> None:
        store.upsert(payloads())

        by_doc = store.search(
            "leave",
            limit=10,
            filters=RetrievalFilters(document_ids=[DOC_B]),
            chunking_version=CHUNKING_VERSION,
        )
        by_department = store.search(
            "call claims",
            limit=10,
            filters=RetrievalFilters(department="HR"),
            chunking_version=CHUNKING_VERSION,
        )

        assert all(hit.payload["document_id"] == str(DOC_B) for hit in by_doc)
        assert chunk_id("dental") not in [hit.chunk_id for hit in by_department]

    def test_an_absent_index_returns_no_hits_not_an_error(
        self, store: OpenSearchLexicalStore
    ) -> None:
        store.client.indices.delete(index=store.index_name)

        assert store.search("anything", limit=5) == []

    def test_deleting_a_version_removes_its_chunks(self, store: OpenSearchLexicalStore) -> None:
        store.upsert(payloads())
        store.delete_by_version(str(DOC_B))

        assert store.count() == len(CORPUS) - 1


class TestMetadataNarrowingLive:
    """Task 6.3 against real OpenSearch: sync in place, count, and filter."""

    def test_updating_metadata_in_place_changes_what_filters_admit(
        self, store: OpenSearchLexicalStore
    ) -> None:
        store.upsert(payloads())
        # "military", "due_process" and "leave_general" share DOC_A as their version.
        updated = store.update_version_fields(
            str(DOC_A), {"department": "academic_affairs", "employee_type": "faculty"}
        )

        assert updated == 3
        assert (
            store.count(
                chunking_version=CHUNKING_VERSION,
                filters=RetrievalFilters(department="academic_affairs"),
            )
            == 3
        )

    def test_employee_type_filters_also_admit_documents_for_everyone(
        self, store: OpenSearchLexicalStore
    ) -> None:
        store.upsert(payloads())
        store.update_version_fields(str(DOC_A), {"employee_type": "faculty"})
        store.update_version_fields(str(DOC_B), {"employee_type": "all"})

        faculty = store.count(filters=RetrievalFilters(employee_type="faculty"))
        staff = store.count(filters=RetrievalFilters(employee_type="staff"))

        assert faculty == len(CORPUS)  # 3 faculty + 1 for everyone
        assert staff == 1  # only the one for everyone
