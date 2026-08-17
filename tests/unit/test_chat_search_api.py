"""Unit tests for the chat and search endpoints (Task 3.6, ADR-035)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Generator

import pytest
from starlette.testclient import TestClient

from app.db.session import get_db_session
from app.generation.citation import SupportState
from app.generation.service import AnswerResult, get_generation_service
from app.main import app
from app.retrieval.dense import get_dense_retriever
from app.retrieval.schemas import Citation, RetrievalResult, RetrievedChunk

DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
VER_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
CHUNK_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def make_citation() -> Citation:
    return Citation(
        marker="1",
        document_id=DOC_ID,
        version_id=VER_ID,
        chunk_id=CHUNK_ID,
        document_title="Staff Handbook",
        version_number=1,
        page_number=14,
        section_path=["Leave Policy", "Annual Leave"],
        element_ids=["p1"],
        bounding_box={"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "page_number": 14, "unit": "pt"},
        quote="Employees receive 21 days of annual leave.",
    )


def make_answer(**overrides: object) -> AnswerResult:
    defaults: dict[str, object] = {
        "query": "How many annual leave days?",
        "answer": "Employees receive 21 days of annual leave [1].",
        "support": SupportState.GROUNDED,
        "citations": [make_citation()],
        "abstained": False,
        "retrieved_chunk_ids": [CHUNK_ID],
        "provider": "fake",
        "model_name": "fake-llm",
        "model_version": "v1",
        "prompt_versions": {"answer": "answer_v1", "citation": "citation_v1"},
        "prompt_hashes": {"answer": "f" * 64},
        "retrieval_config": {"top_k": 8, "channels": ["dense"]},
        "token_counts": {"total_tokens": 230},
        "retrieval_latency_ms": 12.345,
        "generation_latency_ms": 25.0,
        "total_latency_ms": 40.0,
        "evidence_tokens": 300,
    }
    defaults.update(overrides)
    return AnswerResult(**defaults)  # type: ignore[arg-type]


class StubGenerationService:
    """Stands in for GenerationService at the HTTP boundary."""

    def __init__(self, result: AnswerResult | None = None, error: Exception | None = None) -> None:
        self.result = result or make_answer()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def answer(self, query: str, session: object, top_k: int | None = None, filters: object = None) -> AnswerResult:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        if self.error:
            raise self.error
        return self.result

    async def stream_answer(
        self, query: str, session: object, top_k: int | None = None, filters: object = None
    ):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters, "stream": True})
        if self.error:
            raise self.error
        result = self.result
        yield ("metadata", {"support": str(result.support), "abstained": result.abstained})
        yield ("token", {"text": result.answer})
        yield (
            "citations",
            {"citations": [c.model_dump(mode="json") for c in result.citations]},
        )
        yield ("done", {"support": str(result.support), "citation_count": len(result.citations)})


class StubRetriever:
    """Stands in for DenseRetriever at the HTTP boundary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        session: object = None,
        top_k: int | None = None,
        filters: object = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        self.calls.append({"query": query, "top_k": top_k, "min_score": min_score, "filters": filters})
        return RetrievalResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    chunk_id=CHUNK_ID,
                    document_id=DOC_ID,
                    version_id=VER_ID,
                    content="Employees receive 21 days of annual leave.",
                    score=0.91,
                    rank=1,
                    chunk_index=3,
                    page_number=14,
                    section_path=["Leave Policy", "Annual Leave"],
                    element_ids=["p1"],
                    document_title="Staff Handbook",
                    version_number=1,
                )
            ],
            total_candidates=1,
            latency_ms=9.5,
            embedding_version="test-embed-v1",
            retrieval_config={"top_k": 8, "channels": ["dense"]},
        )


async def _fake_session() -> AsyncGenerator[None, None]:
    yield None


@pytest.fixture
def generation_stub() -> Generator[StubGenerationService, None, None]:
    stub = StubGenerationService()
    app.dependency_overrides[get_generation_service] = lambda: stub
    app.dependency_overrides[get_db_session] = _fake_session
    yield stub
    app.dependency_overrides.clear()


@pytest.fixture
def retriever_stub() -> Generator[StubRetriever, None, None]:
    stub = StubRetriever()
    app.dependency_overrides[get_dense_retriever] = lambda: stub
    app.dependency_overrides[get_db_session] = _fake_session
    yield stub
    app.dependency_overrides.clear()


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


class TestChatEndpoint:
    def test_returns_grounded_answer_with_citations(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        response = api_client.post("/api/v1/chat", json={"query": "How many annual leave days?"})

        assert response.status_code == 200
        data = response.json()
        assert data["support"] == "grounded"
        assert data["abstained"] is False
        assert len(data["citations"]) == 1

        citation = data["citations"][0]
        assert citation["marker"] == "1"
        assert citation["page_number"] == 14
        assert citation["section_path"] == ["Leave Policy", "Annual Leave"]
        assert citation["element_ids"] == ["p1"]
        assert citation["bounding_box"] is not None

    def test_response_records_reproducibility_metadata(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        """Stage 3 exit gate: model_version, prompt_version, config, chunk IDs."""
        data = api_client.post("/api/v1/chat", json={"query": "q"}).json()
        metadata = data["metadata"]

        assert metadata["model_name"] == "fake-llm"
        assert metadata["model_version"] == "v1"
        assert metadata["prompt_versions"]["answer"] == "answer_v1"
        assert metadata["retrieval_config"]["channels"] == ["dense"]
        assert metadata["retrieved_chunk_ids"] == [str(CHUNK_ID)]
        assert metadata["token_counts"]["total_tokens"] == 230

    def test_abstention_is_reported_without_citations(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        generation_stub.result = make_answer(
            answer="The HR knowledge base does not cover housing.",
            support=SupportState.INSUFFICIENT,
            citations=[],
            abstained=True,
        )
        data = api_client.post(
            "/api/v1/chat", json={"query": "Does the company provide housing?"}
        ).json()

        assert data["support"] == "insufficient"
        assert data["abstained"] is True
        assert data["citations"] == []

    def test_top_k_and_filters_are_forwarded(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        api_client.post(
            "/api/v1/chat",
            json={
                "query": "maternity leave",
                "top_k": 3,
                "filters": {"department": "Human Resources", "policy_type": "leave"},
            },
        )
        call = generation_stub.calls[-1]

        assert call["top_k"] == 3
        assert call["filters"].department == "Human Resources"  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "payload",
        [{}, {"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 999}],
    )
    def test_invalid_requests_are_rejected(
        self, api_client: TestClient, generation_stub: StubGenerationService, payload: dict
    ) -> None:
        assert api_client.post("/api/v1/chat", json=payload).status_code == 422


class TestChatStreaming:
    def test_stream_emits_ordered_sse_events(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        with api_client.stream(
            "POST", "/api/v1/chat", json={"query": "annual leave", "stream": True}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

        events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
        assert events == ["metadata", "token", "citations", "done"]

    def test_stream_payloads_are_valid_json(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        with api_client.stream(
            "POST", "/api/v1/chat", json={"query": "annual leave", "stream": True}
        ) as response:
            body = "".join(response.iter_text())

        payloads = [
            json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
        ]
        assert len(payloads) == 4
        assert payloads[1]["text"] == "Employees receive 21 days of annual leave [1]."
        assert payloads[2]["citations"][0]["marker"] == "1"
        assert payloads[3]["citation_count"] == 1

    def test_streaming_matches_non_streaming_content(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        direct = api_client.post("/api/v1/chat", json={"query": "annual leave"}).json()

        with api_client.stream(
            "POST", "/api/v1/chat", json={"query": "annual leave", "stream": True}
        ) as response:
            body = "".join(response.iter_text())

        payloads = [
            json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
        ]
        assert payloads[1]["text"] == direct["answer"]
        assert len(payloads[2]["citations"]) == len(direct["citations"])

    def test_mid_stream_failure_becomes_a_terminal_error_event(
        self, api_client: TestClient, generation_stub: StubGenerationService
    ) -> None:
        """Headers are already sent, so the failure must arrive as an event."""
        generation_stub.error = RuntimeError("provider exploded")

        with api_client.stream(
            "POST", "/api/v1/chat", json={"query": "q", "stream": True}
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

        assert "event: error" in body
        assert "GENERATION_ERROR" in body
        # The internal error text must not leak to the client.
        assert "provider exploded" not in body


class TestSearchEndpoint:
    def test_returns_raw_hits_with_provenance(
        self, api_client: TestClient, retriever_stub: StubRetriever
    ) -> None:
        response = api_client.post("/api/v1/search", json={"query": "annual leave"})

        assert response.status_code == 200
        data = response.json()
        assert data["total_candidates"] == 1
        assert data["embedding_version"] == "test-embed-v1"

        hit = data["hits"][0]
        assert hit["chunk_id"] == str(CHUNK_ID)
        assert hit["score"] == pytest.approx(0.91)
        assert hit["rank"] == 1
        assert hit["channel"] == "dense"
        assert hit["page_number"] == 14
        assert hit["section_path"] == ["Leave Policy", "Annual Leave"]
        assert hit["content"]

    def test_search_forwards_overrides(
        self, api_client: TestClient, retriever_stub: StubRetriever
    ) -> None:
        api_client.post(
            "/api/v1/search",
            json={"query": "clause 4.2", "top_k": 20, "min_score": 0.4},
        )
        call = retriever_stub.calls[-1]

        assert call["top_k"] == 20
        assert call["min_score"] == pytest.approx(0.4)

    def test_search_exposes_retrieval_config(
        self, api_client: TestClient, retriever_stub: StubRetriever
    ) -> None:
        data = api_client.post("/api/v1/search", json={"query": "q"}).json()
        assert data["retrieval_config"]["channels"] == ["dense"]

    @pytest.mark.parametrize(
        "payload", [{}, {"query": ""}, {"query": "x", "min_score": 1.5}, {"query": "x", "top_k": 0}]
    )
    def test_invalid_search_requests_are_rejected(
        self, api_client: TestClient, retriever_stub: StubRetriever, payload: dict
    ) -> None:
        assert api_client.post("/api/v1/search", json=payload).status_code == 422
