"""Cross-encoder reranking over the retrieved pool (Task 6.5)."""

from __future__ import annotations

import uuid
from typing import Any

from app.core.config import AppSettings
from app.models.schemas import ModelMetadata, RerankResult, ScoredDocument
from app.retrieval.channels import get_retriever
from app.retrieval.fusion import FusionRetriever
from app.retrieval.narrowing import NarrowingRetriever
from app.retrieval.reranking import RerankingRetriever
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk


def chunk(text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        content=text,
        score=10.0 - rank,
        rank=rank,
    )


class _Inner:
    channel = "sparse"

    def __init__(self, texts: list[str]) -> None:
        self.settings = AppSettings(APP_ENV="testing", RETRIEVAL_TOP_K=3, RERANK_CANDIDATES=5)
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        return 42

    async def retrieve(
        self,
        query: str,
        session: Any = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        self.calls.append({"top_k": top_k, "filters": filters})
        chunks = [chunk(t, i) for i, t in enumerate(self.texts[:top_k], start=1)]
        return RetrievalResult(
            query=query, chunks=chunks, latency_ms=100.0, retrieval_config={"channels": ["sparse"]}
        )


class _Gateway:
    """Reverses the order it is given, as an unmistakable reranking."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None, **_: Any
    ) -> RerankResult:
        self.calls.append({"documents": documents, "top_k": top_k})
        if self.fail:
            raise RuntimeError("jina 429")
        order = list(reversed(range(len(documents))))[:top_k]
        return RerankResult(
            results=[
                ScoredDocument(index=i, text=documents[i], score=1.0 - n / 10)
                for n, i in enumerate(order)
            ],
            metadata=ModelMetadata(provider="fake", model_name="fake-reranker", latency_ms=5.0),
        )


def reranker(inner: _Inner, gateway: _Gateway) -> RerankingRetriever:
    return RerankingRetriever(inner, gateway=gateway, settings=inner.settings)  # type: ignore[arg-type]


class TestReranking:
    async def test_a_wider_pool_is_reranked_down_to_top_k(self) -> None:
        inner, gateway = _Inner(list("abcdefg")), _Gateway()

        result = await reranker(inner, gateway).retrieve("q")

        assert inner.calls[0]["top_k"] == 5  # RERANK_CANDIDATES, not top K
        assert gateway.calls[0] == {"documents": list("abcde"), "top_k": 3}
        assert [c.content for c in result.chunks] == ["e", "d", "c"]
        assert [c.rank for c in result.chunks] == [1, 2, 3]

    async def test_each_chunk_keeps_its_retrieval_rank_and_score(self) -> None:
        result = await reranker(_Inner(list("abcde")), _Gateway()).retrieve("q")
        top = result.chunks[0]

        assert top.metadata["retrieval_rank"] == 5
        assert top.metadata["retrieval_score"] == 5.0
        assert top.score == 1.0  # the reranker's score replaces the channel's

    async def test_the_trace_records_model_latency_and_moves(self) -> None:
        result = await reranker(_Inner(list("abcde")), _Gateway()).retrieve("q")
        trace = result.retrieval_config["reranking"]

        assert trace["status"] == "ok" and trace["model"] == "fake-reranker"
        assert trace["moves"] == [[5, 1], [4, 2], [3, 3]]
        assert trace["candidates"] == 5
        assert result.retrieval_config["channels"] == ["sparse"]  # inner config kept

    async def test_a_reranker_failure_keeps_the_retrieval_order(self) -> None:
        result = await reranker(_Inner(list("abcde")), _Gateway(fail=True)).retrieve("q")

        assert [c.content for c in result.chunks] == ["a", "b", "c"]
        trace = result.retrieval_config["reranking"]
        assert trace["status"] == "failed" and "429" in trace["error"]

    async def test_one_candidate_needs_no_reranker_call(self) -> None:
        gateway = _Gateway()
        result = await reranker(_Inner(["only"]), gateway).retrieve("q")

        assert gateway.calls == []
        assert result.retrieval_config["reranking"]["status"] == "skipped"

    async def test_filters_pass_through_to_the_inner_retriever(self) -> None:
        inner = _Inner(list("abc"))
        filters = RetrievalFilters(department="benefits")

        await reranker(inner, _Gateway()).retrieve("q", filters=filters)

        assert inner.calls[0]["filters"] is filters


class TestWiring:
    def test_the_flag_cleanly_disables_reranking(self) -> None:
        assert not isinstance(get_retriever(AppSettings(APP_ENV="testing")), RerankingRetriever)

    def test_reranking_sits_between_narrowing_and_the_channel(self) -> None:
        """Narrowing's filters must shape the pool the reranker draws from."""
        retriever = get_retriever(
            AppSettings(
                APP_ENV="testing",
                RETRIEVAL_MODE="hybrid",
                ENABLE_RERANKING=True,
                ENABLE_METADATA_NARROWING=True,
                INFERENCE_PROFILE="stub",
            )
        )

        assert isinstance(retriever, NarrowingRetriever)
        assert isinstance(retriever.inner, RerankingRetriever)
        assert isinstance(retriever.inner.inner, FusionRetriever)
        assert retriever.channel == "hybrid"
