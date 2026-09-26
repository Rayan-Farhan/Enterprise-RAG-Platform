"""Late-interaction reordering (Task 6.6): a tested capability that ships off."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.core.exceptions import ModelProviderException
from app.models.gateway import LocalModelGateway, StubModelGateway
from app.models.schemas import ModelMetadata, MultiVectorEmbedding, MultiVectorResponse
from app.retrieval.channels import get_retriever
from app.retrieval.late_interaction import LateInteractionRetriever
from app.retrieval.narrowing import NarrowingRetriever
from app.retrieval.reranking import RerankingRetriever
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk

SETTINGS = AppSettings(APP_ENV="testing", RETRIEVAL_TOP_K=2, LATE_INTERACTION_CANDIDATES=4)


def chunk(text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, text),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        content=text,
        score=1.0,
        rank=rank,
    )


class _Inner:
    channel = "sparse"
    settings = SETTINGS

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[int | None] = []

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        return 7

    async def retrieve(
        self, query: str, session: Any = None, top_k: int | None = None, *_: Any, **__: Any
    ) -> RetrievalResult:
        self.calls.append(top_k)
        return RetrievalResult(
            query=query, chunks=[chunk(t, i) for i, t in enumerate(self.texts[:top_k], start=1)]
        )


class _Store:
    """Scores points by a fixed table; can pretend some are not indexed."""

    def __init__(self, scores: dict[str, float], unindexed: set[str] | None = None) -> None:
        self.scores = scores
        self.unindexed = unindexed or set()
        self.queried: list[str] = []

    def point_id(self, chunk_id: uuid.UUID) -> str:
        return str(chunk_id)

    def existing_ids(self, point_ids: list[str]) -> set[str]:
        names = {str(uuid.uuid5(uuid.NAMESPACE_URL, n)): n for n in self.scores}
        return {pid for pid in point_ids if names.get(pid) not in self.unindexed}

    def maxsim(
        self, query_vectors: Any, point_ids: list[str], limit: int
    ) -> list[tuple[str, float]]:
        self.queried = point_ids
        names = {str(uuid.uuid5(uuid.NAMESPACE_URL, n)): n for n in self.scores}
        ranked = sorted(point_ids, key=lambda pid: -self.scores[names[pid]])
        return [(pid, self.scores[names[pid]]) for pid in ranked[:limit]]


class _Gateway:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.input_types: list[str] = []

    async def embed_multivector(self, texts: list[str], input_type: str) -> MultiVectorResponse:
        self.input_types.append(input_type)
        if self.fail:
            raise RuntimeError("jina 503")
        return MultiVectorResponse(
            embeddings=[MultiVectorEmbedding(vectors=[[1.0, 0.0]], index=0)],
            dimensions=2,
            metadata=ModelMetadata(provider="fake", model_name="fake-colbert", latency_ms=1.0),
        )


def wrap(texts: list[str], store: _Store, gateway: _Gateway) -> LateInteractionRetriever:
    return LateInteractionRetriever(
        _Inner(texts),  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        settings=SETTINGS,
    )


class TestLateInteractionRetriever:
    async def test_the_pool_is_reordered_by_maxsim_among_its_own_points(self) -> None:
        store = _Store({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2})
        gateway = _Gateway()
        retriever = wrap(list("abcdef"), store, gateway)

        result = await retriever.retrieve("q")

        assert retriever.inner.calls == [4]  # type: ignore[attr-defined]
        assert len(store.queried) == 4  # scored among the candidates only
        assert [c.content for c in result.chunks] == ["b", "c"]
        assert result.chunks[0].metadata["pre_late_interaction_rank"] == 2
        assert gateway.input_types == ["query"]  # ColBERT marks queries differently
        trace = result.retrieval_config["late_interaction"]
        assert trace["status"] == "ok" and trace["moves"] == [[2, 1], [3, 2]]

    async def test_an_incompletely_indexed_pool_keeps_retrieval_order(self) -> None:
        """Mixing MaxSim-scored and unscored chunks would rank nothing meaningfully."""
        store = _Store({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2}, unindexed={"c"})
        gateway = _Gateway()

        result = await wrap(list("abcd"), store, gateway).retrieve("q")

        assert [c.content for c in result.chunks] == ["a", "b"]
        trace = result.retrieval_config["late_interaction"]
        assert trace["status"] == "incomplete_index" and trace["unindexed"] == 1
        assert gateway.input_types == []  # no model call for a pool it cannot score

    async def test_an_encoder_failure_keeps_retrieval_order(self) -> None:
        store = _Store({"a": 0.1, "b": 0.9})
        result = await wrap(["a", "b"], store, _Gateway(fail=True)).retrieve("q")

        assert [c.content for c in result.chunks] == ["a", "b"]
        assert result.retrieval_config["late_interaction"]["status"] == "failed"


class TestCapabilityIsOff:
    def test_disabled_by_default(self) -> None:
        settings = AppSettings(APP_ENV="testing")
        assert settings.ENABLE_LATE_INTERACTION is False
        assert not isinstance(get_retriever(settings), LateInteractionRetriever)

    def test_it_reorders_the_channel_pool_before_any_reranker(self) -> None:
        """Blueprint §17: candidate pool -> late interaction -> final ranking."""
        retriever = get_retriever(
            AppSettings(
                APP_ENV="testing",
                INFERENCE_PROFILE="stub",
                ENABLE_METADATA_NARROWING=True,
                ENABLE_RERANKING=True,
                ENABLE_LATE_INTERACTION=True,
            )
        )
        assert isinstance(retriever, NarrowingRetriever)
        assert isinstance(retriever.inner, RerankingRetriever)
        assert isinstance(retriever.inner.inner, LateInteractionRetriever)

    async def test_the_local_profile_fails_loudly_rather_than_skipping(self) -> None:
        gateway = LocalModelGateway(AppSettings(APP_ENV="testing"))
        with pytest.raises(ModelProviderException, match="ADR-012"):
            await gateway.embed_multivector(["q"], input_type="query")

    async def test_the_stub_profile_serves_token_matrices(self) -> None:
        gateway = StubModelGateway(AppSettings(APP_ENV="testing"))
        result = await gateway.embed_multivector(["annual leave days"], input_type="document")

        assert len(result.embeddings[0].vectors) == 3
        assert len(result.embeddings[0].vectors[0]) == 128
