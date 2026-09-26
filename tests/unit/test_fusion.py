"""Hybrid retrieval by rank fusion (Task 6.4)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.retrieval.channels import get_retriever
from app.retrieval.fusion import (
    FusionRetriever,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from app.retrieval.narrowing import NarrowingRetriever
from app.retrieval.schemas import RetrievalFilters, RetrievalResult, RetrievedChunk

IDS = {name: uuid.uuid5(uuid.NAMESPACE_URL, name) for name in "abcde"}


def hit(name: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=IDS[name],
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        content=name,
        score=score,
    )


def names(fused: list[Any]) -> list[str]:
    reverse = {v: k for k, v in IDS.items()}
    return [reverse[c.chunk_id] for c in fused]


class TestReciprocalRankFusion:
    def test_agreement_across_channels_beats_a_single_first_place(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "dense": [hit("a", 0.9), hit("b", 0.8)],
                "bm25": [hit("c", 30.0), hit("b", 20.0)],
                "sparse": [hit("d", 9.0), hit("b", 8.0)],
            },
            k=60,
            weights={},
        )

        assert names(fused)[0] == "b"  # second everywhere > first once
        assert fused[0].ranks == {"dense": 2, "bm25": 2, "sparse": 2}
        assert fused[0].score == pytest.approx(3 / 62)

    def test_scores_are_ignored_only_ranks_count(self) -> None:
        """Why RRF is the default: cosine and BM25 scales are incomparable."""
        small = reciprocal_rank_fusion({"x": [hit("a", 0.01), hit("b", 0.001)]}, 60, {})
        huge = reciprocal_rank_fusion({"x": [hit("a", 900.0), hit("b", 1.0)]}, 60, {})

        assert [c.score for c in small] == [c.score for c in huge]

    def test_weights_scale_a_channels_contribution(self) -> None:
        rankings = {"dense": [hit("a", 1.0)], "sparse": [hit("b", 1.0)]}

        even = reciprocal_rank_fusion(rankings, 60, {})
        sparse_heavy = reciprocal_rank_fusion(rankings, 60, {"sparse": 2.0})

        assert even[0].score == even[1].score
        assert names(sparse_heavy) == ["b", "a"]

    def test_k_controls_how_steeply_rank_is_discounted(self) -> None:
        rankings = {"x": [hit("a", 1), hit("b", 1)]}
        steep = reciprocal_rank_fusion(rankings, 1, {})
        flat = reciprocal_rank_fusion(rankings, 1000, {})

        assert steep[0].score / steep[1].score > flat[0].score / flat[1].score


class TestWeightedScoreFusion:
    def test_scores_are_normalised_within_each_channel(self) -> None:
        fused = weighted_score_fusion(
            {
                "bm25": [hit("a", 40.0), hit("b", 10.0)],
                "dense": [hit("b", 0.9), hit("a", 0.3)],
            },
            {},
        )

        # Each chunk tops one channel (1.0) and bottoms the other (0.0).
        assert [c.score for c in fused] == [1.0, 1.0]

    def test_a_single_hit_channel_does_not_divide_by_zero(self) -> None:
        assert weighted_score_fusion({"x": [hit("a", 5.0)]}, {})[0].score == 1.0


class _Channel:
    def __init__(self, name: str, hits: list[RetrievedChunk], fail: bool = False) -> None:
        self.channel = name
        self.settings = AppSettings(APP_ENV="testing")
        self.hits = hits
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        if self.fail:
            raise ConnectionError("down")
        return 1937

    async def retrieve(
        self,
        query: str,
        session: Any = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        self.calls.append({"session": session, "top_k": top_k, "filters": filters})
        if self.fail:
            raise ConnectionError(f"{self.channel} unreachable")
        return RetrievalResult(query=query, chunks=self.hits, latency_ms=5.0)


def fusion(*channels: _Channel, **settings: Any) -> FusionRetriever:
    return FusionRetriever(
        channels={c.channel: c for c in channels},  # type: ignore[misc]
        settings=AppSettings(APP_ENV="testing", RETRIEVAL_TOP_K=2, **settings),
    )


class TestFusionRetriever:
    async def test_every_channel_ranks_its_own_pool_then_fusion_cuts_to_top_k(self) -> None:
        dense = _Channel("dense", [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)])
        bm25 = _Channel("bm25", [hit("b", 12.0), hit("c", 9.0)])

        result = await fusion(dense, bm25, RETRIEVAL_CANDIDATE_LIMIT=50).retrieve("q")

        assert dense.calls[0]["top_k"] == 50 and bm25.calls[0]["top_k"] == 50
        assert dense.calls[0]["session"] is None  # only the fused top-K is hydrated
        # c (3rd and 2nd) outranks a (1st in one channel only): agreement wins.
        assert [c.content for c in result.chunks] == ["b", "c"]
        assert [c.rank for c in result.chunks] == [1, 2]
        assert all(c.channel == "hybrid" for c in result.chunks)
        assert result.chunks[0].metadata["fusion_ranks"] == {"dense": 2, "bm25": 1}

    async def test_filters_reach_every_channel_unchanged(self) -> None:
        dense, bm25 = _Channel("dense", [hit("a", 1)]), _Channel("bm25", [hit("a", 1)])
        filters = RetrievalFilters(policy_type="dental_benefits")

        await fusion(dense, bm25).retrieve("q", filters=filters)

        assert dense.calls[0]["filters"] is filters and bm25.calls[0]["filters"] is filters

    async def test_a_failed_channel_degrades_the_result_instead_of_failing_it(self) -> None:
        dense = _Channel("dense", [hit("a", 0.9), hit("b", 0.5)])
        sparse = _Channel("sparse", [], fail=True)

        result = await fusion(dense, sparse).retrieve("q")

        assert [c.content for c in result.chunks] == ["a", "b"]
        trace = result.retrieval_config["fusion"]["channels"]
        assert trace["dense"]["status"] == "ok"
        assert trace["sparse"]["status"] == "failed"
        assert "unreachable" in trace["sparse"]["error"]

    async def test_losing_every_channel_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="Every hybrid channel failed"):
            await fusion(_Channel("dense", [], fail=True)).retrieve("q")

    async def test_candidate_count_skips_a_dead_channel(self) -> None:
        retriever = fusion(_Channel("bm25", [], fail=True), _Channel("dense", []))
        assert await retriever.candidate_count(None) == 1937

    async def test_the_trace_records_method_constant_and_weights(self) -> None:
        result = await fusion(
            _Channel("dense", [hit("a", 1)]),
            FUSION_RRF_K=10,
            FUSION_WEIGHTS={"dense": 0.5},
        ).retrieve("q")
        trace = result.retrieval_config["fusion"]

        assert (trace["method"], trace["rrf_k"], trace["weights"]) == ("rrf", 10, {"dense": 0.5})
        assert result.retrieval_config["channels"] == ["hybrid:dense"]


class TestSelection:
    def test_hybrid_skips_sparse_while_it_is_disabled(self) -> None:
        retriever = get_retriever(
            AppSettings(APP_ENV="testing", RETRIEVAL_MODE="hybrid", ENABLE_NEURAL_SPARSE=False)
        )

        assert isinstance(retriever, FusionRetriever)
        assert list(retriever.channels) == ["dense", "bm25"]

    def test_hybrid_channels_are_configurable(self) -> None:
        retriever = get_retriever(
            AppSettings(
                APP_ENV="testing",
                RETRIEVAL_MODE="hybrid",
                ENABLE_NEURAL_SPARSE=True,
                HYBRID_CHANNELS=["bm25", "sparse"],
            )
        )
        assert list(retriever.channels) == ["bm25", "sparse"]  # type: ignore[attr-defined]

    def test_no_active_channel_is_a_configuration_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            get_retriever(
                AppSettings(
                    APP_ENV="testing",
                    RETRIEVAL_MODE="hybrid",
                    HYBRID_CHANNELS=["sparse"],
                    ENABLE_NEURAL_SPARSE=False,
                )
            )

    def test_narrowing_wraps_the_fused_retriever(self) -> None:
        """So inferred filters are pushed into every engine at once (Task 6.3)."""
        retriever = get_retriever(
            AppSettings(APP_ENV="testing", RETRIEVAL_MODE="hybrid", ENABLE_METADATA_NARROWING=True)
        )
        assert isinstance(retriever, NarrowingRetriever)
        assert retriever.channel == "hybrid"
