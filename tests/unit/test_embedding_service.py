"""Unit tests for rate-limit-aware batched embedding (Task 3.2)."""

from __future__ import annotations

import pytest

from app.core.config import AppSettings
from app.core.exceptions import ModelProviderException
from app.models.schemas import (
    EmbeddingResult,
    EmbeddingsResponse,
    ModelMetadata,
    TokenCounts,
)
from app.retrieval.embedding import (
    EmbeddingService,
    _is_rate_limit_error,
    _RpmLimiter,
)


class FakeEmbeddingGateway:
    """Records batch sizes and can be told to fail a set number of times."""

    def __init__(
        self,
        dimensions: int = 4,
        fail_times: int = 0,
        error: Exception | None = None,
        shuffle: bool = False,
    ) -> None:
        self.dimensions = dimensions
        self.fail_times = fail_times
        self.error = error or RuntimeError("boom")
        self.shuffle = shuffle
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], model_name: str | None = None) -> EmbeddingsResponse:
        self.calls.append(list(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error

        results = [
            EmbeddingResult(embedding=[float(i)] * self.dimensions, index=i)
            for i in range(len(texts))
        ]
        if self.shuffle:
            results.reverse()

        return EmbeddingsResponse(
            embeddings=results,
            metadata=ModelMetadata(
                provider="fake",
                model_name="fake-embed",
                latency_ms=5.0,
                token_counts=TokenCounts(total_tokens=len(texts) * 3),
            ),
        )

    async def generate(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError

    async def rerank(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError

    async def vision(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError


def make_service(gateway: FakeEmbeddingGateway, **overrides: object) -> EmbeddingService:
    values: dict[str, object] = {
        "APP_ENV": "testing",
        "EMBEDDING_BATCH_SIZE": 3,
        "EMBEDDING_MAX_RPM": 1000,
        "EMBEDDING_MAX_RETRIES": 2,
    }
    values.update(overrides)
    settings = AppSettings(**values)  # type: ignore[arg-type]
    return EmbeddingService(gateway=gateway, settings=settings)  # type: ignore[arg-type]


class TestBatching:
    async def test_empty_input_makes_no_provider_calls(self) -> None:
        gateway = FakeEmbeddingGateway()
        result = await make_service(gateway).embed_texts([])

        assert result.vectors == []
        assert gateway.calls == []

    async def test_texts_are_split_into_configured_batches(self) -> None:
        gateway = FakeEmbeddingGateway()
        result = await make_service(gateway).embed_texts([f"t{i}" for i in range(7)])

        assert [len(call) for call in gateway.calls] == [3, 3, 1]
        assert len(result.vectors) == 7
        assert result.batches == 3

    async def test_vector_order_matches_input_order(self) -> None:
        """A provider returning results out of order must not scramble vectors."""
        gateway = FakeEmbeddingGateway(shuffle=True)
        result = await make_service(gateway).embed_texts(["a", "b", "c"])

        assert [v[0] for v in result.vectors] == [0.0, 1.0, 2.0]

    async def test_metadata_is_aggregated_across_batches(self) -> None:
        gateway = FakeEmbeddingGateway()
        result = await make_service(gateway).embed_texts([f"t{i}" for i in range(6)])

        assert result.provider == "fake"
        assert result.model_name == "fake-embed"
        assert result.dimensions == 4
        assert result.total_latency_ms == pytest.approx(10.0)
        assert result.tokens_used == 18

    async def test_embed_query_returns_a_single_vector(self) -> None:
        gateway = FakeEmbeddingGateway()
        vector = await make_service(gateway).embed_query("how many leave days?")

        assert len(vector) == 4
        assert gateway.calls == [["how many leave days?"]]

    async def test_count_mismatch_is_a_hard_failure(self) -> None:
        class ShortGateway(FakeEmbeddingGateway):
            async def embed(
                self, texts: list[str], model_name: str | None = None
            ) -> EmbeddingsResponse:
                return EmbeddingsResponse(
                    embeddings=[EmbeddingResult(embedding=[1.0], index=0)],
                    metadata=ModelMetadata(provider="fake", model_name="m", latency_ms=1.0),
                )

        with pytest.raises(ModelProviderException, match="returned 1 vectors for 2 inputs"):
            await make_service(ShortGateway()).embed_texts(["a", "b"])


class TestRateLimitHandling:
    @pytest.mark.parametrize(
        "message",
        [
            "429 Too Many Requests",
            "Rate limit exceeded for this key",
            "rate_limit_error",
            "monthly quota exhausted",
            "TOO MANY REQUESTS",
        ],
    )
    def test_rate_limit_errors_are_recognised(self, message: str) -> None:
        assert _is_rate_limit_error(RuntimeError(message))

    def test_unrelated_errors_are_not_rate_limits(self) -> None:
        assert not _is_rate_limit_error(RuntimeError("connection reset by peer"))

    async def test_transient_failure_is_retried_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("app.retrieval.embedding.asyncio.sleep", fake_sleep)

        gateway = FakeEmbeddingGateway(fail_times=2, error=RuntimeError("connection reset"))
        result = await make_service(gateway).embed_texts(["a"])

        assert len(result.vectors) == 1
        assert len(slept) == 2
        # Backoff must grow, not stay flat.
        assert slept[1] > slept[0]

    async def test_rate_limit_backoff_is_longer_than_generic_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays: dict[str, float] = {}

        async def record(label: str, seconds: float) -> None:
            delays.setdefault(label, seconds)

        for label, error in (
            ("generic", RuntimeError("connection reset")),
            ("rate_limit", RuntimeError("429 rate limit exceeded")),
        ):
            captured: list[float] = []

            async def fake_sleep(seconds: float, sink: list[float] = captured) -> None:
                sink.append(seconds)

            monkeypatch.setattr("app.retrieval.embedding.asyncio.sleep", fake_sleep)
            await make_service(FakeEmbeddingGateway(fail_times=1, error=error)).embed_texts(["a"])
            await record(label, captured[0])

        assert delays["rate_limit"] > delays["generic"]

    async def test_exhausted_retries_raise_provider_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr("app.retrieval.embedding.asyncio.sleep", fake_sleep)

        gateway = FakeEmbeddingGateway(fail_times=99, error=RuntimeError("429 rate limit"))
        with pytest.raises(ModelProviderException, match="Embedding failed after 3 attempts"):
            await make_service(gateway).embed_texts(["a"])

    async def test_rpm_limiter_records_waits_instead_of_failing(self) -> None:
        """Hitting the RPM ceiling must slow the job, never fail it."""
        gateway = FakeEmbeddingGateway()
        service = make_service(gateway, EMBEDDING_BATCH_SIZE=1, EMBEDDING_MAX_RPM=2)
        # Shrink the window so the wait path runs in milliseconds, not a minute.
        service._limiter = _RpmLimiter(max_rpm=2, window_seconds=0.15)

        result = await service.embed_texts(["a", "b", "c", "d"])

        assert len(result.vectors) == 4
        assert result.rate_limit_waits > 0
        assert len(gateway.calls) == 4


class TestRpmLimiter:
    async def test_allows_up_to_max_without_waiting(self) -> None:
        limiter = _RpmLimiter(max_rpm=3, window_seconds=5.0)
        assert [await limiter.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]

    async def test_waits_once_the_window_is_full(self) -> None:
        limiter = _RpmLimiter(max_rpm=1, window_seconds=0.1)
        assert await limiter.acquire() == 0.0
        assert await limiter.acquire() > 0.0
