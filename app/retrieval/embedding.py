"""Rate-limit-aware batched embedding over the model gateway (Task 3.2, ADR-046).

Free provider tiers have low RPM ceilings and bulk indexing will hit them. The
contract here is that hitting a rate limit slows the job down; it never fails it.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from app.core.config import AppSettings, get_settings
from app.core.exceptions import ModelProviderException
from app.core.logging import get_logger
from app.models.gateway import ModelGateway, get_model_gateway
from app.models.schemas import EmbeddingsResponse

logger = get_logger("app.retrieval.embedding")

_RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "quota", "too many requests")


@dataclass
class EmbeddingBatchResult:
    """Vectors for a set of texts plus the provider metadata behind them."""

    vectors: list[list[float]]
    provider: str
    model_name: str
    embedding_version: str
    total_latency_ms: float = 0.0
    batches: int = 0
    rate_limit_waits: int = 0
    tokens_used: int = 0
    dimensions: int = field(default=0)


class _RpmLimiter:
    """Sliding-window request-per-minute limiter.

    ``window_seconds`` is injectable so tests can exercise the waiting path
    without a minute of wall clock; production always uses the 60s default.
    """

    def __init__(self, max_rpm: int, window_seconds: float = 60.0) -> None:
        self.max_rpm = max(1, max_rpm)
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    async def acquire(self) -> float:
        """Block until a request slot is free; returns seconds waited."""
        waited = 0.0
        while True:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self.window_seconds]
            if len(self._timestamps) < self.max_rpm:
                self._timestamps.append(now)
                return waited

            sleep_for = self.window_seconds - (now - self._timestamps[0]) + 0.05
            waited += sleep_for
            await asyncio.sleep(sleep_for)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Heuristically classify a provider error as a rate limit.

    Providers surface 429s inconsistently across SDKs, so this matches on the
    message. Misclassifying a non-rate-limit error only costs extra backoff.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


class EmbeddingService:
    """Embeds text in bounded batches, degrading to slower batching under limits."""

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gateway = gateway or get_model_gateway()
        self.batch_size = max(1, self.settings.EMBEDDING_BATCH_SIZE)
        self.max_retries = max(0, self.settings.EMBEDDING_MAX_RETRIES)
        self._limiter = _RpmLimiter(self.settings.EMBEDDING_MAX_RPM)

    async def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Embed many texts, preserving input order in the returned vectors."""
        result = EmbeddingBatchResult(
            vectors=[],
            provider="unknown",
            model_name="unknown",
            embedding_version=self.settings.EMBEDDING_VERSION,
        )
        if not texts:
            return result

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            waited = await self._limiter.acquire()
            if waited > 0:
                result.rate_limit_waits += 1

            response = await self._embed_with_retry(batch)

            # Provider responses are not guaranteed ordered; sort by index.
            ordered = sorted(response.embeddings, key=lambda e: e.index)
            if len(ordered) != len(batch):
                raise ModelProviderException(
                    f"Embedding provider returned {len(ordered)} vectors for {len(batch)} inputs",
                    provider=response.metadata.provider,
                )
            result.vectors.extend(item.embedding for item in ordered)

            result.provider = response.metadata.provider
            result.model_name = response.metadata.model_name
            result.total_latency_ms += response.metadata.latency_ms
            result.tokens_used += response.metadata.token_counts.total_tokens
            result.batches += 1

        result.dimensions = len(result.vectors[0]) if result.vectors else 0
        logger.info(
            "embedding_batch_complete",
            texts=len(texts),
            batches=result.batches,
            dimensions=result.dimensions,
            rate_limit_waits=result.rate_limit_waits,
            provider=result.provider,
        )
        return result

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        if not result.vectors:
            raise ModelProviderException(
                "Embedding provider returned no vector for the query",
                provider=result.provider,
            )
        return result.vectors[0]

    async def _embed_with_retry(self, batch: list[str]) -> EmbeddingsResponse:
        """Call the gateway with bounded, jittered retry on transient failures."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.gateway.embed(texts=batch)
            except Exception as exc:  # noqa: BLE001 - provider errors are opaque
                last_exc = exc
                if attempt >= self.max_retries:
                    break

                rate_limited = _is_rate_limit_error(exc)
                # Rate limits get a much longer floor than generic errors: the
                # window has to actually roll over before a retry can succeed.
                base = 20.0 if rate_limited else 1.0
                delay = min(base * (2**attempt), 120.0)
                delay += random.uniform(0, delay * 0.25)  # noqa: S311 - jitter, not crypto

                logger.warning(
                    "embedding_batch_retry",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    rate_limited=rate_limited,
                    delay_seconds=round(delay, 2),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

        raise ModelProviderException(
            f"Embedding failed after {self.max_retries + 1} attempts: {last_exc}",
            provider="embedding",
        )


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the singleton EmbeddingService."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
