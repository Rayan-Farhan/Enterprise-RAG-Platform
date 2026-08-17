"""Provider rate-limit handling (Task 0.6, exercised hard by Stage 4).

A 429 is the one provider error that is expected on the free tiers and that
resolves on its own. Before these tests it was raised as a generic
``ModelProviderException``, which the retry predicate did not cover — so a bulk
job reported the rate limiter's behaviour as the system's quality.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.exceptions import ModelProviderException, ProviderRateLimitException
from app.models.providers.base import (
    MAX_RETRY_AFTER_SECONDS,
    RATE_LIMIT_MAX_ATTEMPTS,
    BaseProvider,
    parse_retry_after,
)


def _response(status_code: int, text: str = "", headers: dict[str, str] | None = None):
    return httpx.Response(
        status_code=status_code,
        text=text,
        headers=headers or {},
        request=httpx.Request("POST", "https://example.invalid/v1/chat"),
    )


class TestParseRetryAfter:
    def test_reads_the_retry_after_header(self) -> None:
        assert parse_retry_after(_response(429, headers={"retry-after": "12"})) == 12.0

    def test_falls_back_to_the_seconds_named_in_the_body(self) -> None:
        # Groq states the wait in the body rather than the header.
        body = '{"error":{"message":"Please try again in 19.5825s.","code":"rate_limit_exceeded"}}'
        assert parse_retry_after(_response(429, text=body)) == pytest.approx(19.5825)

    def test_returns_none_when_nothing_is_stated(self) -> None:
        assert parse_retry_after(_response(429, text="slow down")) is None

    def test_ignores_an_unparseable_header_and_uses_the_body(self) -> None:
        body = "try again in 5s"
        response = _response(429, text=body, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        assert parse_retry_after(response) == 5.0


class TestRaiseForResponse:
    def test_200_raises_nothing(self) -> None:
        BaseProvider("groq").raise_for_response(_response(200, text="{}"))

    def test_429_raises_the_rate_limit_type_with_the_stated_wait(self) -> None:
        provider = BaseProvider("groq")
        with pytest.raises(ProviderRateLimitException) as exc_info:
            provider.raise_for_response(_response(429, text="try again in 7.5s"))

        assert exc_info.value.retry_after_seconds == pytest.approx(7.5)
        assert exc_info.value.details["provider"] == "groq"

    def test_other_errors_stay_generic_so_they_are_not_waited_out(self) -> None:
        provider = BaseProvider("gemini")
        with pytest.raises(ModelProviderException) as exc_info:
            provider.raise_for_response(_response(500, text="boom"))

        assert not isinstance(exc_info.value, ProviderRateLimitException)

    def test_long_response_bodies_are_truncated(self) -> None:
        provider = BaseProvider("jina")
        with pytest.raises(ModelProviderException) as exc_info:
            provider.raise_for_response(_response(400, text="x" * 5000))

        assert len(str(exc_info.value)) < 1000


class TestRetryBehaviour:
    @pytest.mark.asyncio
    async def test_a_rate_limited_call_is_retried_and_then_succeeds(self, monkeypatch) -> None:
        provider = BaseProvider("groq", max_retries=3)
        # Waiting for real would make the suite take as long as a free-tier window.
        monkeypatch.setattr(provider, "_wait", lambda retry_state: 0.0)
        attempts = 0

        async def call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                provider.raise_for_response(_response(429, text="try again in 20s"))
            return "ok"

        assert await provider.execute_with_retry(call) == "ok"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_rate_limits_get_a_larger_attempt_budget_than_transport_faults(
        self, monkeypatch
    ) -> None:
        provider = BaseProvider("groq", max_retries=2)
        monkeypatch.setattr(provider, "_wait", lambda retry_state: 0.0)

        rate_limited = 0

        async def always_rate_limited() -> str:
            nonlocal rate_limited
            rate_limited += 1
            provider.raise_for_response(_response(429, text="nope"))
            raise AssertionError("unreachable")

        with pytest.raises(ModelProviderException):
            await provider.execute_with_retry(always_rate_limited)
        assert rate_limited == RATE_LIMIT_MAX_ATTEMPTS

        transport_faults = 0

        async def always_timing_out() -> str:
            nonlocal transport_faults
            transport_faults += 1
            raise httpx.ConnectTimeout("timed out")

        with pytest.raises(ModelProviderException):
            await provider.execute_with_retry(always_timing_out)
        assert transport_faults == 2

    @pytest.mark.asyncio
    async def test_a_daily_quota_refusal_is_not_retried_at_all(self, monkeypatch) -> None:
        # A wait longer than the cap is a per-day quota, not a per-minute window:
        # it will not clear inside this process, so retrying spends six capped
        # sleeps to arrive at the same refusal.
        provider = BaseProvider("groq", max_retries=3)
        monkeypatch.setattr(provider, "_wait", lambda retry_state: 0.0)
        attempts = 0

        async def call() -> str:
            nonlocal attempts
            attempts += 1
            provider.raise_for_response(
                _response(429, text="tokens per day (TPD) exceeded. try again in 578.4s")
            )
            raise AssertionError("unreachable")

        with pytest.raises(ProviderRateLimitException):
            await provider.execute_with_retry(call)
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_a_per_minute_window_is_still_retried(self, monkeypatch) -> None:
        provider = BaseProvider("groq", max_retries=3)
        monkeypatch.setattr(provider, "_wait", lambda retry_state: 0.0)
        attempts = 0

        async def call() -> str:
            nonlocal attempts
            attempts += 1
            provider.raise_for_response(_response(429, text="try again in 19.5s"))
            raise AssertionError("unreachable")

        with pytest.raises(ProviderRateLimitException):
            await provider.execute_with_retry(call)
        assert attempts == RATE_LIMIT_MAX_ATTEMPTS

    def test_the_wait_honours_the_provider_stated_window(self) -> None:
        provider = BaseProvider("groq")
        state = _retry_state(ProviderRateLimitException("limited", "groq", retry_after_seconds=19.5))

        assert provider._wait(state) == pytest.approx(20.0)

    def test_the_wait_is_capped_so_a_worker_cannot_stall_indefinitely(self) -> None:
        provider = BaseProvider("groq")
        state = _retry_state(ProviderRateLimitException("limited", "groq", retry_after_seconds=1e6))

        assert provider._wait(state) == pytest.approx(MAX_RETRY_AFTER_SECONDS + 0.5)

    def test_non_rate_limit_failures_use_the_exponential_curve(self) -> None:
        provider = BaseProvider("groq")
        state = _retry_state(httpx.ConnectTimeout("timed out"))

        assert 0.0 <= provider._wait(state) <= 10.0


def _retry_state(exception: BaseException):
    """Build the minimal tenacity state the wait/stop callables read."""
    from tenacity import Future, RetryCallState

    state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})
    state.attempt_number = 1
    future: Future = Future(attempt_number=1)
    future.set_exception(exception)
    state.outcome = future
    return state
