"""Base provider abstractions and resilience helpers."""

import asyncio
import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.exceptions import ModelProviderException, ProviderRateLimitException
from app.core.logging import get_logger

T = TypeVar("T")

#: Rate limits get their own, larger attempt budget. A 429 is a wait, not a
#: fault, and the free tiers' per-minute windows routinely outlast the three
#: attempts that are right for a flaky socket.
RATE_LIMIT_MAX_ATTEMPTS = 6

#: Cap on how long a single provider-requested wait is honoured. Without it a
#: misconfigured or hostile Retry-After could stall a worker indefinitely.
MAX_RETRY_AFTER_SECONDS = 90.0


def parse_retry_after(response: httpx.Response) -> float | None:
    """Extract the wait a 429 response asks for.

    Two sources, in order of reliability: the standard ``Retry-After`` header,
    and the seconds embedded in the body text that Groq and several other
    OpenAI-compatible tiers return instead of (or alongside) the header.
    """
    header = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset")
    if header:
        try:
            return float(header)
        except ValueError:
            pass  # An HTTP-date form is possible; fall through to the body.

    match = re.search(r"try again in ([0-9.]+)\s*s", response.text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


class BaseProvider:
    """Base class for all model providers with common retry and timeout policies."""

    def __init__(
        self, provider_name: str, timeout_seconds: float = 30.0, max_retries: int = 3
    ) -> None:
        self.provider_name = provider_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.logger = get_logger(f"app.models.{provider_name}")

    def require_credentials(self, api_key: str | None, setting_name: str) -> None:
        """Fail loudly when a provider has no credentials.

        Providers must never substitute placeholder output for a real call. A
        fabricated embedding is indistinguishable from a real one downstream: it
        indexes cleanly, retrieves cleanly, and silently destroys answer quality
        while every health signal stays green. Use `INFERENCE_PROFILE=stub` when a
        keyless run is genuinely wanted — that path is explicit and self-labelling.
        """
        if not api_key:
            raise ModelProviderException(
                message=(
                    f"Provider '{self.provider_name}' has no credentials: {setting_name} is "
                    f"not set. Set it, or run with INFERENCE_PROFILE=stub for an explicitly "
                    f"fake gateway."
                ),
                provider=self.provider_name,
                details={"missing_setting": setting_name},
            )

    def raise_for_response(self, response: httpx.Response) -> None:
        """Turn a non-200 provider response into the right exception type.

        Every provider funnels through here so that 429 handling is uniform:
        a rate limit raised as a generic provider error is not retried, and the
        caller sees a failure that never needed to happen.
        """
        if response.status_code == 200:
            return

        if response.status_code == 429:
            raise ProviderRateLimitException(
                message=(
                    f"{self.provider_name} rate limit reached "
                    f"(HTTP 429): {response.text[:500]}"
                ),
                provider=self.provider_name,
                retry_after_seconds=parse_retry_after(response),
            )

        raise ModelProviderException(
            message=(
                f"{self.provider_name} API returned error "
                f"{response.status_code}: {response.text[:500]}"
            ),
            provider=self.provider_name,
        )

    def _stop(self, retry_state: RetryCallState) -> bool:
        """Give rate limits a larger attempt budget than transport faults.

        With one exception: when the provider asks for longer than
        ``MAX_RETRY_AFTER_SECONDS``, stop immediately. A wait that long is a
        *daily* quota, not a per-minute window, and it will not clear inside this
        process. Retrying anyway spends six capped sleeps — about nine minutes —
        per question to arrive at the same refusal, which on a 100-question run
        is hours of sleeping to learn something the first response already said.
        """
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, ProviderRateLimitException):
            requested = exception.retry_after_seconds
            if requested is not None and requested > MAX_RETRY_AFTER_SECONDS:
                self.logger.warning(
                    "provider_quota_exhausted",
                    provider=self.provider_name,
                    retry_after_seconds=round(requested, 1),
                )
                return True
            return bool(stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS)(retry_state))

        return bool(stop_after_attempt(self.max_retries)(retry_state))

    def _wait(self, retry_state: RetryCallState) -> float:
        """Back off for the window the provider asked for, else exponentially.

        Free tiers state a precise recovery time in the 429 body or in
        ``Retry-After``. Ignoring it and backing off on a generic curve either
        retries too early — burning an attempt and the quota with it — or waits
        far longer than needed across hundreds of calls.
        """
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exception, ProviderRateLimitException) and exception.retry_after_seconds:
            wait_seconds = min(exception.retry_after_seconds, MAX_RETRY_AFTER_SECONDS)
            self.logger.warning(
                "provider_rate_limited",
                provider=self.provider_name,
                attempt=retry_state.attempt_number,
                sleeping_seconds=round(wait_seconds, 1),
            )
            return wait_seconds + 0.5  # a small margin so the window has definitely closed
        return float(wait_random_exponential(multiplier=1, max=10)(retry_state))

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an asynchronous provider call with exponential backoff and jitter."""
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=self._stop,
                wait=self._wait,
                retry=retry_if_exception_type(
                    (httpx.HTTPError, httpx.TimeoutException, ProviderRateLimitException)
                ),
            ):
                with attempt:
                    return await func(*args, **kwargs)
        except ProviderRateLimitException:
            # Re-raised unchanged rather than wrapped: callers distinguish "the
            # provider refused on quota" from "the provider is broken", and the
            # evaluation runner uses that distinction to decide whether a
            # question was measured or merely declined.
            self.logger.warning("provider_rate_limit_exhausted", provider=self.provider_name)
            raise
        except Exception as exc:
            self.logger.error("Provider execution failed after retries", error=str(exc))
            raise ModelProviderException(
                message=f"Provider '{self.provider_name}' failed: {exc}",
                provider=self.provider_name,
                details={"original_error": str(exc)},
            ) from exc

    @staticmethod
    async def batch_process(
        items: Sequence[T],
        batch_size: int,
        process_func: Callable[[Sequence[T]], Any],
        rate_limit_delay_seconds: float = 0.0,
    ) -> list[Any]:
        """Process a sequence in rate-limit-aware batches."""
        results: list[Any] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            batch_result = await process_func(batch)
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
            if rate_limit_delay_seconds > 0 and i + batch_size < len(items):
                await asyncio.sleep(rate_limit_delay_seconds)
        return results
