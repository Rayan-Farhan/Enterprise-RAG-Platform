"""Base provider abstractions and resilience helpers."""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.exceptions import ModelProviderException
from app.core.logging import get_logger

T = TypeVar("T")


class BaseProvider:
    """Base class for all model providers with common retry and timeout policies."""

    def __init__(
        self, provider_name: str, timeout_seconds: float = 30.0, max_retries: int = 3
    ) -> None:
        self.provider_name = provider_name
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.logger = get_logger(f"app.models.{provider_name}")

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
                stop=stop_after_attempt(self.max_retries),
                wait=wait_random_exponential(multiplier=1, max=10),
                retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            ):
                with attempt:
                    return await func(*args, **kwargs)
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
