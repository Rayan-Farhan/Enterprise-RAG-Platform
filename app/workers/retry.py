"""Retry policy for job tasks: what to retry, how often, how long to wait (Task 7.5).

A failure is retried only when it is transient: the network, a timeout, a
provider throttling or briefly down, a store that is restarting. Anything else
- a corrupt PDF, a validation mismatch, a missing artifact - fails the same way
every time, and retrying it only delays the dead letter that a person has to
look at. Classification walks the exception's cause chain, because the
services wrap client errors (MinIO's in StorageException, providers' in
ModelProviderException) and the wrapper alone does not say which it was.

`attempt` counts every start of a job, retries and redeliveries alike. A job
whose worker died mid-task is redelivered and counts an attempt, so a message
that kills every worker that takes it - a poison message - runs out of
attempts instead of being redelivered forever.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import opensearchpy
import urllib3.exceptions
from minio.error import S3Error
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import DBAPIError, OperationalError

from app.core.exceptions import ModelProviderException
from app.retrieval.sparse_store import SparseModelNotReadyError

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    httpx.TransportError,
    opensearchpy.ConnectionError,
    ResponseHandlingException,
    urllib3.exceptions.HTTPError,
    OperationalError,
    ModelProviderException,
    SparseModelNotReadyError,
)
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_S3_CODES = frozenset(
    {"SlowDown", "InternalError", "ServiceUnavailable", "RequestTimeout"}
)


def _causes(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient(exc: BaseException) -> bool:
    """True when the same work might succeed if simply tried again later."""
    for cause in _causes(exc):
        if isinstance(cause, _TRANSIENT_TYPES):
            return True
        if isinstance(cause, DBAPIError) and cause.connection_invalidated:
            return True
        if (
            isinstance(cause, opensearchpy.TransportError)
            and cause.status_code in _TRANSIENT_STATUS
        ):
            return True
        if isinstance(cause, UnexpectedResponse) and cause.status_code in _TRANSIENT_STATUS:
            return True
        if isinstance(cause, S3Error) and cause.code in _TRANSIENT_S3_CODES:
            return True
    return False


@dataclass(frozen=True)
class RetryPolicy:
    """How many times a task may start, and how long to wait between tries."""

    max_attempts: int = 4
    base_delay_seconds: float = 5.0
    # Kept well under RabbitMQ's 30-minute consumer timeout: a countdown is
    # held unacknowledged by the worker until it is due.
    max_delay_seconds: float = 300.0

    def delay_for(self, attempt: int, rng: random.Random | None = None) -> float:
        """Exponential backoff with equal jitter, for the retry after ``attempt``.

        Half the delay is fixed and half random: the spread keeps a burst of
        failures (a provider outage hitting every document at once) from coming
        back in lockstep, and the fixed half keeps a retry from firing at once.
        """
        ceiling = min(self.max_delay_seconds, self.base_delay_seconds * 2.0 ** max(0, attempt - 1))
        return ceiling / 2 + (rng or random).uniform(0, ceiling / 2)

    def allows_retry_after(self, attempt: int) -> bool:
        return attempt < self.max_attempts

    def allows_start(self, attempt: int) -> bool:
        """False for a delivery past the limit: run no more of a poison message."""
        return attempt <= self.max_attempts


DEFAULT_POLICY = RetryPolicy()
# Provider calls: throttling is the common failure and it clears in minutes.
INFERENCE_POLICY = RetryPolicy(max_attempts=6, base_delay_seconds=15.0)
# The final steps are cheap and their failures are rarely transient.
CONTROL_POLICY = RetryPolicy(max_attempts=3, base_delay_seconds=2.0, max_delay_seconds=30.0)
