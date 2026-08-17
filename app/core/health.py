"""Dependency health probes (Task 3.7, master §44).

Required vs. optional is the load-bearing distinction: a required dependency going
down must remove the instance from the load balancer, while an optional one must
not. Stage 11 extends this with circuit-breaker state and per-dependency detail.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy import text

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger

logger = get_logger("app.core.health")

# A probe must never hang readiness. Kubernetes probe timeouts are short, so a
# slow dependency is reported unreachable rather than blocking the response.
#
# 5s, not 3s: measured round-trips against local Docker on Windows reach ~2.8s for
# OpenSearch because the Redis and OpenSearch clients are constructed per probe. A
# 3s budget reported healthy dependencies as unreachable. Stage 11 should cache
# those clients and can then tighten this back down.
PROBE_TIMEOUT_SECONDS = 5.0


class DependencyStatus(StrEnum):
    """Outcome of a single dependency probe."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"


@dataclass
class DependencyReport:
    """The result of probing one dependency."""

    name: str
    status: DependencyStatus
    required: bool
    latency_ms: float | None = None
    error: str | None = None

    @property
    def blocks_readiness(self) -> bool:
        return self.required and self.status is not DependencyStatus.HEALTHY


async def _probe(
    name: str,
    required: bool,
    check: Callable[[], Awaitable[None]],
) -> DependencyReport:
    """Run one probe with a timeout, converting any failure into a report."""
    started = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout=PROBE_TIMEOUT_SECONDS)
    except TimeoutError:
        return DependencyReport(
            name=name,
            status=DependencyStatus.UNREACHABLE,
            required=required,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=f"probe exceeded {PROBE_TIMEOUT_SECONDS}s timeout",
        )
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        return DependencyReport(
            name=name,
            status=DependencyStatus.UNREACHABLE,
            required=required,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=str(exc)[:300],
        )

    return DependencyReport(
        name=name,
        status=DependencyStatus.HEALTHY,
        required=required,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


async def _check_postgres() -> None:
    from app.db.session import get_engine

    engine = get_engine()
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


def _qdrant_probe() -> None:
    from app.retrieval.vector_store import get_vector_store

    if not get_vector_store().health_check():
        raise RuntimeError("Qdrant did not respond to get_collections()")


async def _check_qdrant() -> None:
    # Client *construction* is also blocking (it performs a version-compatibility
    # request), so the whole probe runs in a thread. Constructing it on the event
    # loop stalls every other probe and makes them all time out together.
    await asyncio.to_thread(_qdrant_probe)


def _minio_probe() -> None:
    from app.storage.minio_service import get_storage_service

    settings = get_settings()
    if not get_storage_service().client.bucket_exists(settings.MINIO_BUCKET_NAME):
        raise RuntimeError(f"Bucket '{settings.MINIO_BUCKET_NAME}' does not exist")


async def _check_minio() -> None:
    await asyncio.to_thread(_minio_probe)


async def _check_redis() -> None:
    import redis.asyncio as aioredis

    settings = get_settings()
    client = aioredis.from_url(settings.redis_url)
    try:
        await client.ping()
    finally:
        await client.aclose()


def _opensearch_probe() -> None:
    # The synchronous client is used deliberately: `AsyncOpenSearch` requires the
    # optional `opensearch-py[async]` extra (aiohttp), and OpenSearch is not on the
    # request path until Stage 6. Running the sync client in a thread avoids taking
    # that dependency for a health probe.
    from opensearchpy import OpenSearch

    settings = get_settings()
    client = OpenSearch(
        hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
        use_ssl=settings.OPENSEARCH_USE_SSL,
        verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        ssl_show_warn=False,
    )
    try:
        if not client.ping():
            raise RuntimeError("OpenSearch ping returned false")
    finally:
        client.close()


async def _check_opensearch() -> None:
    await asyncio.to_thread(_opensearch_probe)


async def _check_rabbitmq() -> None:
    settings = get_settings()
    _, writer = await asyncio.open_connection(settings.RABBITMQ_HOST, settings.RABBITMQ_PORT)
    writer.close()
    await writer.wait_closed()


# Required set reflects what the Stage 3 request path genuinely cannot serve
# without: PostgreSQL (source of truth), Qdrant (the only retrieval channel), and
# MinIO (source documents). Redis, RabbitMQ, and OpenSearch are not yet on the
# request path — they become required in Stages 11, 7, and 6 respectively.
_PROBES: tuple[tuple[str, bool, Callable[[], Awaitable[None]]], ...] = (
    ("postgres", True, _check_postgres),
    ("qdrant", True, _check_qdrant),
    ("minio", True, _check_minio),
    ("redis", False, _check_redis),
    ("rabbitmq", False, _check_rabbitmq),
    ("opensearch", False, _check_opensearch),
)


async def check_dependencies(settings: AppSettings | None = None) -> list[DependencyReport]:
    """Probe every dependency concurrently and return their reports."""
    _ = settings or get_settings()
    reports = await asyncio.gather(
        *(_probe(name, required, check) for name, required, check in _PROBES)
    )

    unhealthy = [r.name for r in reports if r.status is not DependencyStatus.HEALTHY]
    if unhealthy:
        logger.warning("dependency_probe_failures", dependencies=unhealthy)

    return list(reports)


def overall_status(reports: list[DependencyReport]) -> str:
    """Reduce reports to an overall status string.

    ``degraded`` is deliberately distinct from ``unhealthy``: an optional
    dependency being down is worth reporting but must not fail readiness.
    """
    if any(r.blocks_readiness for r in reports):
        return "unhealthy"
    if any(r.status is not DependencyStatus.HEALTHY for r in reports):
        return "degraded"
    return "healthy"
