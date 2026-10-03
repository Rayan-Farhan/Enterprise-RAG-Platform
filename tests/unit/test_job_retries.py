"""Retry classification, backoff, and the retry / dead-letter paths of a job (Task 7.5)."""

from __future__ import annotations

import random
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import opensearchpy
import pytest
import urllib3.exceptions
from minio.error import S3Error
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.exceptions import ModelProviderException, StorageException
from app.db.models.base import Base
from app.db.models.job import FailureKind, JobStatus, JobType
from app.ingestion.pipeline import IndexValidationError
from app.jobs.service import JobService
from app.workers import job_task as job_task_module
from app.workers.job_task import JobContext, RetryLater, execute_job
from app.workers.retry import RetryPolicy, is_transient

POLICY = RetryPolicy(max_attempts=3, base_delay_seconds=4.0, max_delay_seconds=10.0)


def _wrapped(outer: Exception, inner: BaseException) -> Exception:
    try:
        raise outer from inner
    except Exception as exc:
        return exc


def _s3(code: str) -> S3Error:
    return S3Error(None, code, "m", "r", "h", "i")  # type: ignore[arg-type]


class TestClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("refused"),
            TimeoutError("slow"),
            httpx.ConnectTimeout("connect"),
            ModelProviderException("jina down", provider="jina"),
            opensearchpy.ConnectionTimeout("TIMEOUT", "read", Exception()),
            opensearchpy.TransportError(503, "unavailable"),
            _wrapped(
                StorageException("download failed"), urllib3.exceptions.ProtocolError("reset")
            ),
            _wrapped(StorageException("throttled"), _s3("SlowDown")),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_transient_failures_are_retried(self, exc: Exception) -> None:
        assert is_transient(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("corrupt PDF"),
            IndexValidationError("expected 8 chunks everywhere; found qdrant points 7"),
            opensearchpy.TransportError(400, "mapper_parsing_exception"),
            _wrapped(StorageException("Object not found: derived/x"), _s3("NoSuchKey")),
            KeyError("parser"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_permanent_failures_are_not(self, exc: Exception) -> None:
        assert not is_transient(exc)


class TestBackoff:
    def test_delay_doubles_and_stays_within_equal_jitter_bounds(self) -> None:
        rng = random.Random(7)
        for attempt, ceiling in ((1, 4.0), (2, 8.0), (3, 10.0), (9, 10.0)):
            for _ in range(50):
                delay = POLICY.delay_for(attempt, rng)
                assert ceiling / 2 <= delay <= ceiling

    def test_delays_spread_rather_than_repeat(self) -> None:
        rng = random.Random(1)
        assert len({round(POLICY.delay_for(2, rng), 6) for _ in range(20)}) > 15

    def test_attempt_limits(self) -> None:
        assert POLICY.allows_retry_after(2) and not POLICY.allows_retry_after(3)
        assert POLICY.allows_start(3) and not POLICY.allows_start(4)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retries.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def jobs(engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> JobService:
    url = f"sqlite+aiosqlite:///{tmp_path / 'retries.db'}"
    monkeypatch.setattr(job_task_module, "worker_engine_factory", lambda: create_async_engine(url))
    return JobService(async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False))


async def queued(jobs: JobService) -> uuid.UUID:
    return (await jobs.create(task_type=JobType.DIAGNOSTIC, queue="ingestion")).id


def failing(exc: Exception):  # type: ignore[no-untyped-def]
    async def body(ctx: JobContext) -> None:
        raise exc

    return body


class TestRetryPath:
    async def test_a_transient_failure_requeues_the_job_with_its_error(
        self, jobs: JobService
    ) -> None:
        job_id = await queued(jobs)

        with pytest.raises(RetryLater) as later:
            await execute_job(
                failing(ConnectionError("qdrant refused")), job_id, "w", {}, policy=POLICY
            )

        job = await jobs.get(job_id)
        assert job.status == JobStatus.QUEUED
        assert job.attempt == 1
        assert job.error == "ConnectionError: qdrant refused"
        assert "retrying in" in (job.progress_message or "")
        assert 2.0 <= later.value.delay <= 4.0

    async def test_the_retry_runs_as_the_same_job_and_can_succeed(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        with pytest.raises(RetryLater):
            await execute_job(failing(TimeoutError("slow")), job_id, "w", {}, policy=POLICY)

        async def ok(ctx: JobContext) -> None:
            return None

        assert await execute_job(ok, job_id, "w", {}, policy=POLICY) == "succeeded"
        job = await jobs.get(job_id)
        assert (job.status, job.attempt) == (JobStatus.SUCCEEDED, 2)

    async def test_a_permanent_failure_is_dead_lettered_at_once(self, jobs: JobService) -> None:
        job_id = await queued(jobs)

        with pytest.raises(ValueError):
            await execute_job(failing(ValueError("corrupt PDF")), job_id, "w", {}, policy=POLICY)

        job = await jobs.get(job_id)
        assert (job.status, job.failure_kind, job.attempt) == (
            JobStatus.FAILED,
            FailureKind.PERMANENT,
            1,
        )
        assert [j.id for j in await jobs.dead_letters()] == [job_id]

    async def test_transient_failures_past_the_limit_exhaust_the_retries(
        self, jobs: JobService
    ) -> None:
        job_id = await queued(jobs)
        body = failing(ConnectionError("down"))
        for _ in range(POLICY.max_attempts - 1):
            with pytest.raises(RetryLater):
                await execute_job(body, job_id, "w", {}, policy=POLICY)

        with pytest.raises(ConnectionError):
            await execute_job(body, job_id, "w", {}, policy=POLICY)

        job = await jobs.get(job_id)
        assert (job.status, job.failure_kind, job.attempt) == (
            JobStatus.FAILED,
            FailureKind.RETRIES_EXHAUSTED,
            POLICY.max_attempts,
        )

    async def test_a_poison_message_is_dead_lettered_without_running(
        self, jobs: JobService
    ) -> None:
        # Every delivery so far died with its worker: the row is still running
        # and each redelivery's start counted an attempt.
        job_id = await queued(jobs)
        for _ in range(POLICY.max_attempts):
            await jobs.start(job_id, worker="lost")
        ran = False

        async def body(ctx: JobContext) -> None:
            nonlocal ran
            ran = True

        assert await execute_job(body, job_id, "w", {}, policy=POLICY) == "dead_lettered"
        job = await jobs.get(job_id)
        assert not ran
        assert (job.status, job.failure_kind) == (JobStatus.FAILED, FailureKind.DELIVERY_LIMIT)

    async def test_a_cancel_during_a_failing_attempt_wins_over_the_retry(
        self, jobs: JobService
    ) -> None:
        job_id = await queued(jobs)

        async def body(ctx: JobContext) -> None:
            await jobs.request_cancel(ctx.job_id)
            raise ConnectionError("down")

        assert await execute_job(body, job_id, "w", {}, policy=POLICY) == "cancelled"
        assert (await jobs.get(job_id)).status == JobStatus.CANCELLED
