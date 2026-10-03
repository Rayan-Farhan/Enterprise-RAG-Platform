"""Distributed locks and where they are applied (Task 7.6).

The lock logic runs against the in-memory store from tests/conftest.py; the
Lua scripts themselves are exercised against a real Redis in
tests/integration/test_locks_live.py.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import redis.exceptions
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core import locks
from app.core.locks import DistributedLock, LockLost, LockNotAcquired, distributed_lock
from app.db.models.base import Base
from app.db.models.document import Document
from app.db.models.job import JobStatus, JobType
from app.db.models.version import DocumentVersion, VersionStatus
from app.ingestion import pipeline
from app.jobs.service import JobService
from app.workers import job_task as job_task_module
from app.workers.job_task import JobContext, RetryLater, execute_job, single_instance
from app.workers.retry import is_transient
from tests.conftest import InMemoryLockRedis


class TestLockPrimitive:
    async def test_only_one_holder_at_a_time(self, lock_store: InMemoryLockRedis) -> None:
        first = DistributedLock(lock_store.client(), "x", ttl=5)
        second = DistributedLock(lock_store.client(), "x", ttl=5)

        assert await first.acquire()
        assert not await second.acquire()
        assert await first.release()
        assert await second.acquire()

    async def test_a_non_owner_cannot_release_or_extend(
        self, lock_store: InMemoryLockRedis
    ) -> None:
        owner = DistributedLock(lock_store.client(), "x", ttl=5)
        other = DistributedLock(lock_store.client(), "x", ttl=5)
        await owner.acquire()

        assert not await other.release()
        assert not await other.extend()
        assert other.lost
        assert await owner.held()

    async def test_an_expired_holder_cannot_free_the_next_holders_lock(
        self, lock_store: InMemoryLockRedis
    ) -> None:
        stale = DistributedLock(lock_store.client(), "x", ttl=0.05)
        await stale.acquire()
        await asyncio.sleep(0.1)
        fresh = DistributedLock(lock_store.client(), "x", ttl=5)

        assert await fresh.acquire()
        assert not await stale.release()
        assert await fresh.held()

    async def test_acquire_waits_for_the_holder_to_let_go(
        self, lock_store: InMemoryLockRedis
    ) -> None:
        holder = DistributedLock(lock_store.client(), "x", ttl=5)
        waiter = DistributedLock(lock_store.client(), "x", ttl=5)
        await holder.acquire()

        async def let_go() -> None:
            await asyncio.sleep(0.15)
            await holder.release()

        releaser = asyncio.create_task(let_go())
        assert await waiter.acquire(wait=2.0, poll=0.02)
        await releaser

    async def test_a_held_context_renews_past_its_ttl(self, lock_store: InMemoryLockRedis) -> None:
        async with distributed_lock("x", ttl=0.15) as lock:
            await asyncio.sleep(0.5)
            assert await lock.held()
            lock.ensure_held()
        assert await lock_store.client().get("lock:x") is None

    async def test_renewal_survives_a_holder_that_blocks_its_event_loop(
        self, lock_store: InMemoryLockRedis
    ) -> None:
        # Pipeline steps call synchronous clients (OpenSearch bulk, the parser)
        # from async code, freezing their loop for minutes. Renewal must not
        # depend on that loop, or the lock expires under a live holder.
        async with distributed_lock("x", ttl=0.15) as lock:
            time.sleep(0.6)  # noqa: ASYNC251 - blocking on purpose
            assert await lock.held()
            lock.ensure_held()

    async def test_a_lock_lost_while_held_is_detected(self, lock_store: InMemoryLockRedis) -> None:
        async with distributed_lock("x", ttl=0.15) as lock:
            lock_store._values.clear()  # expired, or Redis restarted
            await asyncio.sleep(0.15)
            with pytest.raises(LockLost):
                lock.ensure_held()

    async def test_a_taken_lock_raises_and_is_transient(
        self, lock_store: InMemoryLockRedis
    ) -> None:
        await DistributedLock(lock_store.client(), "x", ttl=5).acquire()

        with pytest.raises(LockNotAcquired) as raised:
            async with distributed_lock("x", ttl=5):
                pass
        assert is_transient(raised.value)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def draft(sessions: async_sessionmaker[AsyncSession]) -> DocumentVersion:
    document = Document(
        id=uuid.uuid4(),
        title="Leave Policy",
        mime_type="application/pdf",
        file_size_bytes=1,
        file_hash=uuid.uuid4().hex * 2,
        storage_key="original/x.pdf",
    )
    version = DocumentVersion(
        id=uuid.uuid4(),
        document_id=document.id,
        status=VersionStatus.DRAFT.value,
        parser_name="test",
    )
    async with sessions() as session:
        session.add(document)
        await session.flush()
        session.add(version)
        await session.commit()
    return version


async def publish(sessions: async_sessionmaker[AsyncSession], version_id: uuid.UUID) -> Any:
    async with sessions() as session:
        return await pipeline.publish_version(None, session, version_id)  # type: ignore[arg-type]


async def status_of(sessions: async_sessionmaker[AsyncSession], version_id: uuid.UUID) -> str:
    async with sessions() as session:
        version = await session.get(DocumentVersion, version_id)
        assert version is not None
        return version.status


class TestVersionActivation:
    async def test_two_concurrent_activations_activate_exactly_once(
        self, sessions: async_sessionmaker[AsyncSession], draft: DocumentVersion
    ) -> None:
        outcomes = await asyncio.gather(publish(sessions, draft.id), publish(sessions, draft.id))

        assert sorted(o.noop for o in outcomes) == [False, True]
        assert await status_of(sessions, draft.id) == VersionStatus.ACTIVE

    async def test_activation_waits_while_another_holds_the_documents_lock(
        self,
        sessions: async_sessionmaker[AsyncSession],
        draft: DocumentVersion,
        lock_store: InMemoryLockRedis,
    ) -> None:
        other = DistributedLock(
            lock_store.client(), pipeline.activation_lock_name(draft.document_id), ttl=5
        )
        await other.acquire()

        attempt = asyncio.create_task(publish(sessions, draft.id))
        await asyncio.sleep(0.4)
        assert not attempt.done()
        assert await status_of(sessions, draft.id) == VersionStatus.DRAFT

        await other.release()
        outcome = await attempt
        assert not outcome.noop
        assert await status_of(sessions, draft.id) == VersionStatus.ACTIVE

    async def test_an_activation_that_cannot_get_the_lock_fails_transiently(
        self,
        sessions: async_sessionmaker[AsyncSession],
        draft: DocumentVersion,
        lock_store: InMemoryLockRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(pipeline, "ACTIVATION_LOCK_WAIT", 0.1)
        await DistributedLock(
            lock_store.client(), pipeline.activation_lock_name(draft.document_id), ttl=5
        ).acquire()

        with pytest.raises(LockNotAcquired) as raised:
            await publish(sessions, draft.id)

        assert is_transient(raised.value)  # the job retries it with backoff
        assert await status_of(sessions, draft.id) == VersionStatus.DRAFT


@pytest.fixture
def jobs(
    sessions: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> JobService:
    url = f"sqlite+aiosqlite:///{tmp_path / 'locks.db'}"
    monkeypatch.setattr(job_task_module, "worker_engine_factory", lambda: create_async_engine(url))
    return JobService(sessions)


async def queued(jobs: JobService) -> uuid.UUID:
    return (await jobs.create(task_type=JobType.DIAGNOSTIC, queue="ingestion")).id


class TestDuplicateProcessing:
    async def test_two_deliveries_of_one_job_run_its_body_once(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        runs = 0

        async def body(ctx: JobContext) -> None:
            nonlocal runs
            runs += 1
            await asyncio.sleep(0.3)

        results = await asyncio.gather(
            execute_job(body, job_id, "a", {}),
            execute_job(body, job_id, "b", {}),
            return_exceptions=True,
        )

        assert runs == 1
        assert sorted(type(r).__name__ for r in results) == ["RetryLater", "str"]
        job = await jobs.get(job_id)
        assert (job.status, job.attempt) == (JobStatus.SUCCEEDED, 1)

    async def test_a_delivery_after_a_crash_waits_out_the_dead_workers_lock(
        self, jobs: JobService, lock_store: InMemoryLockRedis
    ) -> None:
        job_id = await queued(jobs)
        # The previous worker died holding the job lock; its TTL has not run out.
        await DistributedLock(lock_store.client(), f"job:{job_id}", ttl=0.2).acquire()

        async def body(ctx: JobContext) -> None:
            return None

        with pytest.raises(RetryLater):
            await execute_job(body, job_id, "w", {})
        assert (await jobs.get(job_id)).status == JobStatus.QUEUED  # untouched

        await asyncio.sleep(0.25)
        assert await execute_job(body, job_id, "w", {}) == "succeeded"

    async def test_redis_down_defers_the_delivery_without_touching_the_job(
        self, jobs: JobService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Down:
            async def set(self, *args: Any, **kwargs: Any) -> None:
                raise redis.exceptions.ConnectionError("Redis unreachable")

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(locks, "lock_client_factory", Down)
        job_id = await queued(jobs)

        async def body(ctx: JobContext) -> None:
            raise AssertionError("must not run")

        with pytest.raises(RetryLater):
            await execute_job(body, job_id, "w", {})
        job = await jobs.get(job_id)
        assert (job.status, job.attempt) == (JobStatus.QUEUED, 0)


class TestSingleInstance:
    async def test_overlapping_runs_of_a_maintenance_job_run_once(self, jobs: JobService) -> None:
        runs = 0

        @single_instance("sweep")
        async def sweep(ctx: JobContext) -> None:
            nonlocal runs
            runs += 1
            await asyncio.sleep(0.3)

        first, second = await queued(jobs), await queued(jobs)
        outcomes = await asyncio.gather(
            execute_job(sweep, first, "a", {}), execute_job(sweep, second, "b", {})
        )

        assert outcomes == ["succeeded", "succeeded"]
        assert runs == 1
        messages = {(await jobs.get(j)).progress_message for j in (first, second)}
        assert "skipped: sweep is already running elsewhere" in messages

    async def test_a_lock_error_inside_the_body_is_not_swallowed(self, jobs: JobService) -> None:
        @single_instance("sweep")
        async def sweep(ctx: JobContext) -> None:
            raise LockNotAcquired("from the body's own lock")

        job_id = await queued(jobs)
        with pytest.raises(RetryLater):  # transient: retried, not reported as a skip
            await execute_job(sweep, job_id, "w", {})
